import csv
import json
import os
import pathlib

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate, signal
from scipy.signal import argrelmax, argrelmin

from basics.base_binarizer import BaseBinarizer, BinarizationError
from basics.base_pe import BasePE
from modules.fastspeech.tts_modules import LengthRegulator
from modules.pe import initialize_pe
from utils.binarizer_utils import (
    SinusoidalSmoothingConv1d,
    get_mel2ph_torch,
    get_energy_librosa,
    get_breathiness,
    get_voicing,
    get_tension_base_harmonic,
)
from utils.decomposed_waveform import DecomposedWaveform
from utils.hparams import hparams
from utils.infer_utils import resample_align_curve
from utils.pitch_utils import interp_f0
from utils.plot import distribution_to_figure

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='librosa')

os.environ["OMP_NUM_THREADS"] = "1"
VARIANCE_ITEM_ATTRIBUTES = [
    'spk_id',  # index number of dataset/speaker, int64
    'languages',  # index numbers of phoneme languages, int64[T_ph,]
    'tokens',  # index numbers of phonemes, int64[T_ph,]
    'ph_dur',  # durations of phonemes, in number of frames, int64[T_ph,]
    'midi',  # phoneme-level mean MIDI pitch, int64[T_ph,]
    'ph2word',  # similar to mel2ph format, representing number of phones within each note, int64[T_ph,]
    'mel2ph',  # mel2ph format representing number of frames within each phone, int64[T_s,]
    'note_midi',  # note-level MIDI pitch, float32[T_n,]
    'note_rest',  # flags for rest notes, bool[T_n,]
    'note_dur',  # durations of notes, in number of frames, int64[T_n,]
    'note_glide',  # flags for glides, 0 = none, 1 = up, 2 = down, int64[T_n,]
    'mel2note',  # mel2ph format representing number of frames within each note, int64[T_s,]
    'base_pitch',  # interpolated and smoothed frame-level MIDI pitch, float32[T_s,]
    'pitch',  # actual pitch in semitones, float32[T_s,]
    'uv',  # unvoiced masks (only for objective evaluation metrics), bool[T_s,]
    'energy',  # frame-level RMS (dB), float32[T_s,]
    'breathiness',  # frame-level RMS of aperiodic parts (dB), float32[T_s,]
    'voicing',  # frame-level RMS of harmonic parts (dB), float32[T_s,]
    'tension',  # frame-level tension (logit), float32[T_s,]
]
DS_INDEX_SEP = '#'

# These operators are used as global variables due to a PyTorch shared memory bug on Windows platforms.
# See https://github.com/pytorch/pytorch/issues/100358
pitch_extractor: BasePE = None
midi_smooth: SinusoidalSmoothingConv1d = None
energy_smooth: SinusoidalSmoothingConv1d = None
breathiness_smooth: SinusoidalSmoothingConv1d = None
voicing_smooth: SinusoidalSmoothingConv1d = None
tension_smooth: SinusoidalSmoothingConv1d = None

class VarianceBinarizer(BaseBinarizer):
    def __init__(self):
        super().__init__(data_attrs=VARIANCE_ITEM_ATTRIBUTES)

        self.use_glide_embed = hparams['use_glide_embed']
        glide_types = hparams['glide_types']
        assert 'none' not in glide_types, 'Type name \'none\' is reserved and should not appear in glide_types.'
        self.glide_map = {
            'none': 0,
            **{
                typename: idx + 1
                for idx, typename in enumerate(glide_types)
            }
        }

        predict_energy = hparams['predict_energy']
        predict_breathiness = hparams['predict_breathiness']
        predict_voicing = hparams['predict_voicing']
        predict_tension = hparams['predict_tension']
        self.predict_variances = predict_energy or predict_breathiness or predict_voicing or predict_tension
        self.lr = LengthRegulator().to(self.device)
        self.prefer_ds = self.binarization_args['prefer_ds']
        self.cached_ds = {}

    # from https://github.com/nnsvs/nnsvs/blob/master/nnsvs/dsp.py
    def lowpass_filter(self, x, sr, cutoff=5, N=5):
        """Lowpass filter"""
        nyquist = sr // 2
        norm_cutoff = cutoff / nyquist
        b, a = signal.butter(N, norm_cutoff, btype="low")
        if len(x) <= max(len(a), len(b)) * (N // 2 + 1):
            return x
        return signal.filtfilt(b, a, x)

    def compute_extent(self, pitch_seg):
        peak_high_pos = argrelmax(pitch_seg)[0]
        peak_low_pos = argrelmin(pitch_seg)[0]

        if len(peak_high_pos) == 1 or len(peak_low_pos) == 1:
            return np.array([-1])

        if len(peak_high_pos) < len(peak_low_pos):
            peak_low_pos = peak_low_pos[:-2]
        elif len(peak_high_pos) == len(peak_low_pos):
            peak_low_pos = peak_low_pos[:-1]

        peak_high_pitch = pitch_seg[peak_high_pos]
        peak_low_pitch = pitch_seg[peak_low_pos]

        peak_high_pos_diff = np.diff(peak_high_pos)
        peak_low_pos_diff = np.diff(peak_low_pos)

        # TODO: would probably be a bug...
        if len(peak_high_pitch) != len(peak_low_pitch) + 1:
            return np.array([-1])

        E = np.zeros(len(peak_high_pos_diff) + len(peak_low_pos_diff))
        E[0::2] = (peak_high_pitch[1:] + peak_high_pitch[:-1]) / 2 - peak_low_pitch
        E[1::2] = peak_high_pitch[1:-1] - (peak_low_pitch[1:] + peak_low_pitch[:-1]) / 2

        return E

    def nonzero_segments(self, f0):
        vuv = f0 > 0
        started = False
        s, e = 0, 0
        segments = []
        for idx in range(len(f0)):
            if vuv[idx] > 0 and not started:
                started = True
                s = idx
            elif started and (vuv[idx] <= 0):
                e = idx
                started = False
                segments.append((s, e))
            else:
                pass

        if started and vuv[-1] > 0:
            segments.append((s, len(vuv) - 1))

        return segments

    def extract_vibrato_parameters_impl(self, pitch_seg, sr):
        peak_high_pos = argrelmax(pitch_seg)[0]
        peak_low_pos = argrelmin(pitch_seg)[0]

        m_a = np.zeros(len(pitch_seg))
        m_f = np.zeros(len(pitch_seg))

        if len(peak_high_pos) != len(peak_low_pos) + 1:
            print("Warning! Probably a bug...T.T")
            print(peak_high_pos, peak_low_pos)
            return None, None, None, None

        peak_high_pos_diff = np.diff(peak_high_pos)
        peak_low_pos_diff = np.diff(peak_low_pos)

        R = np.zeros(len(peak_high_pos_diff) + len(peak_low_pos_diff))
        R[0::2] = peak_high_pos_diff
        R[1::2] = peak_low_pos_diff

        m_f_ind = np.zeros(len(R), dtype=int)
        m_f_ind[0::2] = peak_high_pos[:-1]
        m_f_ind[1::2] = peak_low_pos[:-1]
        m_f[m_f_ind] = (1 / R) * sr

        peak_high_pitch = pitch_seg[peak_high_pos]
        peak_low_pitch = pitch_seg[peak_low_pos]

        E = np.zeros(len(R))
        E[0::2] = (peak_high_pitch[1:] + peak_high_pitch[:-1]) / 2 - peak_low_pitch
        E[1::2] = peak_high_pitch[1:-1] - (peak_low_pitch[1:] + peak_low_pitch[:-1]) / 2

        m_a_ind = np.zeros(len(R), dtype=int)
        m_a_ind[0::2] = peak_low_pos
        m_a_ind[1::2] = peak_high_pos[1:-1]
        m_a[m_a_ind] = 0.5 * E

        rate = 1 / R.mean() * sr
        extent = 0.5 * E.mean()
        print(f"Rate: {rate}, Extent: {extent}")

        return R, E, m_a, m_f

    def extract_smoothed_f0(self, f0, sr, cutoff=8):
        segments = self.nonzero_segments(f0)

        f0_smooth = f0.copy()
        for s, e in segments:
            f0_smooth[s:e] = self.lowpass_filter(f0[s:e], sr, cutoff=cutoff)

        return f0_smooth

    def extract_vibrato_likelihood(self, f0_smooth, sr, win_length=32, n_fft=128, min_freq=3, max_freq=8):
        # STFT on 1st order diffference of F0
        X = np.abs(
            librosa.stft(
                np.diff(f0_smooth),
                hop_length=1,
                win_length=win_length,
                n_fft=n_fft,
                window="hann",
            )
        )
        X_norm = X / (X.sum(0) + 1e-7)

        freq_per_bin = sr / n_fft
        min_freq_bin = int(min_freq / freq_per_bin)
        max_freq_bin = int(max_freq / freq_per_bin)

        # Compute vibrato likelhiood
        St = np.abs(np.diff(X_norm, axis=0)).sum(0)
        Ft = X_norm[min_freq_bin:max_freq_bin, :].sum(0)
        vibrato_likelihood = St * Ft

        return vibrato_likelihood

    def interp_vibrato(self, m_f):
        nonzero_indices = np.where(m_f > 0)[0]
        nonzero_indices = [0] + list(nonzero_indices) + [len(m_f) - 1]
        out = np.interp(np.arange(len(m_f)), nonzero_indices, m_f[nonzero_indices])
        return out

    def extract_vibrato_parameters(self, pitch, vibrato_likelihood, sr=100, threshold=0.12,
        min_cross_count=5, min_extent=30, max_extent=150, interp_params=True, smooth_params=False,
        smooth_width=15, clip_extent=True):

        T = len(vibrato_likelihood)

        vibrato_flags = np.zeros(T, dtype=int)
        m_a = np.zeros(T)
        m_f = np.zeros(T)

        peak_high_pos = argrelmax(pitch)[0]
        peak_low_pos = argrelmin(pitch)[0]

        # iterate over every peak position
        peak_high_idx = 0
        while peak_high_idx < len(peak_high_pos):
            peak_frame_idx = peak_high_pos[peak_high_idx]

            found = False
            if vibrato_likelihood[peak_frame_idx] > threshold:
                # Initial positions for vibrato section
                start_index = peak_frame_idx
                peaks = peak_low_pos[peak_low_pos > peak_frame_idx]
                if len(peaks) > 0:
                    end_index = peaks[0]
                else:
                    peak_high_idx += 1
                    continue
                next_start_peak_high_idx = -1

                # Find a peak position that is close to the next non-speech segment
                # assuming that there's a non-speech segment right after vibrato
                # NOTE: we may want to remove this constraint
                peak_high_pos_rest = peak_high_pos[peak_high_pos > peak_frame_idx]
                for frame_idx in range(end_index, T):
                    if pitch[frame_idx] <= 0:
                        peaks = peak_high_pos_rest[peak_high_pos_rest < frame_idx]
                        if len(peaks) > 0:
                            end_index = peaks[-1]
                            next_start_peak_high_idx = (
                                len(peak_high_pos[peak_high_pos < end_index]) + 1
                            )
                        break

                # Set the search width (backward)
                search_width_backward = 0
                for frame_idx in range(start_index, 0, -1):
                    if pitch[frame_idx] <= 0:
                        peaks_backward = peak_high_pos[
                            (peak_high_pos < peak_frame_idx) & (peak_high_pos > frame_idx)
                        ]
                        if len(peaks_backward) > 0:
                            backward = peaks_backward[0]
                            search_width_backward = len(
                                peak_high_pos[
                                    (peak_high_pos > backward)
                                    & (peak_high_pos <= peak_frame_idx)
                                ]
                            )
                        break

                # Find a peak position that satisfies the following vibrato constraints
                # 1) more than 5 times crossing
                # 2) 30 ~ 150 cent oscillation
                estimate_start_index = start_index
                rate = 0
                for peak_idx in range(
                    max(peak_high_idx - search_width_backward, 0), peak_high_idx
                ):
                    if peak_high_pos[peak_idx] >= T:
                        break
                    f0_seg = pitch[peak_high_pos[peak_idx] : end_index]

                    # Check if the segment satisfies vibrato constraints
                    m = f0_seg.mean()
                    cross_count = len(np.where(np.diff(np.sign(f0_seg - m)))[0])

                    # Find the start_index so that the vibrato section has more than 5 crossing
                    E = self.compute_extent(f0_seg)
                    extent = 0.5 * E.mean()
                    having_large_deviation = ((0.5 * E) > max_extent * 2).any()
                    if (
                        cross_count >= min_cross_count
                        and cross_count >= rate
                        and extent >= min_extent
                        and extent <= max_extent
                        and not having_large_deviation
                        and (E > 0).all()
                    ):
                        rate = cross_count
                        estimate_start_index = peak_high_pos[peak_idx]

                start_index = estimate_start_index

                if rate >= min_cross_count:
                    R, E, m_a_seg, m_f_seg = self.extract_vibrato_parameters_impl(
                        pitch[start_index - 1 : end_index + 2], sr
                    )
                    if m_a_seg is None:
                        found = False
                        break
                    found = True
                    vibrato_flags[start_index:end_index] = 1

                    if interp_params:
                        m_a_seg = self.interp_vibrato(m_a_seg)
                        m_f_seg = np.clip(self.interp_vibrato(m_f_seg), 3, 8)
                    if smooth_params:
                        m_a_seg = np.convolve(
                            m_a_seg, np.ones(smooth_width) / smooth_width, mode="same"
                        )
                        m_f_seg = np.convolve(
                            m_f_seg, np.ones(smooth_width) / smooth_width, mode="same"
                        )

                    if clip_extent:
                        m_a_seg = np.clip(m_a_seg, min_extent, max_extent)
                    m_a[start_index:end_index] = m_a_seg[1:-2]
                    m_f[start_index:end_index] = m_f_seg[1:-2]

                    assert next_start_peak_high_idx > peak_high_idx
                    peak_high_idx = next_start_peak_high_idx

            if not found:
                peak_high_idx += 1

        return vibrato_flags, m_a, m_f

    def gen_sine_vibrato(self, f0, sr, m_a, m_f, scale=1.0):
        f0_gen = f0.copy()

        voiced_end_indices = np.asarray([e for _, e in self.nonzero_segments(f0)])

        for s, e in self.nonzero_segments(m_a):
            # limit vibrato rate to [3, 8] Hz
            m_f_seg = np.clip(m_f[s:e], 3, 8)
            # limit vibrato extent to [30, 150] cent
            m_a_seg = np.clip(m_a[s:e], 30, 150)

            cent = scale * m_a_seg * np.sin(2 * np.pi / sr * m_f_seg * np.arange(0, e - s))
            new_f0 = f0[s:e] * np.exp(cent * np.log(2) / 1200)
            f0_gen[s:e] = new_f0

            # NOTE: this is a hack to avoid discontinuity at the end of vibrato
            voiced_ends_next_to_vibrato = voiced_end_indices[voiced_end_indices > e]
            if len(voiced_ends_next_to_vibrato) > 0:
                voiced_end = voiced_ends_next_to_vibrato[0]
                f0_gen[s:voiced_end] = self.lowpass_filter(f0_gen[s:voiced_end], sr, cutoff=12)

        return f0_gen

    # attempt to pitch modeling as in nnsvs pitch.py
    def apply_pitch_modeling(self, f0_hz: np.ndarray, note_midi: np.ndarray, mel2note: torch.Tensor) -> np.ndarray:
        note_hz = librosa.midi_to_hz(note_midi)
        note_hz_tensor = torch.from_numpy(note_hz).to(mel2note.device)
        frame_note_hz = torch.gather(F.pad(note_hz_tensor, [1, 0], value=0), 0, mel2note).cpu().numpy()

        valid = (f0_hz > 0) & (frame_note_hz > 0)
        if valid.sum() == 0:
            return f0_hz

        portamento_margin_b = hparams['portamento_margin_beginning']
        portamento_margin_e = hparams['portamento_margin_end']
        vibrato_cutoff = hparams['vibrato_smoothing_cutoff']
        vibrato_scale = hparams['vibrato_scale']

        smoothed_f0 = self.extract_smoothed_f0(f0_hz, sr=100, cutoff=vibrato_cutoff)
        vibrato_likelihood = self.extract_vibrato_likelihood(smoothed_f0, sr=100)
        vibrato_flags, m_a, m_f = self.extract_vibrato_parameters(smoothed_f0, vibrato_likelihood, sr=100)
        vibrato_f0 = self.gen_sine_vibrato(frame_note_hz.copy(), sr=100, m_a=m_a, m_f=m_f, scale=vibrato_scale)

        f0_hz_corrected = np.copy(f0_hz)
        note_ids = mel2note.cpu().numpy()
        num_notes = note_midi.shape[0]

        for note_id in range(num_notes):
            mask = note_ids == note_id
            if not mask.any():
                continue
            idx = np.where(mask)[0]
            if len(idx) < 2:
                continue

            start, end = idx[0], idx[-1]
            length = end - start + 1

            margin_len_start = max(int(length * portamento_margin_b), 1)
            margin_len_end = max(int(length * portamento_margin_e), 1)

            blend_len = min(4, margin_len_start, margin_len_end)

            mid_start = start + margin_len_start
            mid_end = end - margin_len_end + 1

            for i in range(blend_len):
                alpha = 0.5 * (1 - np.cos(np.pi * (i / blend_len)))
                idx_blend = mid_start + i
                if idx_blend <= end:
                    f0_hz_corrected[idx_blend] = (
                        f0_hz[idx_blend] * (1 - alpha) + vibrato_f0[idx_blend] * alpha
                    )
            for i in range(blend_len):
                alpha = 0.5 * (1 - np.cos(np.pi * (i / blend_len)))
                idx_blend = mid_end - 1 - i
                if idx_blend >= start:
                    f0_hz_corrected[idx_blend] = (
                        f0_hz[idx_blend] * (1 - alpha) + vibrato_f0[idx_blend] * alpha
                    )
            if mid_end > mid_start:
                center_slice = slice(mid_start + blend_len, mid_end - blend_len)
                f0_hz_corrected[center_slice] = vibrato_f0[center_slice]
        return np.nan_to_num(f0_hz_corrected, nan=0.0, posinf=0.0, neginf=0.0)

    def load_attr_from_ds(self, ds_id, name, attr, idx=0):
        item_name = f'{ds_id}:{name}'
        item_name_with_idx = f'{item_name}{DS_INDEX_SEP}{idx}'
        if item_name_with_idx in self.cached_ds:
            ds = self.cached_ds[item_name_with_idx][0]
        elif item_name in self.cached_ds:
            ds = self.cached_ds[item_name][idx]
        else:
            ds_path = self.raw_data_dirs[ds_id] / 'ds' / f'{name}{DS_INDEX_SEP}{idx}.ds'
            if ds_path.exists():
                cache_key = item_name_with_idx
            else:
                ds_path = self.raw_data_dirs[ds_id] / 'ds' / f'{name}.ds'
                cache_key = item_name
            if not ds_path.exists():
                return None
            with open(ds_path, 'r', encoding='utf8') as f:
                ds = json.load(f)
            if not isinstance(ds, list):
                ds = [ds]
            self.cached_ds[cache_key] = ds
            ds = ds[idx]
        return ds.get(attr)

    def load_meta_data(self, raw_data_dir: pathlib.Path, ds_id, spk, lang):
        meta_data_dict = {}

        with open(raw_data_dir / 'transcriptions.csv', 'r', encoding='utf8') as f:
            for utterance_label in csv.DictReader(f):
                utterance_label: dict
                item_name = utterance_label['name']
                item_idx = int(item_name.rsplit(DS_INDEX_SEP, maxsplit=1)[-1]) if DS_INDEX_SEP in item_name else 0

                def require(attr, optional=False):
                    if self.prefer_ds:
                        value = self.load_attr_from_ds(ds_id, item_name, attr, item_idx)
                    else:
                        value = None
                    if value is None:
                        value = utterance_label.get(attr)
                    if value is None and not optional:
                        raise ValueError(f'Missing required attribute {attr} of item \'{item_name}\'.')
                    return value

                temp_dict = {
                    'ds_idx': item_idx,
                    'spk_id': self.spk_map[spk],
                    'spk_name': spk,
                    'language_id': self.lang_map[lang],
                    'language_name': lang,
                    'wav_fn': str(raw_data_dir / 'wavs' / f'{item_name}.wav'),
                    'lang_seq': [
                        (
                            self.lang_map[lang if '/' not in p else p.split('/', maxsplit=1)[0]]
                            if self.phoneme_dictionary.is_cross_lingual(p)
                            else 0
                        )
                        for p in utterance_label['ph_seq'].split()
                    ],
                    'ph_seq': self.phoneme_dictionary.encode(require('ph_seq'), lang=lang),
                    'ph_dur': [float(x) for x in require('ph_dur').split()],
                    'ph_text': require('ph_seq'),
                }

                assert len(temp_dict['ph_seq']) == len(temp_dict['ph_dur']), \
                    f'Lengths of ph_seq and ph_dur mismatch in \'{item_name}\'.'
                assert all(ph_dur >= 0 for ph_dur in temp_dict['ph_dur']), \
                    f'Negative ph_dur found in \'{item_name}\'.'

                if hparams['predict_dur']:
                    temp_dict['ph_num'] = [int(x) for x in require('ph_num').split()]
                    assert len(temp_dict['ph_seq']) == sum(temp_dict['ph_num']), \
                        f'Sum of ph_num does not equal length of ph_seq in \'{item_name}\'.'

                if hparams['predict_pitch']:
                    temp_dict['note_seq'] = require('note_seq').split()
                    temp_dict['note_dur'] = [float(x) for x in require('note_dur').split()]
                    assert all(note_dur >= 0 for note_dur in temp_dict['note_dur']), \
                        f'Negative note_dur found in \'{item_name}\'.'
                    assert len(temp_dict['note_seq']) == len(temp_dict['note_dur']), \
                        f'Lengths of note_seq and note_dur mismatch in \'{item_name}\'.'
                    assert any([note != 'rest' for note in temp_dict['note_seq']]), \
                        f'All notes are rest in \'{item_name}\'.'
                    if hparams['use_glide_embed']:
                        note_glide = require('note_glide', optional=True)
                        if note_glide is None:
                            note_glide = ['none' for _ in temp_dict['note_seq']]
                        else:
                            note_glide = note_glide.split()
                            assert len(note_glide) == len(temp_dict['note_seq']), \
                                f'Lengths of note_seq and note_glide mismatch in \'{item_name}\'.'
                            assert all(g in self.glide_map for g in note_glide), \
                                f'Invalid glide type found in \'{item_name}\'.'
                        temp_dict['note_glide'] = note_glide

                meta_data_dict[f'{ds_id}:{item_name}'] = temp_dict

        return meta_data_dict

    def check_coverage(self):
        super().check_coverage()
        if not hparams['predict_pitch']:
            return

        # MIDI pitch distribution summary
        midi_map = {}
        for item_name in self.items:
            for midi in self.items[item_name]['note_seq']:
                if midi == 'rest':
                    continue
                midi = librosa.note_to_midi(midi, round_midi=True)
                if midi in midi_map:
                    midi_map[midi] += 1
                else:
                    midi_map[midi] = 1

        print('===== MIDI Pitch Distribution Summary =====')
        for i, key in enumerate(sorted(midi_map.keys())):
            if i == len(midi_map) - 1:
                end = '\n'
            elif i % 10 == 9:
                end = ',\n'
            else:
                end = ', '
            print(f'\'{librosa.midi_to_note(key, unicode=False)}\': {midi_map[key]}', end=end)

        # Draw graph.
        midis = sorted(midi_map.keys())
        notes = [librosa.midi_to_note(m, unicode=False) for m in range(midis[0], midis[-1] + 1)]
        plt = distribution_to_figure(
            title='MIDI Pitch Distribution Summary',
            x_label='MIDI Key', y_label='Number of occurrences',
            items=notes, values=[midi_map.get(m, 0) for m in range(midis[0], midis[-1] + 1)]
        )
        filename = self.binary_data_dir / 'midi_distribution.jpg'
        plt.savefig(fname=filename,
                    bbox_inches='tight',
                    pad_inches=0.25)
        print(f'| save summary to \'{filename}\'')

        if self.use_glide_embed:
            # Glide type distribution summary
            glide_count = {
                g: 0
                for g in self.glide_map
            }
            for item_name in self.items:
                for glide in self.items[item_name]['note_glide']:
                    if glide == 'none' or glide not in self.glide_map:
                        glide_count['none'] += 1
                    else:
                        glide_count[glide] += 1

            print('===== Glide Type Distribution Summary =====')
            for i, key in enumerate(sorted(glide_count.keys(), key=lambda k: self.glide_map[k])):
                if i == len(glide_count) - 1:
                    end = '\n'
                elif i % 10 == 9:
                    end = ',\n'
                else:
                    end = ', '
                print(f'\'{key}\': {glide_count[key]}', end=end)

            if any(n == 0 for _, n in glide_count.items()):
                raise BinarizationError(
                    f'Missing glide types in dataset: '
                    f'{sorted([g for g, n in glide_count.items() if n == 0], key=lambda k: self.glide_map[k])}'
                )

    @torch.no_grad()
    def process_item(self, item_name, meta_data, binarization_args):
        ds_id, name = item_name.split(':', maxsplit=1)
        name = name.rsplit(DS_INDEX_SEP, maxsplit=1)[0]
        ds_id = int(ds_id)
        ds_seg_idx = meta_data['ds_idx']
        seconds = sum(meta_data['ph_dur'])
        length = round(seconds / self.timestep)
        T_ph = len(meta_data['ph_seq'])
        processed_input = {
            'name': item_name,
            'wav_fn': meta_data['wav_fn'],
            'spk_id': meta_data['spk_id'],
            'spk_name': meta_data['spk_name'],
            'seconds': seconds,
            'length': length,
            'languages': np.array(meta_data['lang_seq'], dtype=np.int64),
            'tokens': np.array(meta_data['ph_seq'], dtype=np.int64),
            'ph_text': meta_data['ph_text'],
        }

        ph_dur_sec = torch.FloatTensor(meta_data['ph_dur']).to(self.device)
        ph_acc = torch.round(torch.cumsum(ph_dur_sec, dim=0) / self.timestep + 0.5).long()
        ph_dur = torch.diff(ph_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))
        processed_input['ph_dur'] = ph_dur.cpu().numpy()

        mel2ph = get_mel2ph_torch(
            self.lr, ph_dur_sec, length, self.timestep, device=self.device
        )

        if hparams['predict_pitch'] or self.predict_variances:
            processed_input['mel2ph'] = mel2ph.cpu().numpy()

        # Below: extract actual f0, convert to pitch and calculate delta pitch
        if pathlib.Path(meta_data['wav_fn']).exists():
            waveform, _ = librosa.load(meta_data['wav_fn'], sr=hparams['audio_sample_rate'], mono=True)
        elif not self.prefer_ds:
            raise FileNotFoundError(meta_data['wav_fn'])
        else:
            waveform = None

        global pitch_extractor
        if pitch_extractor is None:
            pitch_extractor = initialize_pe()
        f0 = uv = None
        if self.prefer_ds:
            f0_seq = self.load_attr_from_ds(ds_id, name, 'f0_seq', idx=ds_seg_idx)
            if f0_seq is not None:
                f0 = resample_align_curve(
                    np.array(f0_seq.split(), np.float32),
                    original_timestep=float(self.load_attr_from_ds(ds_id, name, 'f0_timestep', idx=ds_seg_idx)),
                    target_timestep=self.timestep,
                    align_length=length
                )
                uv = f0 == 0
                f0, _ = interp_f0(f0, uv)
        if f0 is None:
            f0, uv = pitch_extractor.get_pitch(
                waveform, samplerate=hparams['audio_sample_rate'], length=length,
                hop_size=hparams['hop_size'], f0_min=hparams['f0_min'], f0_max=hparams['f0_max'],
                interp_uv=True
            )
        if uv.all():  # All unvoiced
            print(f'Skipped \'{item_name}\': empty gt f0')
            return None
        pitch = torch.from_numpy(librosa.hz_to_midi(f0.astype(np.float32))).to(self.device)

        if hparams['predict_dur']:
            ph_num = torch.LongTensor(meta_data['ph_num']).to(self.device)
            ph2word = self.lr(ph_num[None])[0]
            processed_input['ph2word'] = ph2word.cpu().numpy()
            mel2dur = torch.gather(F.pad(ph_dur, [1, 0], value=1), 0, mel2ph)  # frame-level phone duration
            ph_midi = pitch.new_zeros(T_ph + 1).scatter_add(
                0, mel2ph, pitch / mel2dur
            )[1:]
            processed_input['midi'] = ph_midi.round().long().clamp(min=0, max=127).cpu().numpy()

        if hparams['predict_pitch']:
            # Below: get note sequence and interpolate rest notes
            note_midi = np.array(
                [(librosa.note_to_midi(n, round_midi=False) if n != 'rest' else -1) for n in meta_data['note_seq']],
                dtype=np.float32
            )
            if hparams['use_midi_correction']:
                note_midi = np.where(note_midi >= 0, np.round(note_midi), note_midi)
            note_rest = note_midi < 0
            interp_func = interpolate.interp1d(
                np.where(~note_rest)[0], note_midi[~note_rest],
                kind='nearest', fill_value='extrapolate'
            )
            note_midi[note_rest] = interp_func(np.where(note_rest)[0])
            processed_input['note_midi'] = note_midi
            processed_input['note_rest'] = note_rest
            note_midi = torch.from_numpy(note_midi).to(self.device)

            note_dur_sec = torch.FloatTensor(meta_data['note_dur']).to(self.device)
            note_acc = torch.round(torch.cumsum(note_dur_sec, dim=0) / self.timestep + 0.5).long()
            note_dur = torch.diff(note_acc, dim=0, prepend=torch.LongTensor([0]).to(self.device))
            processed_input['note_dur'] = note_dur.cpu().numpy()

            mel2note = get_mel2ph_torch(
                self.lr, note_dur_sec, mel2ph.shape[0], self.timestep, device=self.device
            )
            processed_input['mel2note'] = mel2note.cpu().numpy()
            if hparams['use_pitch_modeling']:
                f0 = self.apply_pitch_modeling(
                    f0, note_midi.cpu().numpy(), mel2note
                )
                uv = f0 == 0
                f0, _ = interp_f0(f0, uv)
                pitch = torch.from_numpy(librosa.hz_to_midi(f0.astype(np.float32))).to(self.device)
            # Below: get ornament attributes
            if hparams['use_glide_embed']:
                processed_input['note_glide'] = np.array([
                    self.glide_map.get(x, 0) for x in meta_data['note_glide']
                ], dtype=np.int64)

            # Below:
            # 1. Get the frame-level MIDI pitch, which is a step function curve
            # 2. smoothen the pitch step curve as the base pitch curve
            frame_midi_pitch = torch.gather(F.pad(note_midi, [1, 0], value=0), 0, mel2note)
            global midi_smooth
            if midi_smooth is None:
                midi_smooth = SinusoidalSmoothingConv1d(
                    round(hparams['midi_smooth_width'] / self.timestep)
                ).eval().to(self.device)
            smoothed_midi_pitch = midi_smooth(frame_midi_pitch[None])[0]
            processed_input['base_pitch'] = smoothed_midi_pitch.cpu().numpy()

        if hparams['predict_pitch'] or self.predict_variances:
            processed_input['pitch'] = pitch.cpu().numpy()
            processed_input['uv'] = uv

        # Below: extract energy
        if hparams['predict_energy']:
            energy = None
            energy_from_wav = False
            if self.prefer_ds:
                energy_seq = self.load_attr_from_ds(ds_id, name, 'energy', idx=ds_seg_idx)
                if energy_seq is not None:
                    energy = resample_align_curve(
                        np.array(energy_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'energy_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if energy is None:
                energy = get_energy_librosa(
                    waveform, length,
                    hop_size=hparams['hop_size'], win_size=hparams['win_size']
                ).astype(np.float32)
                energy_from_wav = True

            if energy_from_wav:
                global energy_smooth
                if energy_smooth is None:
                    energy_smooth = SinusoidalSmoothingConv1d(
                        round(hparams['energy_smooth_width'] / self.timestep)
                    ).eval().to(self.device)
                energy = energy_smooth(torch.from_numpy(energy).to(self.device)[None])[0].cpu().numpy()

            processed_input['energy'] = energy

        # create a DecomposedWaveform object for further feature extraction
        dec_waveform = DecomposedWaveform(
            waveform, samplerate=hparams['audio_sample_rate'], f0=f0 * ~uv,
            hop_size=hparams['hop_size'], fft_size=hparams['fft_size'], win_size=hparams['win_size'],
            algorithm=hparams['hnsep']
        ) if waveform is not None else None

        # Below: extract breathiness
        if hparams['predict_breathiness']:
            breathiness = None
            breathiness_from_wav = False
            if self.prefer_ds:
                breathiness_seq = self.load_attr_from_ds(ds_id, name, 'breathiness', idx=ds_seg_idx)
                if breathiness_seq is not None:
                    breathiness = resample_align_curve(
                        np.array(breathiness_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'breathiness_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if breathiness is None:
                breathiness = get_breathiness(
                    dec_waveform, None, None, length=length
                )
                breathiness_from_wav = True

            if breathiness_from_wav:
                global breathiness_smooth
                if breathiness_smooth is None:
                    breathiness_smooth = SinusoidalSmoothingConv1d(
                        round(hparams['breathiness_smooth_width'] / self.timestep)
                    ).eval().to(self.device)
                breathiness = breathiness_smooth(torch.from_numpy(breathiness).to(self.device)[None])[0].cpu().numpy()

            processed_input['breathiness'] = breathiness

        # Below: extract voicing
        if hparams['predict_voicing']:
            voicing = None
            voicing_from_wav = False
            if self.prefer_ds:
                voicing_seq = self.load_attr_from_ds(ds_id, name, 'voicing', idx=ds_seg_idx)
                if voicing_seq is not None:
                    voicing = resample_align_curve(
                        np.array(voicing_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'voicing_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if voicing is None:
                voicing = get_voicing(
                    dec_waveform, None, None, length=length
                )
                voicing_from_wav = True

            if voicing_from_wav:
                global voicing_smooth
                if voicing_smooth is None:
                    voicing_smooth = SinusoidalSmoothingConv1d(
                        round(hparams['voicing_smooth_width'] / self.timestep)
                    ).eval().to(self.device)
                voicing = voicing_smooth(torch.from_numpy(voicing).to(self.device)[None])[0].cpu().numpy()

            processed_input['voicing'] = voicing

        # Below: extract tension
        if hparams['predict_tension']:
            tension = None
            tension_from_wav = False
            if self.prefer_ds:
                tension_seq = self.load_attr_from_ds(ds_id, name, 'tension', idx=ds_seg_idx)
                if tension_seq is not None:
                    tension = resample_align_curve(
                        np.array(tension_seq.split(), np.float32),
                        original_timestep=float(self.load_attr_from_ds(
                            ds_id, name, 'tension_timestep', idx=ds_seg_idx
                        )),
                        target_timestep=self.timestep,
                        align_length=length
                    )
            if tension is None:
                tension = get_tension_base_harmonic(
                    dec_waveform, None, None, length=length, domain='logit'
                )
                tension_from_wav = True

            if tension_from_wav:
                global tension_smooth
                if tension_smooth is None:
                    tension_smooth = SinusoidalSmoothingConv1d(
                        round(hparams['tension_smooth_width'] / self.timestep)
                    ).eval().to(self.device)
                tension = tension_smooth(torch.from_numpy(tension).to(self.device)[None])[0].cpu().numpy()

            processed_input['tension'] = tension

        return processed_input

    def arrange_data_augmentation(self, data_iterator):
        return {}
