import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from modules.commons.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from utils.hparams import hparams


class ConvBlock(nn.Module):
    """Depthwise separable convolution block with GLU."""
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with [B, C, T] tensor."""
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        x = self.dropout(x)
        return x


class FiLMLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class GlobalSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding(dim // heads)

    def forward(self, x):
        x = self.norm(x)
        b, t, d = x.shape
        h = self.attn.num_heads
        qkv = x.view(b, t, h, -1).transpose(1, 2)
        pos = torch.arange(t, device=x.device, dtype=x.dtype)
        freqs = self.rotary(pos, seq_len=t).unsqueeze(0).unsqueeze(0)
        qkv = apply_rotary_emb(freqs, qkv, seq_dim=2)
        x = qkv.transpose(1, 2).reshape(b, t, d)
        out, _ = self.attn(x, x, x, need_weights=False)
        return x + self.dropout(out)


class PNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=None, num_channels=None,
                 num_heads=None, dropout=None, ffn_kernel_size=None):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats

        num_layers = num_layers or hparams['num_layers']
        num_channels = num_channels or hparams['hidden_size']
        num_heads = num_heads or hparams['num_heads']
        dropout = hparams['dropout'] if dropout is None else dropout
        ffn_kernel_size = (
            ffn_kernel_size
            if ffn_kernel_size is not None
            else hparams.get('ffn_kernel_size', hparams.get('enc_ffn_kernel_size'))
        )

        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)

        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.SiLU(),
            nn.Linear(num_channels * 4, num_channels),
        )

        self.film_layers = nn.ModuleList([
            FiLMLayer(num_channels) for _ in range(num_layers // 2)
        ])

        self.note_proj = nn.Conv1d(1, num_channels, 1)
        self.deviation_proj = nn.Conv1d(1, num_channels, 1)
        # residual projection to match output dims
        self.note_res_proj = nn.Conv1d(1, in_dims * n_feats, 1)
        self.vibrato_proj = nn.Conv1d(2, num_channels, 1)
        kernel_size = ffn_kernel_size

        self.conv_blocks = nn.ModuleList([
            ConvBlock(num_channels, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers // 2)
        ])

        self.global_attn = GlobalSelfAttention(num_channels, heads=num_heads, dropout=dropout)

        self.mix_proj = nn.Conv1d(num_channels, num_channels, 1)

        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, spec: torch.Tensor, diffusion_step: torch.Tensor, cond: torch.Tensor, vibrato_hint: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.
        Args:
            spec: [B, F, M, T]
            diffusion_step: [B, 1]
            cond: [B, H, T] - note_midi must be channel 0
            vibrato_hint: [B, 2, T] (freq, depth)
        """
        B, F, M, T = spec.shape
        note_midi = cond[:, :1, :]
        pitch_deviation = spec[:, 0, 0, :] - note_midi[:, 0, :]
        pitch_deviation = pitch_deviation.unsqueeze(1)

        if self.training and torch.rand(1).item() < 0.1:
            cond = cond.clone()
            cond[:, :1, :] = 0
            if vibrato_hint is not None:
                vibrato_hint = vibrato_hint * 0

        if self.n_feats == 1:
            x = spec[:, 0].transpose(1, 2)
        else:
            x = spec.flatten(start_dim=1, end_dim=2).transpose(1, 2)
        x = self.input_projection(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        diffusion_step = torch.clamp(diffusion_step, -5.0, 5.0)

        x = self.film_layers[0](x, diffusion_step)

        x = x + self.note_proj(note_midi).transpose(1, 2)
        x = x + self.deviation_proj(pitch_deviation).transpose(1, 2)

        if vibrato_hint is not None:
            vib = self.vibrato_proj(vibrato_hint).transpose(1, 2)
            x = x + vib

        for i, block in enumerate(self.conv_blocks):
            x = x + block(x.transpose(1, 2)).transpose(1, 2)
            if i + 1 < len(self.film_layers):
                x = self.film_layers[i + 1](x, diffusion_step)

        x = self.global_attn(x)
        x = self.mix_proj(x.transpose(1, 2)).transpose(1, 2)

        x = self.norm(x)
        x = self.output_projection(x)
        x = x.transpose(1, 2)

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            x = x.reshape(-1, self.n_feats, self.in_dims, T)

        note_midi_residual = self.note_res_proj(note_midi)
        note_midi_residual = note_midi_residual.reshape(B, self.n_feats, self.in_dims, T)
        x = x + note_midi_residual
        return x
