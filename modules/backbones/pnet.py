import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from utils.hparams import hparams


class ConvBlock(nn.Module):
    """Depthwise separable convolution block with GLU."""
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with [B, C, T] tensor."""
        x = self.conv(x)
        x = F.glu(x, dim=1)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        return x


class FiLMLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x, cond):
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        x = x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return x


class GlobalSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        return residual + self.dropout(x)


class PNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=None, num_channels=None):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats

        num_layers = num_layers or hparams['num_layers']
        num_channels = num_channels or hparams['hidden_size']

        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)

        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.SiLU(),
            nn.Linear(num_channels * 4, num_channels),
        )

        self.film = FiLMLayer(num_channels)

        self.note_proj = nn.Conv1d(1, num_channels, 1)
        self.deviation_proj = nn.Conv1d(1, num_channels, 1)
        # residual projection to match output dims
        self.note_res_proj = nn.Conv1d(1, in_dims * n_feats, 1)
        kernel_size = hparams['ffn_kernel_size']
        dropout = hparams['dropout']
        
        self.conv_blocks = nn.ModuleList([
            ConvBlock(num_channels, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_layers // 2)
        ])

        self.global_attn = GlobalSelfAttention(num_channels, heads=hparams['num_heads'], dropout=dropout)

        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.xavier_uniform_(self.output_projection.weight)

        self.vibrato_proj = nn.Conv1d(2, num_channels, 1)

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

        if self.n_feats == 1:
            x = spec[:, 0].transpose(1, 2)
        else:
            x = spec.flatten(start_dim=1, end_dim=2).transpose(1, 2)
        x = self.input_projection(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        diffusion_step = torch.clamp(diffusion_step, -5.0, 5.0)

        x = self.film(x, diffusion_step)

        x = x + self.note_proj(note_midi).transpose(1, 2)
        x = x + self.deviation_proj(pitch_deviation).transpose(1, 2)

        if self.use_vibrato_hint and vibrato_hint is not None:
            vib = self.vibrato_proj(vibrato_hint)
            vib = vib.transpose(1, 2)
            x = x + vib

        for block in self.conv_blocks:
            x = x + block(x.transpose(1, 2)).transpose(1, 2)

        x = self.global_attn(x)

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
