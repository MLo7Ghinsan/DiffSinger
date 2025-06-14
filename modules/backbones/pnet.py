import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from modules.commons.rotary_embedding_torch import apply_rotary_emb
from utils.hparams import hparams


class ConvBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv = nn.Conv1d(
            channels, channels * 2, kernel_size,
            padding=kernel_size // 2, groups=channels
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.conv(x)
        x = F.glu(x, dim=1)
        return self.dropout(x)


class ExpressionFiLM(nn.Module):
    def __init__(self, dim, extra_cond_dim=0):
        super().__init__()
        self.linear = nn.Linear(dim + extra_cond_dim, dim * 2)

    def forward(self, x, cond, extra=None):
        if extra is not None:
            cond = torch.cat([cond, extra], dim=-1)
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        return x * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class MultiScaleDiffusionEmbedding(nn.Module):
    def __init__(self, dim, scales=[10, 100, 1000]):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.SiLU(),
                nn.Linear(dim * 4, dim)
            ) for _ in scales
        ])
        self.scales = scales

    def forward(self, step):
        results = []
        for emb, scale in zip(self.embeddings, self.scales):
            scaled_step = step / scale
            results.append(emb(scaled_step))
        return torch.cat(results, dim=-1)


class ResidualGate(nn.Module):
    def __init__(self, in_dim, res_channels):
        super().__init__()
        self.proj = nn.Conv1d(1, res_channels, 1)
        self.gate_proj = nn.Linear(in_dim, res_channels)

    def forward(self, x, residual):
        # x: [B, C, T] | residual: [B, 1, T]
        B, C, T = x.shape
        x_t = x.transpose(1, 2)  # [B, T, C]
        gate_in = x_t.reshape(B * T, C)  # [B*T, C]
        gate = torch.sigmoid(self.gate_proj(gate_in))  # [B*T, res_channels]
        gate = gate.reshape(B, T, C).transpose(1, 2)  # [B, C, T]
        residual = self.proj(residual)  # [B, C, T]
        return x + gate * residual


class StochasticBlock(nn.Module):
    def __init__(self, module, drop_prob=0.1):
        super().__init__()
        self.module = module
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        return self.module(x)


class GlobalSelfAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return x + self.dropout(out)


class PNet(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=None, num_channels=None,
                 num_heads=None, dropout=None, ffn_kernel_size=None):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.note_res_gate = None

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
        self.multi_scale_mlp = MultiScaleDiffusionEmbedding(num_channels)
        film_input_dim = num_channels * len(self.multi_scale_mlp.scales)

        self.film_layers = nn.ModuleList([
            ExpressionFiLM(num_channels, extra_cond_dim=film_input_dim - num_channels)
            for _ in range(num_layers // 2)
        ])

        self.note_proj = nn.Conv1d(1, num_channels, 1)
        self.deviation_proj = nn.Conv1d(1, num_channels, 1)
        self.note_res_gate = ResidualGate(in_dim=num_channels, res_channels=num_channels)
        self.vibrato_proj = nn.Conv1d(2, num_channels, 1)

        self.conv_blocks = nn.ModuleList([
            StochasticBlock(
                ConvBlock(num_channels, kernel_size=ffn_kernel_size, dropout=dropout),
                drop_prob=0.1
            ) for _ in range(num_layers // 2)
        ])

        self.global_attn = GlobalSelfAttention(num_channels, heads=num_heads, dropout=dropout)

        self.mix_proj = nn.Conv1d(num_channels, num_channels, 1)
        self.norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, spec: torch.Tensor, diffusion_step: torch.Tensor, cond: torch.Tensor, vibrato_hint: torch.Tensor = None) -> torch.Tensor:
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
        diffusion_step = self.multi_scale_mlp(diffusion_step)
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

        B = x.shape[0]
        T = x.shape[-1]
        if self.n_feats == 1:
            x_gate = x[:, 0, :, :]  # [B, in_dims, T]
        else:
            x_gate = x.reshape(B, self.n_feats * self.in_dims, T)  # [B, C, T]

        if self.note_res_gate is None or self.note_res_gate.gate_proj.in_features != x_gate.shape[1]:
            self.note_res_gate = ResidualGate(in_dim=x_gate.shape[1], res_channels=x_gate.shape[1]).to(x_gate.device)
          
        x_gate = self.note_res_gate(x_gate, note_midi)  # [B, C, T]

        if self.n_feats == 1:
            x = x_gate[:, None, :, :]
        else:
            x = x_gate.reshape(B, self.n_feats, self.in_dims, T)
        return x
