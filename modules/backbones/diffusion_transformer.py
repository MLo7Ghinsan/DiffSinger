import torch
import torch.nn as nn

from modules.commons.common_layers import SinusoidalPosEmb
from modules.commons.rotary_embedding_torch import RotaryEmbedding
from modules.fastspeech.tts_modules import TransformerEncoderLayer
from utils.hparams import hparams


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim_cond, dim, num_heads=8, ffn_kernel_size=3, act='gelu', dropout=0.1, use_rope=False):
        super().__init__()
        self.diffusion_projection = nn.Linear(dim, dim)
        self.conditioner_projection = nn.Linear(dim_cond, dim)
        rotary = RotaryEmbedding(dim // num_heads) if use_rope else None
        self.transformer = TransformerEncoderLayer(
            dim, dropout, kernel_size=ffn_kernel_size, act=act,
            num_heads=num_heads, rotary_embed=rotary
        )

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(1)
        conditioner = self.conditioner_projection(conditioner)
        x = x + diffusion_step + conditioner
        mask = torch.zeros(x.size(0), x.size(1), dtype=torch.bool, device=x.device)
        x = self.transformer(x, encoder_padding_mask=mask)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, in_dims, n_feats, *, num_layers=6, num_channels=512, num_heads=8,
                 ffn_kernel_size=3, act='gelu', dropout=0.1, use_rope=False):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels)
        )
        self.layers = nn.ModuleList([
            DiffusionTransformerLayer(
                dim_cond=hparams['hidden_size'],
                dim=num_channels,
                num_heads=num_heads,
                ffn_kernel_size=ffn_kernel_size,
                act=act,
                dropout=dropout,
                use_rope=use_rope
            )
            for _ in range(num_layers)
        ])
        self.layer_norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, spec, diffusion_step, cond):
        """Forward pass.

        Args:
            spec: [B, F, M, T]
            diffusion_step: [B, 1]
            cond: [B, H, T]
        Returns:
            [B, F, M, T]
        """
        if self.n_feats == 1:
            x = spec[:, 0].transpose(1, 2)
        else:
            x = spec.flatten(start_dim=1, end_dim=2).transpose(1, 2)
        x = self.input_projection(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)
        cond = cond.transpose(1, 2)
        for layer in self.layers:
            x = layer(x, cond, diffusion_step)
        x = self.layer_norm(x)
        x = self.output_projection(x)
        x = x.transpose(1, 2)
        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
