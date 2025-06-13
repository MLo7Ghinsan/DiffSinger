import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.commons.common_layers import SinusoidalPosEmb
from modules.commons.rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from modules.backbones.lynxnet import LYNXNetResidualLayer
from utils.hparams import hparams

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor

class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True)
        return x * (self.scale / (norm + self.eps))

class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims
    def forward(self, x):
        return x.transpose(*self.dims)

class FFN(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self attention with optional rotary embeddings."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, use_rope: bool = False):
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dim = dim
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.to_qkv(x)
        q, k, v = torch.split(qkv, [self.dim, self.dim, self.dim], dim=-1)
        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            pos = torch.arange(t, device=x.device, dtype=x.dtype)
            freqs = self.rotary(pos, seq_len=t)
            freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1,1,t,d]
            q = apply_rotary_emb(freqs, q, seq_dim=2)
            k = apply_rotary_emb(freqs, k, seq_dim=2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        out = self.out_proj(out)
        return out

class DiTConVBlock(nn.Module):
    """Diffusion Transformer convolutional block with AdaLN conditioning."""
    def __init__(self, dim: int, dim_cond: int, num_heads: int = 8, dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        self.dim = dim
        self.norm1 = ScaleNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout, use_rope=use_rope)
        self.norm2 = ScaleNorm(dim)
        self.ffn = FFN(dim, dropout=dropout)

        self.cond_proj = nn.Conv1d(dim_cond, dim * 4, 1)
        self.diff_proj = nn.Linear(dim, dim * 4)
        self.layer_scale = nn.Parameter(1e-2 * torch.ones(dim))
        self.drop_path = DropPath(0.1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        # cond: [B, H, T], diffusion_step: [B, dim]
        style = self.cond_proj(cond).transpose(1, 2) + self.diff_proj(diffusion_step).unsqueeze(1)
        style = torch.clamp(style, min=-5.0, max=5.0) # avoiding crazy modulations
        gamma1, beta1, gamma2, beta2 = torch.split(style, [self.dim, self.dim, self.dim, self.dim], dim=-1)

        h = self.norm1(x)
        h = h * (1 + gamma1) + beta1
        h = self.attn(h)
        x = x + h

        h = self.norm2(x)
        h = h * (1 + gamma2) + beta2
        h = self.ffn(h)
        h = h * self.layer_scale
        x = x + self.drop_path(0.5 * h)
        return x

class DiTLynxFusionBlock(nn.Module):
    def __init__(self, dim: int, dim_cond: int, num_heads: int = 8, dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        self.dit_block = DiTConVBlock(dim, dim_cond, num_heads, dropout, use_rope)
        self.lynx_block = LYNXNetResidualLayer(
            dim_cond=dim_cond,
            dim=dim,
            expansion_factor=int(2.0),
            kernel_size=31,
            activation='PReLU',
            dropout=dropout
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.drop_path = DropPath(0.1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, diffusion_step: torch.Tensor) -> torch.Tensor:
        x_dit = self.dit_block(x, cond, diffusion_step)
        x_lynx = self.lynx_block(
            x.transpose(1, 2),
            cond,
            diffusion_step.unsqueeze(-1).repeat(1, 1, x.shape[1])
        ).transpose(1, 2)
        x_proj = self.fusion_proj(torch.cat([x_dit, x_lynx], dim=-1))
        return x + self.drop_path(x_proj)

class DiffusionTransformer(nn.Module):
    def __init__(self, in_dims: int, n_feats: int, *, num_layers: int = 6, num_channels: int = 512,
                 num_heads: int = 8, dropout: float = 0.1, use_rope: bool = False):
        super().__init__()
        self.in_dims = in_dims
        self.n_feats = n_feats
        self.input_projection = nn.Linear(in_dims * n_feats, num_channels)
        self.diffusion_embedding = SinusoidalPosEmb(num_channels)
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels * 4),
            nn.GELU(),
            nn.Linear(num_channels * 4, num_channels),
        )
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if hparams.get("fusion_style", "parallel") == "parallel":
                self.layers.append(
                    DiTLynxFusionBlock(
                        dim=num_channels,
                        dim_cond=hparams['hidden_size'],
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=use_rope
                    )
                )
            elif hparams["fusion_style"] == "alternate":
                if i % 2 == 0:
                    self.layers.append(
                        DiTConVBlock(
                            dim=num_channels,
                            dim_cond=hparams['hidden_size'],
                            num_heads=num_heads,
                            dropout=dropout,
                            use_rope=use_rope
                        )
                    )
                else:
                    self.layers.append(
                        LYNXNetResidualLayer(
                            dim_cond=hparams['hidden_size'],
                            dim=num_channels,
                            expansion_factor=1.5,
                            kernel_size=15,
                            activation='PReLU',
                            dropout=dropout
                        )
                    )
            elif hparams["fusion_style"] == "dit-only":
                self.layers.append(
                    DiTConVBlock(
                        dim=num_channels,
                        dim_cond=hparams['hidden_size'],
                        num_heads=num_heads,
                        dropout=dropout,
                        use_rope=use_rope
                    )
                )
            elif hparams["fusion_style"] == "lynx-only":
                self.layers.append(
                    LYNXNetResidualLayer(
                        dim_cond=hparams['hidden_size'],
                        dim=num_channels,
                        expansion_factor=1.5,
                        kernel_size=15,
                        activation='PReLU',
                        dropout=dropout
                    )
                )
        self.layer_norm = nn.LayerNorm(num_channels)
        self.output_projection = nn.Linear(num_channels, in_dims * n_feats)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, spec: torch.Tensor, diffusion_step: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if self.n_feats == 1:
            x = spec[:, 0].transpose(1, 2)
        else:
            x = spec.flatten(start_dim=1, end_dim=2).transpose(1, 2)
        x = self.input_projection(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step) * 0.5 

        for i, layer in enumerate(self.layers):
            if isinstance(layer, (DiTConVBlock, DiTLynxFusionBlock)):
                x = layer(x, cond, diffusion_step)
            elif isinstance(layer, LYNXNetResidualLayer):
                x = layer(
                    x.transpose(1, 2),
                    cond,
                    diffusion_step.unsqueeze(-1).repeat(1, 1, x.shape[1])
                ).transpose(1, 2)
            if torch.isnan(x).any():
                print(f"NaN detected after layer {i}")
                break #debug

        x = self.layer_norm(x)
        x = self.output_projection(x)
        x = x.transpose(1, 2)

        if self.n_feats == 1:
            x = x[:, None, :, :]
        else:
            x = x.reshape(-1, self.n_feats, self.in_dims, x.shape[2])
        return x
