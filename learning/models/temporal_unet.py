"""SafeFlowMPC's temporal U-Net (safe_flow_mpc/Models/models.py), without its unused attention layers.

A 1D U-Net over the horizon of a chunk with the action dimensions as channels. Every residual block is conditioned on
the sum of an embedding of the flow time and of the observation. The horizon has to be divisible by
``2 ** (len(dim_mults) - 1)``.
"""

import math

import torch
from torch import Tensor, nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        frequencies = torch.exp(torch.arange(half_dim, device=x.device) * -(math.log(10000) / (half_dim - 1)))
        angles = x[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    """Conv1d, GroupNorm over (channels, horizon), Mish."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.activation = nn.Mish()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(x)))


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Conv1dBlock(in_channels, out_channels, kernel_size), Conv1dBlock(out_channels, out_channels, kernel_size)]
        )
        self.time_mlp = nn.Sequential(nn.Mish(), nn.Linear(embed_dim, out_channels))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        """x (batch, in_channels, horizon), embedding (batch, embed_dim) -> (batch, out_channels, horizon)."""
        out = self.blocks[0](x) + self.time_mlp(embedding)[:, :, None]
        return self.blocks[1](out) + self.residual_conv(x)


class TemporalUnet(nn.Module):
    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        cond_dim: int,
        dim: int = 32,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        downsamplings = len(dim_mults) - 1
        if horizon % 2**downsamplings:
            raise ValueError(
                f"The horizon ({horizon}) must be divisible by 2 ** (len(dim_mults) - 1) = {2**downsamplings}: "
                "every level but the first halves it."
            )
        dims = [transition_dim, *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))
        self.cond_mlp = nn.Sequential(nn.Linear(cond_dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index == len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_in, dim_out, embed_dim=dim),
                        ResidualTemporalBlock(dim_out, dim_out, embed_dim=dim),
                        nn.Identity() if is_last else nn.Conv1d(dim_out, dim_out, 3, 2, 1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=dim)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index == len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=dim),
                        ResidualTemporalBlock(dim_in, dim_in, embed_dim=dim),
                        nn.Identity() if is_last else nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1),
                    ]
                )
            )

        self.final_conv = nn.Sequential(Conv1dBlock(dim, dim, kernel_size=5), nn.Conv1d(dim, transition_dim, 1))

    def reset(self) -> None:
        """Start of a new episode. Convolutions and norms keep no state between calls, so there is nothing to clear."""

    def forward(self, x: Tensor, time: Tensor, cond: Tensor) -> Tensor:
        """x (batch, horizon, transition), time (batch,), cond (batch, cond_dim) -> (batch, horizon, transition)."""
        x = x.transpose(1, 2)
        embedding = self.time_mlp(time) + self.cond_mlp(cond)
        skips = []
        for block1, block2, downsample in self.downs:
            x = block2(block1(x, embedding), embedding)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block2(self.mid_block1(x, embedding), embedding)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = upsample(block2(block1(x, embedding), embedding))

        return self.final_conv(x).transpose(1, 2)
