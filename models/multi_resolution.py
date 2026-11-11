"""Pretrained 1D U-Net wrapped with an input and an output module to work at higher resolutions.

The backbone (SongsUNet1D) only accepts [B, 1, backbone_points]. To generate functions sampled at
L = backbone_points * 2^n points:

    INPUT MODULE   x [B, 1, L] -> conv_in (stride 1) -> n stride-2 convs -> z [B, 1, backbone_points]
    BACKBONE       z -> v [B, 1, backbone_points]
    OUTPUT MODULE  v -> n stride-2 transposed convs (+ skip connections) -> conv_out (stride 1) -> v [B, 1, L]

The skip connections are needed because the backbone only sees backbone_points numbers, while the
noise x0 has L independent values: tau says how much noise there is, not which noise is at each point.
"""
import math

import torch
import torch.nn as nn

from models.unet_adaptation import Dense, GaussianFourierProjection


class MultiResolutionUNet1D(nn.Module):
    """
    x: (B, L) or (B, 1, L)    tau: (B,) or (B, 1)    with L = backbone_points * 2^n, n >= 1
    Returns the velocity with the same shape as x.

    Shapes for L = 1024 (n = 2) and C = hidden_channels:
        conv_in            [B, 1, 1024]  -> [B, C, 1024]   (skip 0)
        down_convs[0]      [B, C, 1024]  -> [B, C, 512]    (skip 1)
        conv_to_backbone   [B, C, 512]   -> [B, 1, 256]
        backbone           [B, 1, 256]   -> [B, 1, 256]
        up_tconvs[0]       [B, 1, 256]   -> [B, C, 512]    + skip 1 -> [B, 2C, 512]
        up_tconvs[1]       [B, 2C, 512]  -> [B, C, 1024]   + skip 0 -> [B, 2C, 1024]
        conv_out           [B, 2C, 1024] -> [B, 1, 1024]
    The number of down/up layers depends on L, so they are kept in lists (nn.ModuleList).
    """

    def __init__(self, backbone, n_points, backbone_points=256, hidden_channels=32,
                 kernel_size=13, embed_dim=256, fourier_scale=30.0):
        super().__init__()

        n_layers = round(math.log2(n_points / backbone_points))        # stride-2 layers per side

        assert n_layers >= 1 and n_points == backbone_points * 2 ** n_layers, \
            f"n_points must be {backbone_points} * 2^n with n >= 1, got {n_points}"
        
        C, k, p = hidden_channels, kernel_size, (kernel_size - 1) // 2

        self.backbone = backbone
        self.n_layers = n_layers

        # own time embedding, independent of the backbone's one
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim, scale=fourier_scale),  # (B, embed_dim)
            nn.Linear(embed_dim, embed_dim),                            # (B, embed_dim)
        )

        # ---- Input module (L -> L -> L/2 -> ... -> backbone_points) ----
        # stride-1 layer at full resolution: its output is the first skip connection
        self.conv_in = nn.Conv1d(1, C, k, stride=1, padding=p, bias=False)
        self.dense_in = Dense(embed_dim, C)
        self.gnorm_in = nn.GroupNorm(4, C)

        # stride-2 layers that halve the resolution (all but the last one)
        self.down_convs = nn.ModuleList([nn.Conv1d(C, C, k, stride=2, padding=p, bias=False) for _ in range(n_layers - 1)])
        self.down_denses = nn.ModuleList([Dense(embed_dim, C) for _ in range(n_layers - 1)])
        self.down_gnorms = nn.ModuleList([nn.GroupNorm(4, C) for _ in range(n_layers - 1)])

        # last stride-2 layer: 1 channel, because the backbone needs [B, 1, backbone_points]
        self.conv_to_backbone = nn.Conv1d(C, 1, k, stride=2, padding=p)

        # ---- Output module (backbone_points -> ... -> L/2 -> L -> L) ----
        # stride-2 transposed layers that double the resolution. The first one takes the backbone
        # output (1 channel); the next ones also take the skip connection of the same resolution (2C)
        self.up_tconvs = nn.ModuleList([
            nn.ConvTranspose1d(1 if i == 0 else 2 * C, C, k, stride=2, padding=p, output_padding=1, bias=False)
            for i in range(n_layers)
        ])
        self.up_denses = nn.ModuleList([Dense(embed_dim, C) for _ in range(n_layers)])
        self.up_gnorms = nn.ModuleList([nn.GroupNorm(4, C) for _ in range(n_layers)])

        # stride-1 layer at full resolution, with the first skip connection: the velocity (no activation)
        self.conv_out = nn.Conv1d(2 * C, 1, k, stride=1, padding=p)

        self.act = nn.SiLU()

    def set_backbone_trainable(self, trainable):
        """Freezes or unfreezes the backbone. Its random Fourier features W stay fixed, as in its training."""
        fixed = {id(m.W) for m in self.backbone.modules() if isinstance(m, GaussianFourierProjection)}
        for param in self.backbone.parameters():
            param.requires_grad_(trainable and id(param) not in fixed)

    def forward(self, x, tau):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)                          # (B, L) -> (B, 1, L)
        tau = tau.reshape(-1)                           # (B,)

        embed = self.act(self.embed(tau))               # (B, embed_dim)

        # Input module: the output of every resolution is kept for the skip connections
        h = self.act(self.gnorm_in(self.conv_in(x) + self.dense_in(embed)))           # (B, C, L)
        skips = [h]
        for conv, dense, gnorm in zip(self.down_convs, self.down_denses, self.down_gnorms):
            h = self.act(gnorm(conv(h) + dense(embed)))                                # (B, C, L/2), (B, C, L/4), ...
            skips.append(h)
        z = self.conv_to_backbone(h)                                                    # (B, 1, 256)

        # Pretrained backbone at its own resolution
        h = self.backbone(z, tau)                                                       # (B, 1, 256)

        # Output module: upsample and concatenate the skip connection of the same resolution
        for tconv, dense, gnorm in zip(self.up_tconvs, self.up_denses, self.up_gnorms):
            h = self.act(gnorm(tconv(h) + dense(embed)))                               # (B, C, 2 * previous length)
            h = torch.cat([h, skips.pop()], dim=1)                                      # (B, 2C, same length)
        v = self.conv_out(h)                                                            # (B, 1, L)

        return v.squeeze(1) if squeeze else v
