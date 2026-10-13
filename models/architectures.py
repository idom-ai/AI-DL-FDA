"""BLOCK 2: different kinds of neural networks arquitectures in order to learn 
the velocity vector v(x, t)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityNet(nn.Module):

    ''' The very first implementation of the velocity net. It has no time (tau)
        embedding nor convolutional layers, which it may be the reason why it 
        doesn't work. It has 2 hidden layers and SiLU activation function. '''
    
    def __init__(self, dim, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden),   
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),      
        )
 
    def forward(self, x, t):             
        return self.mlp(torch.cat([x, t], dim=1))

class VelocityNetPooling(nn.Module):
    """
    Same MLP as before, but the borders are learned instead of being a fixed padding.

    The last Linear outputs dim + 2 * pool_pad values (pool_pad = pool_size // 2):
      - the first pool_pad values play the role of the left padding,
      - the next dim values are the raw output,
      - the last pool_pad values play the role of the right padding.
    AvgPool1d (stride=1, no padding) shortens the length by pool_size - 1 = 2 * pool_pad,
    so the final output has exactly dim points.

    `pool_size` must be odd, so there are the same number of neighbours on each side.
    """

    def __init__(self, dim, hidden=256, pool_size=5):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd"
        pool_pad = pool_size // 2

        self.mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim + 2 * pool_pad),  # dim points + learned left and right padding
            nn.AvgPool1d(pool_size, stride=1),      # [B, dim + 2*pool_pad] -> [B, dim]
        )

    def forward(self, x, t):                        # x: [B, dim], t: [B, 1]
        return self.mlp(torch.cat([x, t], dim=1))
    
"""Convolutional velocity field v(x, tau) where the shape '2 vectors of dim points' is written in the layers."""
import torch
import torch.nn as nn


class VelocityNetConvolutional(nn.Module):
    """
    
    Same network as model_conv_simple.py, but the whole thing is ONE nn.Sequential.

    Input:  x and tau glued into one flat vector of 2 * dim numbers: [x_0 ... x_dim-1, tau ... tau]
    Output: [B, dim], the velocity.

    Shapes inside the Sequential (B = batch size):
      [B, 2*dim]      flat input
      Unflatten       [B, 2, dim]       2 vectors of dim points: vector 0 = x, vector 1 = tau repeated
      Conv1d(k=1)     [B, hidden, dim]  each point processed alone (same weights at every point)
      Conv1d(k) x n   [B, hidden, dim]  each point combined with its neighbours
      Conv1d(k=1)     [B, 1, dim]
      Flatten         [B, dim]
    """

    def __init__(self, dim, hidden=64, n_layers=4, kernel_size=5, padding_mode="circular"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2

        layers = [
            nn.Unflatten(1, (2, dim)),                    # [B, 2*dim] -> [B, 2, dim] 
            nn.Conv1d(2, hidden, kernel_size=1),
        ]

        # the first conv layer applies, with the same params, the same convolution 
        # on the (x(t), tau) pairs. this forces the net to learn neibourhood relationships
        # between points instead of learn them on the long term training for many epochs. 

        # research wether the padding can be fixed as before, returning longer vectors 
        for _ in range(n_layers):
            layers += [nn.Conv1d(hidden, hidden, kernel_size, padding=padding, padding_mode=padding_mode), 
                       nn.SiLU()]

        # same here, the net will only weight, with the same parameters, neighbour points 
        # in order to force the model to give velocity vectors with some kind of correlation
  
        layers += [
            nn.Conv1d(hidden, 1, kernel_size=1),
            nn.Flatten(start_dim=1),                      # [B, 1, dim] -> [B, dim]
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x, tau):                            # x: [B, dim], tau: [B, 1]
        tau = tau.repeat(1, x.shape[1])                   # [B, dim]: tau copied once per point
        return self.net(torch.cat([x, tau], dim=1))       # [B, 2*dim] -> [B, dim]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFourierProjection(nn.Module):
    """Embedding de tiempo con features de Fourier aleatorias (fijas)."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)


class Dense(nn.Module):
    """Proyecta el embedding de t y lo deja con forma (B, C, 1) para sumarlo a mapas 1D."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


class SongsUNet1D(nn.Module):
    """U-Net 1D time-dependent for Flow Matching: (x_t, t) -> v_theta(x_t, t).

    x: (B, L) o (B, C, L)   t: (B,) o (B, 1), con t en [0, 1]
    Devuelve la misma forma que x.
    """
    def __init__(self, in_channels=1, channels=(32, 64, 128, 256), embed_dim=256, fourier_scale=30.0):
        super().__init__()
        c0, c1, c2, c3 = channels

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim, scale=fourier_scale),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoder (L -> L -> L/2 -> L/4 -> L/8)
        self.conv1 = nn.Conv1d(in_channels, c0, 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, c0)
        self.gnorm1 = nn.GroupNorm(4, c0)
        self.conv2 = nn.Conv1d(c0, c1, 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, c1)
        self.gnorm2 = nn.GroupNorm(32, c1)
        self.conv3 = nn.Conv1d(c1, c2, 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, c2)
        self.gnorm3 = nn.GroupNorm(32, c2)
        self.conv4 = nn.Conv1d(c2, c3, 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, c3)
        self.gnorm4 = nn.GroupNorm(32, c3)

        # Decoder (L/8 -> L/4 -> L/2 -> L -> L)
        self.tconv4 = nn.ConvTranspose1d(c3, c2, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, c2)
        self.tgnorm4 = nn.GroupNorm(32, c2)
        self.tconv3 = nn.ConvTranspose1d(c2 + c2, c1, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, c1)
        self.tgnorm3 = nn.GroupNorm(32, c1)
        self.tconv2 = nn.ConvTranspose1d(c1 + c1, c0, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, c0)
        self.tgnorm2 = nn.GroupNorm(32, c0)
        self.tconv1 = nn.Conv1d(c0 + c0, in_channels, 3, stride=1, padding=1)

        self.act = nn.SiLU() 

    def forward(self, x, t):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)                      # (B, L) -> (B, 1, L)
        t = t.reshape(-1)                           # (B,)
        L = x.shape[-1]

        # L debe ser múltiplo de 8 (3 downsamplings): se rellena y se recorta al final
        pad = (-L) % 8
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")

        embed = self.act(self.embed(t))

        # Encoder
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder con skip connections
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))  # sin división por std

        h = h[..., :L]                               # quita el relleno
        return h.squeeze(1) if squeeze else h