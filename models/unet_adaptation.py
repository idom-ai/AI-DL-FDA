import torch
import torch.functional as F
import math
import torch.nn as nn 

class GaussianFourierProjection(nn.Module):
    """ 
    
    Time embedding using fixed random Fourier features.
    
    Just a way in order to embed the time with learnable projection. 
    
    """
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t[:, None] * self.W[None, :] * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class Dense(nn.Module):
    """
    Projects the time embedding to shape (B, C, 1) so it can be added to 1D feature maps.
    
    Note that Song needed to project the time embedding to (B, C, 1, 1), with this two last 
    tensor dimension corresponding to image pixels as a form of "casting". This way, the time 
    embeddings were a scalar for each channel of each image of the batch and needed to be added 
    to each pixel (same value), corresponding to its noise level. 

    For us, this is the same way, excepting we need to add one last dimension to the tensor, 
    the one corresponding to each sample of the function. 

    This projection is also learnable.
    
    """
    def __init__(self, input_dim, output_dim,):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


class SongsUNet1D(nn.Module):
    
    """
    
    Time-dependent 1D U-Net for Flow Matching: (x(t, tau), tau) -> v_theta(x_t, t).

    x: (B, L) or (B, C, L)    t: (B,) or (B, 1), with t in [0, 1]
    Returns a tensor with the same shape as x.
    """
    def __init__(self, in_channels=1, channels=[32, 64, 128, 256], embed_dim=256, fourier_scale=30.0, kernel_size=7):

        super().__init__()

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim, scale=fourier_scale),
            nn.Linear(embed_dim, embed_dim),
        )

        self.padding = (kernel_size - 1) // 2

        # Encoder (L -> L -> L/2 -> L/4 -> L/8)
        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size, stride=1, padding=self.padding, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, channels[0])
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, stride=2, padding=self.padding, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, channels[1])
        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size, stride=2, padding=self.padding, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, channels[2])
        self.conv4 = nn.Conv1d(channels[2], channels[3], kernel_size, stride=2, padding=self.padding, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, channels[3])

        # Decoder (L/8 -> L/4 -> L/2 -> L -> L)
        self.tconv4 = nn.ConvTranspose1d(channels[3], channels[2], kernel_size, stride=2, padding=self.padding, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, channels[2])

        self.tconv3 = nn.ConvTranspose1d(channels[2] + channels[2], channels[1], kernel_size, stride=2, padding=self.padding, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] + channels[1], channels[0], kernel_size, stride=2, padding=self.padding, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, channels[0])
        
        self.tconv1 = nn.Conv1d(channels[0] + channels[0], in_channels, kernel_size, stride=1, padding=self.padding)

        self.act = nn.SiLU()

    def forward(self, x, t):
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)                      # (B, L) -> (B, 1, L), now it has channels
        t = t.reshape(-1)                           # (B,)
        L = x.shape[-1]

        # L (the number of samples of the function) must be a multiple of 8 (3 downsamplings): 
        # pad here and crop at the end

        pad = (-L) % 8 
        # if pad is not null, we need to add padding 
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")

        embed = self.act(self.embed(t))

        # Encoder
        # the encoder is responsible of:
            # finding a latent space by augmenting the number of channels and decreasing the sampling of functions
            # adding the time-embedding on each step, so the noise-level is remembered on each normalization
            # normalizing with learnable parameters so that the functions are always on the same scale as other functions of the batch
        h1 = self.act(self.gnorm1(self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))
        h4 = self.act(self.gnorm4(self.conv4(h3) + self.dense4(embed)))

        # Decoder with skip connections
        # the decoder is responsible of:
            # upsampling the data in the latent space so that the returned velocity-vector is the same length as functions
            # implementing skip connections: add the data from the corresponding h on the decoder 
            # normalizing, as the encoder 
        h = self.act(self.tgnorm4(self.tconv4(h4) + self.dense5(embed)))
        h = self.act(self.tgnorm3(self.tconv3(torch.cat([h, h3], dim=1)) + self.dense6(embed)))
        h = self.act(self.tgnorm2(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense7(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))  # no division by std (output is a velocity)

        h = h[..., :L]                              # remove the padding
        return h.squeeze(1) if squeeze else h