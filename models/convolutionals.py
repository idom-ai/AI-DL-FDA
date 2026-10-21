import torch
import math
import torch.nn as nn
import torch.functional as F

class FirstVelocityNetConvolutional(nn.Module):
    """

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

    def __init__(self, dim, channels=64, n_layers=4, kernel_size=5, padding_mode="circular"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2

        layers = [nn.Conv1d(2, channels, kernel_size=1)]

        # the first conv layer applies, with the same params, the same convolution 
        # on the (x(t), tau) pairs. this forces the net to learn neibourhood relationships
        # between points instead of learn them on the long term training for many epochs. 

        # research wether the padding can be fixed as before, just by returning longer vectors 
        for _ in range(n_layers):
            layers += [nn.Conv1d(channels, channels, kernel_size, padding=padding, padding_mode=padding_mode), 
                       nn.SiLU()]

        # same here, the net will only weight, with the same parameters, neighbour points 
        # in order to force the model to give velocity vectors with some kind of correlation
  
        layers += [nn.Conv1d(channels, 1, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, tau):                            # x: [B, dim], tau: [B, 1]
        tau = tau.expand_as(x)                            # [B, 1] -> [B, dim]
        h = torch.stack([x, tau], dim=1)                  # [B, 2, dim]: capnal 0 = x, canal 1 = tau
        return self.net(h).squeeze(1)                     # [B, 1, dim] -> [B, dim]

class SecondVelocityNetConvolutional(nn.Module):
    """

    Minimum size of the velocity net so that it works similar to Songs U-Net. 

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

    def __init__(self, dim, channels=[4, 8, 16], kernel_size=13):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2

        # we will not have time embeddings yet, so this wil work as our embedding
        layers = [nn.Conv1d(2, channels[0], kernel_size=1)]

        # encoder 
        layers += [nn.Conv1d(channels[0], channels[1], kernel_size, padding=padding), 
                   nn.SiLU(), 
                   nn.Conv1d(channels[1], channels[2], kernel_size, padding=padding), 
                   nn.SiLU()]

        #decoder 
        layers += [nn.ConvTranspose1d(channels[2], channels[1], kernel_size=kernel_size, padding=padding), 
                   nn.SiLU(), 
                   nn.ConvTranspose1d(channels[1], channels[0], kernel_size=kernel_size, padding=padding), 
                   nn.SiLU(), 
                   nn.ConvTranspose1d(channels[0], channels[0], kernel_size=kernel_size, padding=padding), 
                   nn.SiLU()]
        
        # final layer must be convolutional
        layers += [nn.Conv1d(channels[0], 1, kernel_size, stride=1, padding=padding)]

        self.net = nn.Sequential(*layers)

    def forward(self, x, tau):                            # x: [B, dim], tau: [B, 1]
        tau = tau.expand_as(x)                            # [B, 1] -> [B, dim]
        h = torch.stack([x, tau], dim=1)                  # [B, 2, dim]: capnal 0 = x, canal 1 = tau
        return self.net(h).squeeze(1)                     # [B, 1, dim] -> [B, dim]

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

class ThirdVelocityNetConvolutional(nn.Module):
    """

    Minimum size of the velocity net so that it works similar to Songs U-Net. 

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

    def __init__(self, dim, channels=[16, 32, 64], embed_dim=64, kernel_size=13):

        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size must be odd"

        padding = kernel_size // 2

        # now we have a time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim, scale=30.0), 
            nn.Linear(embed_dim, embed_dim)
        )

        self.act = nn.SiLU()

        # encoder 
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size, padding=padding, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(2, channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size, padding=padding, bias=False) 
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(4, channels[1])

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size, padding=padding, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(4, channels[2])

        # decoder 
        self.tconv3 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size, padding=padding, bias=False)
        self.dense4 = Dense(embed_dim, channels[1])
        self.tgnorm2 = nn.GroupNorm(4, channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] + channels[1], channels[0], kernel_size, padding=padding, bias=False)
        self.dense5 = Dense(embed_dim, channels[0])
        self.tgnorm1 = nn.GroupNorm(2, channels[0])

        self.tconv1 = nn.Conv1d(channels[0] + channels[0], 1, kernel_size, padding=padding) 


    def forward(self, x, tau):                           

        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)                      # (B, L) -> (B, 1, L), now it has channels
        tau = tau.reshape(-1)                           # (B,)
        L = x.shape[-1]

        # L (the number of samples of the function) must be a multiple of 8 (3 downsamplings): 
        # pad here and crop at the end

        pad = (-L) % 8 
        # if pad is not null, we need to add padding 
        if pad:
            x = F.pad(x, (0, pad), mode="replicate")

        # create a time embedding
        embed = self.act(self.embed(tau))

        # encoder 
        #           conv(x(t, tau))  + embedding(tau)
        h1 = self.act(self.gnorm1( self.conv1(x) + self.dense1(embed)))
        h2 = self.act(self.gnorm2(self.conv2(h1) + self.dense2(embed)))
        h3 = self.act(self.gnorm3(self.conv3(h2) + self.dense3(embed)))

        h = self.act(self.tgnorm2(self.tconv3(h3) + self.dense4(embed)))
        h = self.act(self.tgnorm1(self.tconv2(torch.cat([h, h2], dim=1)) + self.dense5(embed)))
        h = self.tconv1(torch.cat([h, h1], dim=1))

        h = h[..., :L]                              # remove the padding

        return h.squeeze(1) if squeeze else h