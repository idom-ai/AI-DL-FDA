"""BLOCK 2: different kinds of neural networks arquitectures in order to learn 
the velocity vector v(x, t)."""

import torch
import torch.nn as nn

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

        layers = [nn.Conv1d(2, hidden, kernel_size=1)]

        # the first conv layer applies, with the same params, the same convolution 
        # on the (x(t), tau) pairs. this forces the net to learn neibourhood relationships
        # between points instead of learn them on the long term training for many epochs. 

        # research wether the padding can be fixed as before, just by returning longer vectors 
        for _ in range(n_layers):
            layers += [nn.Conv1d(hidden, hidden, kernel_size, padding=padding, padding_mode=padding_mode), 
                       nn.SiLU()]

        # same here, the net will only weight, with the same parameters, neighbour points 
        # in order to force the model to give velocity vectors with some kind of correlation
  
        layers += [nn.Conv1d(hidden, 1, kernel_size=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, tau):                            # x: [B, dim], tau: [B, 1]
        tau = tau.expand_as(x)                            # [B, 1] -> [B, dim]
        h = torch.stack([x, tau], dim=1)                  # [B, 2, dim]: capnal 0 = x, canal 1 = tau
        return self.net(h).squeeze(1)                     # [B, 1, dim] -> [B, dim]