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
