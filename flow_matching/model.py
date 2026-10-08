"""BLOQUE 2: la red que aprende el campo de velocidad v(x, t)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VelocityNetPooling(nn.Module):
    """MLP normal, salvo la última capa: en vez de una Linear que da cada punto de
    salida por separado, un pooling de media que promedia cada punto con sus vecinos.

    El pooling NO tiene pesos entrenables: siempre calcula la media exacta de la
    ventana, sin aprender nada, a diferencia de una convolución.

    `pool_size` es el número de puntos que se tienen en cuenta alrededor de cada
    posición (debe ser impar, para que haya el mismo número de vecinos a cada lado).

    `padding_mode` controla cómo se rellenan los bordes (los vecinos "fantasma" fuera
    del dominio, necesarios para que los puntos de los extremos también tengan vecinos
    a ambos lados):
      - 'reflect'  : espejo, el valor se refleja hacia fuera
      - 'circular' : el final de la función se conecta con el principio (periódica)
      - 'replicate': repite el valor del borde
    """

    def __init__(self, dim, hidden=256, pool_size=5):
        super().__init__()
        pool_pad = pool_size // 2

        self.mlp = nn.Sequential(
            nn.Linear(dim + 1, hidden), 
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.SiLU(),
            nn.Linear(hidden, hidden), 
            nn.SiLU(),
            nn.Linear(hidden, dim),
                    
            # --- Adaptación y Pooling 100% nativo ---
            nn.Unflatten(1, (1, dim)),              # Convierte a 3D: [B, 1, dim]
            nn.CircularPad1d((pool_pad, pool_pad)), # Padding
            nn.AvgPool1d(pool_size, stride=1),      # Pooling
            nn.Flatten(start_dim=1)                 # Vuelve a 2D: [B, dim])
        )
            
    def forward(self, x, t):
        return self.mlp(torch.cat([x, t], dim=1))
    
class VelocityNetConvolutional(nn.Module):
    """Red mayormente convolucional: la función x se trata como una señal 1D a lo
    largo de sus `dim` puntos, y cada capa combina cada punto con sus vecinos
    (ventana `kernel_size`), en vez de mezclar todos los puntos a la vez como haría
    una capa densa.
 
    Solo hay 2 capas densas (Linear), al principio y al final:
      - La primera mezcla una sola vez el valor de la función con el tiempo t, por
        punto, antes de empezar a mirar vecinos.
      - La última proyecta de `hidden` canales a 1 canal (la velocidad de salida).
    Entre medias, todo son convoluciones 1D (`n_layers` capas).
 
    `kernel_size` es el número de puntos vecinos que entran en cada convolución
    (debe ser impar). `padding_mode` controla cómo se rellenan los bordes:
    'circular' (función periódica), 'reflect' (espejo) o 'replicate' (repite el borde).
    """
 
    def __init__(self, dim, hidden=64, n_layers=4, kernel_size=5, padding_mode="circular"):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size debe ser impar"
        padding = kernel_size // 2
 
        # 1ª capa densa: combina, en cada punto por separado, su valor con el tiempo tau
        self.entrada = nn.Linear(2, hidden)
 
        # bloque convolucional: n_layers capas Conv1d, todas de hidden a hidden canales
        capas = []
        for _ in range(n_layers):
            capas += [
                nn.Conv1d(hidden, hidden, kernel_size, padding=padding, padding_mode=padding_mode),
                nn.SiLU(),
            ]
        self.conv = nn.Sequential(*capas)
 
        # 2ª capa densa: de hidden canales a 1 canal (la velocidad), en cada punto por separado
        self.salida = nn.Linear(hidden, 1)
 
    def forward(self, x, tau):                              # x: [B, dim], tau: [B, 1]
        tau = tau.expand(-1, x.shape[1])                       # [B, dim]: repetir tau en cada punto
        entrada = torch.stack([x, tau], dim=-1)               # [B, dim, 2]: (valor, tiempo) por punto
        h = self.entrada(entrada)                            # [B, dim, hidden]
 
        h = h.transpose(1, 2)                               # [B, hidden, dim]: formato de Conv1d
        h = self.conv(h)                                     # [B, hidden, dim]
        h = h.transpose(1, 2)                               # [B, dim, hidden]: de vuelta
 
        return self.salida(h).squeeze(-1)                   # [B, dim, 1] -> [B, dim]