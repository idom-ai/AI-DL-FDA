"""BLOQUE 1: datos. Funciones constantes x(t) = c, con c ~ U[-1, 1]."""
import torch
from visualize import plot_originales

def make_dataset(n_samples=2000, n_points=128):
    c = 2 * torch.rand(n_samples, 1) - 1      # un nivel c por función, en [-1, 1]
    w = 6 * torch.pi
    t = torch.linspace(0, 1, n_points)
    return c.repeat(1, n_points) * torch.cos(w*t)             # [N, M]: cada fila es la función constante