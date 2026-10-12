"""BLOCK 1: DATA. 

CONSTANT FUNCTIONS x(t) = c, with c ~ U[-1, 1].
SINUSOIDAL FUNCTIONS: x(t) = cos(w*t)

"""
import torch
from flow_matching.visualize import plot_originals

def make_constants(n_samples=2000, n_points=128):
    c = 2 * torch.rand(n_samples, 1) - 1
    return c.repeat(1, n_points)

def make_sinusoids(n_samples=2000, n_points=128):
    c = 2 * torch.rand(n_samples, 1) - 1
    w = 6 * torch.pi
    t = torch.linspace(0, 1, n_points)
    return c.repeat(1, n_points) * torch.cos(w*t)