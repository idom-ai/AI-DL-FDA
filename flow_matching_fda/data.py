"""BLOCK 1: DATA.

Every dataset is a scikit-fda FDataGrid: n_samples functions observed on the
same grid of n_points equispaced points t in [0, 1].

CONSTANT FUNCTIONS: x(t) = c, with c ~ U[-1, 1].
SINUSOIDAL FUNCTIONS: x(t) = c * cos(w * t), with c ~ U[-1, 1] and w = 6 * pi.

The neural nets work with torch tensors [N, M], so this block also has the two
conversions between both worlds: to_tensor and to_fdatagrid.
"""
import numpy as np
import torch
from skfda import FDataGrid


def make_constants(n_samples=2000, n_points=128, random_state=None):
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 1, n_points)
    c = rng.uniform(-1, 1, size=(n_samples, 1))
    return to_fdatagrid(c * np.ones_like(t), t, dataset_name="Constant functions")


def make_sinusoids(n_samples=2000, n_points=128, random_state=None):
    rng = np.random.default_rng(random_state)
    t = np.linspace(0, 1, n_points)
    c = rng.uniform(-1, 1, size=(n_samples, 1))
    w = 6 * np.pi
    return to_fdatagrid(c * np.cos(w * t), t, dataset_name="Sinusoidal functions")


def to_tensor(fd, device="cpu"):
    """FDataGrid with N functions on M points -> float32 tensor [N, M] (what the nets expect)."""
    # cast to float32 on CPU before moving: MPS has no float64
    return torch.as_tensor(fd.data_matrix[..., 0], dtype=torch.float32).to(device)


def to_fdatagrid(values, grid_points, dataset_name=None):
    """Values [N, M] (numpy array or torch tensor) on grid_points [M] -> FDataGrid."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return FDataGrid(
        data_matrix=values,
        grid_points=grid_points,
        dataset_name=dataset_name,
        argument_names=("t",),
        coordinate_names=("x(t)",),
    )
