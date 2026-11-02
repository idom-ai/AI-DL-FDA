"""BLOCK 3: Conditional Flow Matching logic (cfm loss and generation).

- cfm_loss stays in torch: training needs gradients and scikit-fda (numpy) has no autograd.
- generate integrates the ODE dx/dtau = v_theta(x, tau) with scikit-fda's integrator
  make_sde_trajectories. With zero diffusion an SDE is just an ODE, and its Euler-Maruyama
  scheme becomes the explicit Euler method: x_{k+1} = x_k + dtau * v_theta(x_k, tau_k).
"""
import numpy as np
import torch
from skfda import FDataGrid
from skfda.datasets import make_sde_trajectories


def cfm_loss(model, x1):
    """CFM loss for real data x1 (torch tensor [B, M])."""
    x0 = torch.randn_like(x1)                                   # x0 ~ N(0, I): noise
    tau = torch.rand(x1.shape[0], 1, device=x1.device)          # t ~ U[0, 1]: time as random features
    xtau = (1 - tau) * x0 + tau * x1                            # noising each x1 according to tau
    dxtau_dtau = x1 - x0                                        # conditional velocity according to the flow matching ODE
    return ((model(xtau, tau) - dxtau_dtau) ** 2).mean()        # L2 distance between v and v_theta


def generate(model, n, grid_points, steps=100, random_state=None):
    """Generate n functions on grid_points following dx/dtau = v(x, tau) from pure noise (tau=0) to tau=1.

    Returns the whole trajectory as an FDataGrid of surfaces x(tau, t):
        - domain: tau (steps + 1 points in [0, 1]) x t (grid_points)
        - data_matrix[:, 0] is the initial noise and data_matrix[:, -1] the generated functions.
    Use functions_at(trajectory, k) to get the n functions at step k as an FDataGrid on t.
    """
    rng = np.random.default_rng(random_state)
    t = np.asarray(grid_points)
    device = next(model.parameters()).device

    @torch.no_grad()
    def velocity(tau, x):
        # skfda gives a float tau and a numpy array x [n, M]; the net wants torch float32 on its device
        x = torch.as_tensor(x, dtype=torch.float32).to(device)     # cast on CPU first: MPS has no float64
        tau = torch.full((x.shape[0], 1), tau, dtype=torch.float32, device=device)
        return model(x, tau).cpu().numpy()

    paths = make_sde_trajectories(
        initial_condition=rng.standard_normal((n, len(t))),     # x0 ~ N(0, I): one row per function
        drift=velocity,                                         # dx/dtau = v_theta(x, tau)
        diffusion=0.0,                                          # no noise: ODE (the default would be 1.0)
        n_grid_points=steps + 1,                                # tau_k = k / steps, k = 0, ..., steps
        start=0.0,
        stop=1.0,
    )

    # paths is a vector-valued function of tau: data_matrix [n, steps + 1, M].
    # The same numbers read as surfaces x(tau, t) keep both grids in one object.
    return FDataGrid(
        data_matrix=paths.data_matrix,
        grid_points=[paths.grid_points[0], t],
        argument_names=("tau", "t"),
        coordinate_names=("x(tau, t)",),
    )


def functions_at(trajectory, k):
    """The n functions of the trajectory at ODE step k (k = -1: generated functions), as an FDataGrid on t."""
    return FDataGrid(
        data_matrix=trajectory.data_matrix[:, k, :, 0],
        grid_points=trajectory.grid_points[1],
        argument_names=("t",),
        coordinate_names=("x(t)",),
    )
