"""BLOCK 3: Conditional Flow Matching logic (cfm loss and generation). """
import torch

def cfm_loss(model, x1):
    """CFM loss for real data x1."""
    x0 = torch.randn_like(x1)                                   # x0 ~ N(0, I): noise
    tau = torch.rand(x1.shape[0], 1, device=x1.device)          # t ~ U[0, 1]: time as random features
    xtau = (1 - tau) * x0 + tau * x1                            # noising each x1 according to tau
    dxtau_dtau = x1 - x0                                        # conditional velocity according to the flow matching ODE
    return ((model(xtau, tau) - dxtau_dtau) ** 2).mean()        # L2 distance between v and v_theta


@torch.no_grad()
def generate(model, n, dim, steps=100, device="cpu"):
    """Generate n functions acording to the ODE dx/dtau = v(x, tau) from pure noise (t=0) to t=1.
 
    Returns the whole trajectory, with the form [steps + 1, n, dim]:
        - trajectory[0] is the initial noise and trajectory[-1] the generated functions.
    """
    x = torch.randn(n, dim, device=device)
    trajectory = [x.cpu()]                 
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt, device=device)
        x = x + dt * model(x, t)                            # future: change the integrator
        trajectory.append(x.cpu())
    return torch.stack(trajectory)
 


 
# instalar version de desarrollo de scikit-fda e implementar modelo de difusion con la misma NNET 
# que este flow matching y ver lo que sale https://github.com/GAA-UAM/scikit-fda/wiki/First-steps-for-developers