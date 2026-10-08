"""BLOQUE 3: la lógica de Conditional Flow Matching (pérdida y generación)."""
import torch

def cfm_loss(model, x1):
    """Pérdida CFM para un batch de datos reales x1."""
    x0 = torch.randn_like(x1)               # x0 ~ N(0, I): ruido
    t = torch.rand(x1.shape[0], 1, device=x1.device)          # t ~ U[0, 1]: un tiempo por muestra
    xt = (1 - t) * x0 + t * x1              # punto de la ruta que une x0 con x1
    v_objetivo = x1 - x0                    # velocidad condicional de esa ruta
    return ((model(xt, t) - v_objetivo) ** 2).mean()


@torch.no_grad()
def generate(model, n, dim, steps=100, device="cpu"):
    """Genera n funciones siguiendo la ODE dx/dt = v(x, t) desde ruido (t=0) hasta t=1.
 
    Devuelve TODA la trayectoria, de forma [steps + 1, n, dim]:
    trayectoria[0] es el ruido inicial y trayectoria[-1] son las funciones generadas.
    """
    x = torch.randn(n, dim, device=device)
    trayectoria = [x.cpu()]                 # se guarda en CPU: matplotlib no dibuja tensores de MPS/CUDA
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n, 1), i * dt, device=device)
        x = x + dt * model(x, t)            # paso de Euler
        trayectoria.append(x.cpu())
    return torch.stack(trayectoria)
 
 
# instalar version de desarrollo de scikit-fda e implementar modelo de difusion con la misma NNET 
# que este flow matching y ver lo que sale https://github.com/GAA-UAM/scikit-fda/wiki/First-steps-for-developers