"""GENERAR: carga un checkpoint ya entrenado y produce funciones, gráficas y métricas."""
import matplotlib.pyplot as plt
import torch

from flow_matching.quality import plot_calidad
from flow_matching.data import make_sinusoids
from flow_matching.flow_matching import generate
from models.unet_adaptation import SongsUNet1D
from flow_matching.visualize import plot_evolucion, plot_originals

# ---- Configuración ----
CHECKPOINT_PATH = "unet1d_100_epochs.pt"
N_GENERATE = 30       # funciones a generar para evaluar
ODE_STEPS = 1000      # pasos de integración (debe ser múltiplo de 5 para las gráficas)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- 1. Cargar checkpoint y reconstruir el modelo ----
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model = SongsUNet1D(**ckpt["model_args"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---- 2. Reconstruir los datos reales (misma semilla que en el entrenamiento, para comparar) ----
torch.manual_seed(ckpt["data_args"]["seed"])
data = make_sinusoids(ckpt["data_args"]["n_samples"], ckpt["data_args"]["n_points"])

# ---- 3. Generar funciones nuevas (devuelve toda la trayectoria: del ruido a las funciones) ----
with torch.no_grad():
    trayectoria = generate(model, N_GENERATE, ckpt["data_args"]["n_points"], steps=ODE_STEPS, device=DEVICE)
funciones = trayectoria[-1]

# ---- 4. Métricas ----
mse = funciones.var(dim=1, unbiased=False).mean().item()   # 0 = perfectamente constantes
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"MSE de constancia de lo generado: {mse:.6f}")
print(f"Número de parámetros del modelo: {n_params}")

# ---- 5. Gráficas ----
plot_originals(data)
plot_evolucion(trayectoria)
plot_calidad(data, funciones)
plt.show()