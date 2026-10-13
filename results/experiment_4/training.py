"""ENTRENAR: configura el experimento, entrena y guarda un checkpoint."""
import os
import torch

from flow_matching.data import make_sinusoids
from models.architectures import SongsUNet1D
from flow_matching.train import train

# ---- Configuración ----
N_SAMPLES = 2000
N_POINTS = 128                      # debe ser múltiplo de 8 (si no, la red rellena y recorta sola)
CHANNELS = (32, 64, 128, 256)
EMBED_DIM = 256
FOURIER_SCALE = 30.0
EPOCHS = 1000
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = f"unet1d_{EPOCHS}_epochs.pt"

torch.manual_seed(SEED)

# ---- 1. Datos y modelo ----
data = make_sinusoids(N_SAMPLES, N_POINTS).to(DEVICE)          # (N_SAMPLES, N_POINTS)
model = SongsUNet1D(
    in_channels=1,
    channels=CHANNELS,
    embed_dim=EMBED_DIM,
    fourier_scale=FOURIER_SCALE,
).to(DEVICE)

# ---- 2. Entrenar ----
train(model, data, epochs=EPOCHS)

# ---- 3. Guardar checkpoint: pesos + todo lo necesario para reconstruir modelo y datos ----
torch.save({
    "model_state": model.state_dict(),
    "model_args": dict(
        in_channels=1,
        channels=CHANNELS,
        embed_dim=EMBED_DIM,
        fourier_scale=FOURIER_SCALE,
    ),
    "data_args": dict(n_samples=N_SAMPLES, n_points=N_POINTS, seed=SEED),
}, CHECKPOINT_PATH)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Checkpoint guardado en {CHECKPOINT_PATH}")
print(f"Número de parámetros entrenados: {n_params}")