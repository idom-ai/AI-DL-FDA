"""ENTRENAR: configura el experimento, entrena y guarda un checkpoint."""
import os
import torch

from flow_matching.data import make_constants
from flow_matching.models import VelocityNet
from flow_matching.train import train

# ---- Configuración ----
N_SAMPLES = 2000      
N_POINTS = 128        
HIDDEN = 256
EPOCHS = 1000
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = f"simple_{EPOCHS}_epochs.pt"

torch.manual_seed(SEED)

# ---- 1. Datos y modelo ----
data = make_constants(N_SAMPLES, N_POINTS).to(DEVICE)
model = VelocityNet(dim=N_POINTS, hidden=HIDDEN,).to(DEVICE)

# ---- 2. Entrenar ----
train(model, data, epochs=EPOCHS)

# ---- 3. Guardar checkpoint: pesos + todo lo necesario para reconstruir modelo y datos ----
torch.save({
    "model_state": model.state_dict(),
    "model_args": dict(
        dim=N_POINTS, hidden=HIDDEN
    ),
    "data_args": dict(n_samples=N_SAMPLES, n_points=N_POINTS, seed=SEED),
}, CHECKPOINT_PATH)

n_params = sum(p.numel() for p in model.parameters())
print(f"Checkpoint guardado en {CHECKPOINT_PATH}")
print(f"Número de parámetros entrenados: {n_params}")