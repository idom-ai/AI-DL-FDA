"""ENTRENAR: configura el experimento, entrena y guarda un checkpoint."""
import os
import torch

from flow_matching.data import make_constants
from models.architectures import VelocityNetPooling
from flow_matching.train import train

# ---- Configuración ----
N_SAMPLES = 2000      
N_POINTS = 128        
HIDDEN = 256
POOL_SIZE = 7
EPOCHS = 1000
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = f"pooling_{EPOCHS}_epochs_{POOL_SIZE}_pool_size.pt"

torch.manual_seed(SEED)

# ---- 1. Datos y modelo ----
data = make_constants(N_SAMPLES, N_POINTS).to(DEVICE)
model = VelocityNetPooling(dim=N_POINTS, hidden=HIDDEN, pool_size=POOL_SIZE).to(DEVICE)

# ---- 2. Entrenar ----
train(model, data, epochs=EPOCHS)

# ---- 3. Guardar checkpoint: pesos + todo lo necesario para reconstruir modelo y datos ----
torch.save({
    "model_state": model.state_dict(),
    "model_args": dict(dim=N_POINTS, hidden=256, pool_size=POOL_SIZE),
    "data_args": dict(n_samples=N_SAMPLES, n_points=N_POINTS, seed=SEED),
}, CHECKPOINT_PATH)

n_params = sum(p.numel() for p in model.parameters())
print(f"Checkpoint guardado en {CHECKPOINT_PATH}")
print(f"Número de parámetros entrenados: {n_params}")