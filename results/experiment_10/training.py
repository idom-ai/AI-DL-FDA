"""TRAINING: configures the experiment, trains and saves a checkpoint.

Same setup as experiment 10, but the data is handled with scikit-fda (FDataGrid).
"""
import torch

from flow_matching_fda.data import make_sinusoids
from flow_matching_fda.train import train
from flow_matching_fda.visualize import plot_losses
from models.unet_adaptation import SongsUNet1D

# ---- Configuration ----
N_SAMPLES = 2000
N_POINTS = 256                      # must be a multiple of 8 (otherwise the net pads and crops by itself)
CHANNELS = (32, 64, 128, 256)
EMBED_DIM = 2048
FOURIER_SCALE = 30.0
KERNEL_SIZE = 65
EPOCHS = 1000
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = f"unet1d_{EPOCHS}_epochs_kernel_{KERNEL_SIZE}.pt"

torch.manual_seed(SEED)              # model initialization and CFM noise

# ---- 1. Data (FDataGrid: N_SAMPLES functions on N_POINTS points of [0, 1]) and model ----
data = make_sinusoids(N_SAMPLES, N_POINTS, random_state=SEED)
model = SongsUNet1D(
    in_channels=1,
    channels=CHANNELS,
    embed_dim=EMBED_DIM,
    fourier_scale=FOURIER_SCALE,
    kernel_size=KERNEL_SIZE
).to(DEVICE)

# ---- 2. Train (train() moves the FDataGrid to a tensor on the model's device) ----
losses = train(model, data, epochs=EPOCHS)

# ---- 3. Save checkpoint: weights + everything needed to rebuild model and data ----
torch.save({
    "model_state": model.state_dict(),
    "model_args": dict(
        in_channels=1,
        channels=CHANNELS,
        embed_dim=EMBED_DIM,
        fourier_scale=FOURIER_SCALE,
        kernel_size=KERNEL_SIZE
    ),
    "data_args": dict(n_samples=N_SAMPLES, n_points=N_POINTS, seed=SEED),
}, CHECKPOINT_PATH)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Checkpoint saved in {CHECKPOINT_PATH}")
print(f"Number of trained parameters: {n_params}")

# ---- 4. Loss plot (losses_kernel_{KERNEL_SIZE}.png) ----
plot_losses(losses, path=f"losses_kernel_{KERNEL_SIZE}.png")
