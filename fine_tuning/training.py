"""TRAINING: fine-tunes the U-Net of experiment 10 (kernel 13) to functions sampled at N_POINTS points.

Phase 1: original backbone frozen, only the new input and output modules are trained.
Phase 2: a copy of the phase-1 model is unfrozen (backbone included) and trained with a lower lr.
The checkpoint of experiment 10 is never modified.
"""
import copy

import torch

from flow_matching_fda.data import make_sinusoids
from flow_matching_fda.train import train
from flow_matching_fda.visualize import plot_fine_tuning_losses
from models.multi_resolution import MultiResolutionUNet1D
from models.unet_adaptation import SongsUNet1D

# ---- Configuration ----
BACKBONE_PATH = "../results/experiment_10/unet1d_1000_epochs_kernel_13.pt"
BACKBONE_POINTS = 256        # resolution the backbone was trained on
N_POINTS = 1024              # target resolution: must be 256 * 2^n
N_SAMPLES = 2000
HIDDEN_CHANNELS = 32         # channels of the new modules
KERNEL_SIZE = 13             # same kernel as the backbone
EMBED_DIM = 256              # time embedding of the new modules
FOURIER_SCALE = 30.0
EPOCHS_PHASE_1 = 300
LR_PHASE_1 = 1e-3
EPOCHS_PHASE_2 = 300
LR_PHASE_2 = 1e-5
SEED = 0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CHECKPOINT_PATH = f"unet1d_kernel_13_finetuned_{N_POINTS}.pt"

torch.manual_seed(SEED)              # initialization of the new modules and CFM noise

# ---- 1. Data and pretrained backbone ----
data = make_sinusoids(N_SAMPLES, N_POINTS, random_state=SEED)
ckpt = torch.load(BACKBONE_PATH, map_location=DEVICE)
backbone = SongsUNet1D(**ckpt["model_args"]).to(DEVICE)
backbone.load_state_dict(ckpt["model_state"])

# ---- 2. Model: backbone + new input and output modules ----
model_args = dict(
    n_points=N_POINTS,
    backbone_points=BACKBONE_POINTS,
    hidden_channels=HIDDEN_CHANNELS,
    kernel_size=KERNEL_SIZE,
    embed_dim=EMBED_DIM,
    fourier_scale=FOURIER_SCALE,
)
model = MultiResolutionUNet1D(backbone, **model_args).to(DEVICE)

# ---- 3. Phase 1: backbone frozen ----
print("Phase 1: backbone frozen")
model.set_backbone_trainable(False)
losses_phase_1 = train(model, data, epochs=EPOCHS_PHASE_1, lr=LR_PHASE_1)

# ---- 4. Phase 2: everything unfrozen, on a copy (model and backbone stay untouched) ----
print("Phase 2: everything unfrozen")
model_ft = copy.deepcopy(model)
model_ft.set_backbone_trainable(True)
losses_phase_2 = train(model_ft, data, epochs=EPOCHS_PHASE_2, lr=LR_PHASE_2)

# ---- 5. Save checkpoint: weights + everything needed to rebuild model and data ----
torch.save({
    "model_state": model_ft.state_dict(),
    "backbone_args": ckpt["model_args"],
    "model_args": model_args,
    "data_args": dict(n_samples=N_SAMPLES, n_points=N_POINTS, seed=SEED),
}, CHECKPOINT_PATH)

n_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
print(f"Checkpoint saved in {CHECKPOINT_PATH}")
print(f"Number of trained parameters: {n_params}")

# ---- 6. Loss plot (losses.png) ----
plot_fine_tuning_losses(losses_phase_1, losses_phase_2)
