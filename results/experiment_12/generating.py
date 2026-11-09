"""GENERATION: loads a trained checkpoint and produces functions, plots and metrics."""
import matplotlib.pyplot as plt
import torch

from flow_matching_fda.data import make_sinusoids
from flow_matching_fda.flow_matching import generate
from flow_matching_fda.visualize import plot_evolution, plot_originals
from models.convolutionals import PropuestaPersonal

# ---- Configuration ----
CHECKPOINT_PATH = "propuesta_10000_epochs.pt"
N_GENERATE = 50       # functions to generate for the evaluation
ODE_STEPS = 1000      # integration steps of the ODE
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ---- 1. Load checkpoint and rebuild the model ----
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model = PropuestaPersonal(**ckpt["model_args"]).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ---- 2. Rebuild the real data (same seed as in training, to compare) ----
args = ckpt["data_args"]
data = make_sinusoids(args["n_samples"], args["n_points"], random_state=args["seed"])

# ---- 3. Generate new functions on the same grid t (the whole trajectory: from noise to functions) ----
trajectory = generate(model, N_GENERATE, data.grid_points[0], steps=ODE_STEPS)   # FDataGrid of surfaces x(tau, t)

# ---- 4. Metrics ----
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters of the model: {n_params}")

# ---- 5. Plots ----
plot_originals(data)
plot_evolution(trajectory)
plt.show()
