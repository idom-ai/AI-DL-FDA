"""GENERATION: loads the three models of experiment 10 (kernels 9, 11 and 13), generates with each of them
from the same initial noise and compares the generated functions."""
import matplotlib.pyplot as plt
import torch

from flow_matching_fda.data import make_sinusoids
from flow_matching_fda.flow_matching import functions_at, generate
from flow_matching_fda.visualize import plot_evolution, plot_final_comparison, plot_originals
from models.unet_adaptation import SongsUNet1D

# ---- Configuration ----
KERNEL_SIZES = [9, 11, 13, 23, 33, 65, 85]      # one trained model per kernel size: unet1d_1000_epochs_kernel_{k}.pt
N_GENERATE = 50                 # functions to generate with each model
ODE_STEPS = 1000                # integration steps of the ODE
NOISE_SEED = 0                  # same initial noise for every model, so that they can be compared
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

generated = {}                  # generated functions (tau = 1) of each model

for kernel_size in KERNEL_SIZES:
    # ---- 1. Load checkpoint and rebuild the model ----
    ckpt = torch.load(f"unet1d_1000_epochs_kernel_{kernel_size}.pt", map_location=DEVICE)
    model = SongsUNet1D(**ckpt["model_args"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # ---- 2. Rebuild the real data (same seed as in training, to compare) ----
    args = ckpt["data_args"]
    data = make_sinusoids(args["n_samples"], args["n_points"], random_state=args["seed"])

    # ---- 3. Generate new functions on the same grid t (the whole trajectory: from noise to functions) ----
    trajectory = generate(model, N_GENERATE, data.grid_points[0], steps=ODE_STEPS, random_state=NOISE_SEED)
    generated[f"kernel {kernel_size}"] = functions_at(trajectory, -1)     # FDataGrid of functions x(t) at tau = 1

    # ---- 4. Metrics ----
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Kernel {kernel_size}: number of parameters of the model: {n_params}")

    # ---- 5. Evolution of this model ----
    plot_evolution(trajectory, title=f"Kernel {kernel_size}: from noise to the function",
                   path=f"evolution_kernel_{kernel_size}.png")

# ---- 6. Plots shared by the three models (all of them were trained on the same data) ----
plot_originals(data)
plot_final_comparison(generated)      # final_comparison.png
plt.show()
