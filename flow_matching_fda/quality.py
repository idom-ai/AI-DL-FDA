"""BLOCK 6: quality metrics. Compares real data with generated data (both FDataGrid).

Two things are measured:
- Constancy MSE: how far each generated function is from being constant,
      MSE_n = (1 / |T|) * || x_n - c_n ||^2_L2 = (1 / |T|) * integral_T (x_n(t) - c_n)^2 dt,
  where c_n is the mean level of x_n. It is 0 for the real data (exact constants).
  The integrals are computed by scikit-fda over the grid (Simpson's rule).
- W1 (Wasserstein) distance between the real and the generated distribution of
  mean levels c: how similar the levels c are, as distributions.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from skfda.misc.metrics import l2_norm


def domain_length(functions):
    """|T|, the length of the domain of the functions (1 for t in [0, 1])."""
    start, stop = functions.domain_range[0]
    return stop - start


def mean_levels(functions):
    """Level c_n of each function: c_n = (1 / |T|) * integral_T x_n(t) dt. FDataGrid [N] -> array [N]."""
    return functions.integrate()[:, 0] / domain_length(functions)


def constancy_mse(functions):
    """MSE_n of each function with respect to its own constant level c_n. FDataGrid [N] -> array [N]."""
    return l2_norm(functions - mean_levels(functions)) ** 2 / domain_length(functions)


def w1_distance(real_levels, generated_levels):
    """Wasserstein (W1) distance between the two distributions of mean levels."""
    return wasserstein_distance(real_levels, generated_levels)


def plot_quality(data, generated):
    """2x2 figure: constancy MSE and distribution of mean levels, real vs generated."""
    mse_real = constancy_mse(data)
    mse_gen = constancy_mse(generated)
    c_real = mean_levels(data)
    c_gen = mean_levels(generated)
    w1 = w1_distance(c_real, c_gen)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # --- Row 1: constancy MSE ---
    axes[0, 0].hist(mse_real, bins=30, color="gray")
    axes[0, 0].set_title("Constancy MSE (real)")
    axes[0, 1].hist(mse_gen, bins=30, color="tab:orange")
    axes[0, 1].set_title(f"Constancy MSE (generated)\nmedian = {np.median(mse_gen):.2e}")
    for ax in axes[0]:
        ax.set_xlabel("MSE")
        ax.set_ylabel("count")

    # --- Row 2: distribution of mean levels c ---
    axes[1, 0].hist(c_real, bins=30, density=True, color="gray")
    axes[1, 0].set_title("Distribution of mean levels (real)")
    axes[1, 1].hist(c_gen, bins=30, density=True, color="tab:orange")
    axes[1, 1].set_title(f"Distribution of mean levels (generated)\nW1 = {w1:.3f}")
    for ax in axes[1]:
        ax.set_xlabel("mean level")
        ax.set_ylabel("density")
        ax.axvline(-1, color="red", linestyle="--", linewidth=1)
        ax.axvline(1, color="red", linestyle="--", linewidth=1)

    fig.suptitle("Quality metrics: real vs generated data (Flow Matching)")
    fig.tight_layout()
    fig.savefig("quality.png", dpi=150)
