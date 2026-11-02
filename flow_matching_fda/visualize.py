"""BLOCK 5: visualization. Plots, no calculations: scikit-fda draws the FDataGrid objects."""
import matplotlib.pyplot as plt
import numpy as np
from flow_matching_fda.flow_matching import functions_at


def plot_originals(data, n=15):
    """Plots the training functions."""
    fig, ax = plt.subplots(figsize=(6, 4))
    data[:n].plot(axes=ax)
    ax.set_ylim(-3, 3)
    ax.set_title("Training functions")
    fig.tight_layout()
    fig.savefig("originals.png", dpi=150)


def plot_evolution(trajectory, n=30, panels=6):
    """Noise evolution turning into functions.

    Each panel is an instant tau of the ODE; each color is the same sample along time.
    """
    taus = trajectory.grid_points[0]
    steps = len(taus) - 1
    fig, axes = plt.subplots(1, panels, figsize=(3 * panels, 3), sharey=True)
    for ax, k in zip(axes, np.linspace(0, steps, panels, dtype=int)):
        functions_at(trajectory, k)[:n].plot(axes=ax, linewidth=1)
        ax.set_title(f"τ = {taus[k]:.1f}")
        ax.set_ylim(-3, 3)
    fig.suptitle("Generation: from noise to the function")
    fig.tight_layout()
    fig.savefig("evolution.png", dpi=150)
