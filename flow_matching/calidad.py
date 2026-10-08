"""BLOQUE 6: métricas de calidad. Compara datos reales con datos generados.

Dos cosas se miden:
- MSE de constancia: cuánto se aleja cada función generada de ser constante.
  En los datos reales es 0 por construcción (son constantes exactas).
- Distancia W1 (Wasserstein) entre la distribución de medias (niveles c) real
  y la generada: cuánto se parecen, como distribuciones, los niveles c.
"""
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance


def mse_constancia(funciones):
    """MSE_n = varianza de cada función respecto a su propia media. [B] -> [B]."""
    return funciones.var(dim=1, unbiased=False)


def medias(funciones):
    """Nivel c de cada función (su media). [B] -> [B]."""
    return funciones.mean(dim=1)


def distancia_w1(medias_reales, medias_generadas):
    """Distancia de Wasserstein (W1) entre las dos distribuciones de medias."""
    return wasserstein_distance(medias_reales.numpy(), medias_generadas.numpy())


def plot_calidad(data, funciones_generadas):
    """Figura 2x2: MSE de constancia y distribución de medias, reales vs generadas."""
    mse_real = mse_constancia(data)
    mse_gen = mse_constancia(funciones_generadas)
    c_real = medias(data)
    c_gen = medias(funciones_generadas)
    w1 = distancia_w1(c_real, c_gen)

    fig, ejes = plt.subplots(2, 2, figsize=(10, 7))

    # --- Fila 1: MSE de constancia ---
    ejes[0, 0].hist(mse_real, bins=30, color="gray")
    ejes[0, 0].set_title("MSE de constancia (reales)")
    ejes[0, 1].hist(mse_gen, bins=30, color="tab:orange")
    ejes[0, 1].set_title(f"MSE de constancia (generadas)\nmediana = {mse_gen.median():.2e}")
    for eje in ejes[0]:
        eje.set_xlabel("MSE")
        eje.set_ylabel("densidad")

    # --- Fila 2: distribución de medias (niveles c) ---
    ejes[1, 0].hist(c_real, bins=30, density=True, color="gray")
    ejes[1, 0].set_title("Distribución de medias (reales)")
    ejes[1, 1].hist(c_gen, bins=30, density=True, color="tab:orange")
    ejes[1, 1].set_title(f"Distribución de medias (generadas)\nW1 = {w1:.3f}")
    for eje in ejes[1]:
        eje.set_xlabel("valor medio")
        eje.set_ylabel("densidad")
        eje.axvline(-1, color="red", linestyle="--", linewidth=1)
        eje.axvline(1, color="red", linestyle="--", linewidth=1)

    fig.suptitle("Métricas de calidad: datos reales vs. generados (Flow Matching)")
    plt.tight_layout()
    plt.savefig("calidad.png", dpi=150)