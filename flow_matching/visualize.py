"""BLOQUE 5: visualización. Solo dibuja; no calcula nada del modelo."""
import matplotlib.pyplot as plt

def plot_originales(data, n=15):
    """Las funciones constantes reales con las que se entrena."""
    plt.figure(figsize=(6, 4))
    for f in data[:n]:
        plt.plot(f)
    plt.ylim(-3, 3)
    plt.title("Funciones constantes originales")
    plt.xlabel("punto de la función")
    plt.tight_layout()
    plt.savefig("originales.png", dpi=150)


def plot_evolucion(trayectoria, n=10, paneles=6):
    """Cómo el ruido se va convirtiendo en funciones constantes durante la generación.

    Cada panel es un instante τ de la ODE; cada color es la misma muestra a lo largo del tiempo.
    """
    pasos = len(trayectoria) - 1
    fig, ejes = plt.subplots(1, paneles, figsize=(3 * paneles, 3), sharey=True)
    for eje, k in zip(ejes, range(0, pasos + 1, pasos // (paneles - 1))):
        for f in trayectoria[k, :n]:
            eje.plot(f, linewidth=1)
        eje.set_title(f"τ = {k / pasos:.1f}")
        eje.set_ylim(-3, 3)
    ejes[0].set_ylabel("valor de la función")
    fig.suptitle("Evolución de la generación: del ruido a la función constante")
    plt.tight_layout()
    plt.savefig("evolucion.png", dpi=150)
