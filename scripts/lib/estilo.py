"""
Estilo común de las figuras v2 (PNG 200 dpi + PDF vectorial).

Colores (paleta categórica validada para daltonismo, orden fijo, nunca
reciclado): la identidad de cada serie siempre va además con etiqueta
directa o posición en el eje, nunca solo por color. Mapas de calor en una
sola tonalidad (azul claro -> oscuro), nunca arcoíris (los papers usan
"jet", que distorsiona la percepción de magnitud).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TINTA = "#0b0b0b"
TINTA2 = "#52514e"
GRILLA = "#e4e3df"
NEUTRO = "#9a9993"
AZUL_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_SEQ = LinearSegmentedColormap.from_list("azul", AZUL_SEQ)
CMAP_DIV = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", "#f0efec", "#ef8e8d", "#b52f2f"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRILLA,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "legend.frameon": False,
    "font.family": "DejaVu Sans", "figure.constrained_layout.use": True,
})


def guardar(fig, ruta_sin_ext):
    fig.savefig(f"{ruta_sin_ext}.png")
    fig.savefig(f"{ruta_sin_ext}.pdf")
    plt.close(fig)


def puntos_con_media(ax, grupos, etiquetas, color=CAT[0], ic=None, jitter=0.12, seed=0):
    """Un punto por réplica (película/experimento) + media e IC95% en negro."""
    import numpy as np
    rng = np.random.default_rng(seed)
    for i, vals in enumerate(grupos):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        x = i + rng.uniform(-jitter, jitter, len(vals))
        ax.scatter(x, vals, s=22, color=color, alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
        if len(vals):
            ax.hlines(vals.mean(), i - 0.28, i + 0.28, color=TINTA, lw=2, zorder=4)
            if ic is not None and ic[i] is not None and all(np.isfinite(ic[i])):
                ax.vlines(i, ic[i][0], ic[i][1], color=TINTA, lw=1.2, zorder=4)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=25, ha="right")
