"""
PASO 6 del pipeline (complementario): relación entre forma y movimiento,
célula por célula.

04_estadisticas.py calcula dinámica (rapidez, direccionalidad, α, D, τ_p) y
morfología (área, elongación, irregularidad) como columnas separadas de
por_celula.csv, pero nunca las cruza entre sí. Este script sí: para cada par
forma-movimiento calcula la correlación de Pearson de dos maneras:

  - "pooled": sobre las ~8000 células juntas, sin distinguir de qué película
    vienen. Fácil de que un solo campo con comportamiento distinto (p.ej. la
    película 15, mucho más rápida Y menos elongada que el resto) infle o
    invente una correlación que en realidad es una diferencia ENTRE campos,
    no una relación real célula a célula.
  - "por película, promediada": la correlación se calcula por separado
    DENTRO de cada una de las 16 películas (controlando el efecto de campo)
    y se promedia. Si el número pooled y el promedio por película se
    parecen, la relación es robusta. Si difieren mucho, la relación pooled
    era en gran parte un artefacto de las diferencias entre campos.

Salidas:
    resultados/estadisticas/correlaciones_forma_movimiento.csv
    resultados/figuras/rel_*.png/pdf/svg (hexbin + recta de tendencia)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = ROOT / "resultados" / "estadisticas"
OUT_DIR = ROOT / "resultados" / "figuras"

MIN_CELLS_PER_MOVIE = 20  # piso para que la correlación "por película" de un campo cuente

sns.set_theme(style="white", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "figure.dpi": 120,
})
CMAP = sns.color_palette("crest", as_cmap=True)   # secuencial, un solo tono (densidad de puntos)
TREND_COLOR = "#b1562c"                            # cálido, contrasta con el hexbin frío -> línea de tendencia


def relate(df, xcol, ycol, xlabel, ylabel, fname_stem, note=""):
    sub = df[[xcol, ycol, "movie"]].dropna()
    n = len(sub)

    r_pooled, p_pooled = stats.pearsonr(sub[xcol], sub[ycol])

    # correlación DENTRO de cada película por separado, para no confundir
    # una diferencia ENTRE campos con una relación real célula-a-célula.
    r_por_pelicula = []
    for movie, g in sub.groupby("movie"):
        if len(g) >= MIN_CELLS_PER_MOVIE:
            r, _ = stats.pearsonr(g[xcol], g[ycol])
            r_por_pelicula.append(r)
    r_within_mean = float(np.mean(r_por_pelicula)) if r_por_pelicula else np.nan
    r_within_sd = float(np.std(r_por_pelicula, ddof=1)) if len(r_por_pelicula) > 1 else np.nan

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    hb = ax.hexbin(sub[xcol], sub[ycol], gridsize=32, cmap=CMAP, mincnt=1, linewidths=0.15)
    cb = fig.colorbar(hb, ax=ax, pad=0.02)
    cb.set_label("células (n)", fontsize=9)

    # recta de tendencia (OLS, sobre las células pooled -- solo para guiar el ojo)
    xs = np.linspace(sub[xcol].min(), sub[xcol].max(), 50)
    slope, intercept = np.polyfit(sub[xcol], sub[ycol], 1)
    ax.plot(xs, intercept + slope * xs, "-", color=TREND_COLOR, linewidth=2, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(
        0.03, 0.96,
        f"r (pooled, n={n}) = {r_pooled:.2f}\nr (promedio por película, n={len(r_por_pelicula)}) = {r_within_mean:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.75", alpha=0.9),
    )
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{fname_stem}.{ext}")
    plt.close(fig)

    return {
        "x": xcol, "y": ycol, "n_celulas": n,
        "r_pooled": r_pooled, "p_pooled": p_pooled,
        "r_promedio_por_pelicula": r_within_mean, "sd_por_pelicula": r_within_sd,
        "n_peliculas": len(r_por_pelicula), "nota": note,
        "fname": fname_stem,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(STATS_DIR / "por_celula.csv")

    rows = []
    rows.append(relate(
        df, "aspect_ratio", "rapidez_media_um_s",
        "Relación de aspecto (elongación)", "Rapidez media (µm/s)",
        "rel_elongacion_rapidez",
        "¿las células más alargadas se mueven más rápido?"))
    rows.append(relate(
        df, "aspect_ratio", "indice_direccionalidad",
        "Relación de aspecto (elongación)", "Índice de direccionalidad",
        "rel_elongacion_direccionalidad",
        "¿las células más alargadas van más 'derecho'?"))
    rows.append(relate(
        df, "area_um2", "rapidez_media_um_s",
        "Área (µm²)", "Rapidez media (µm/s)",
        "rel_area_rapidez",
        "¿el tamaño de la célula se relaciona con qué tan rápido se mueve?"))
    rows.append(relate(
        df, "irregularidad", "rapidez_media_um_s",
        "Irregularidad de contorno", "Rapidez media (µm/s)",
        "rel_irregularidad_rapidez",
        "¿células con más protrusiones (contorno menos liso) se mueven más rápido?"))

    # bonus: elongación vs. exponente alfa individual (solo células con buen ajuste)
    alpha_ok = df[df.r2_alpha >= 0.7] if "r2_alpha" in df else df.iloc[0:0]
    if len(alpha_ok):
        rows.append(relate(
            alpha_ok, "aspect_ratio", "alpha",
            "Relación de aspecto (elongación)", "Exponente α individual",
            "rel_elongacion_alpha",
            "¿las células más alargadas tienen un movimiento más superdifusivo/persistente?"))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(STATS_DIR / "correlaciones_forma_movimiento.csv", index=False)
    print(out_df[["x", "y", "n_celulas", "r_pooled", "r_promedio_por_pelicula", "n_peliculas"]].to_string(index=False))
    print(f"\n{len(rows)} figuras -> {OUT_DIR}/rel_*.png  |  tabla -> correlaciones_forma_movimiento.csv")


if __name__ == "__main__":
    main()
