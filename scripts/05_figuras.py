"""
PASO 5 del pipeline: figuras estilo publicación para los estadísticos de
04_estadisticas.py.

Arreglo #5 (ver ESTADO.md / docstring de 04_estadisticas.py): las barras
únicas en escala log del script anterior se reemplazan por cajas+violín
CON UNA CAJA POR PELÍCULA (no una sola barra pooleando las ~8000 células).
Esto resuelve dos problemas a la vez:
  - una barra sola en escala log no es interpretable (su longitud no
    significa nada si la base no es cero);
  - una sola caja pooleando todas las células escondería que la unidad de
    replicación real son las 16 películas, no las células (arreglo #6).
Cada figura por-célula muestra las 16 cajas/violines (variabilidad real
célula-a-célula dentro de cada película) MÁS una banda horizontal con la
media ± IC95% bootstrap ENTRE películas (el número "oficial" a citar, de
resultados/estadisticas/resumen_bootstrap.csv).

IMPORTANTE: solo hay UNA condición/dataset trackeado (brightfield
MDA-MB-231). Las 16 películas son 16 campos (réplicas biológicas) de esa
misma condición, no grupos experimentales distintos -- por eso hay un
intervalo de confianza poblacional real (n=16), pero sigue sin haber una
prueba de significancia ENTRE grupos, porque no existe un segundo grupo.

Salidas:
    resultados/figuras/*.png (300 dpi), *.pdf, *.svg
    resultados/figuras/indice_figuras.md   (qué gráfica es cada una + captions)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
STATS_DIR = ROOT / "resultados" / "estadisticas"
OUT_DIR = ROOT / "resultados" / "figuras"

R2_MIN = 0.7  # calidad mínima de ajuste individual para incluir una célula en alpha/K_alpha/tau_p/D/A

sns.set_theme(style="white", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "figure.dpi": 120,
})
PALETTE = sns.color_palette("colorblind")
VIOLIN_COLOR = PALETTE[0]
BAND_COLOR = "crimson"


def box_violin_por_pelicula(por_celula, col, label, fname_stem, medida_desc, boot_row=None, log_y=False, filtro_desc=""):
    data = por_celula[["movie", col]].dropna().copy()
    data["movie"] = data["movie"].astype(int)
    n_total = len(data)
    n_movies = data["movie"].nunique()
    order = sorted(data["movie"].unique())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.violinplot(data=data, x="movie", y=col, order=order, ax=ax, color=VIOLIN_COLOR,
                    inner=None, linewidth=0.6, cut=0, saturation=0.6, zorder=1)
    sns.boxplot(data=data, x="movie", y=col, order=order, ax=ax, width=0.15, showfliers=False,
                boxprops={"facecolor": "white", "zorder": 3}, whiskerprops={"zorder": 3},
                medianprops={"color": "black", "zorder": 4}, capprops={"zorder": 3}, zorder=3)

    if boot_row is not None and pd.notna(boot_row.get("mean", np.nan)):
        ax.axhspan(boot_row["ci95_lo"], boot_row["ci95_hi"], color=BAND_COLOR, alpha=0.10, zorder=0)
        ax.axhline(boot_row["mean"], color=BAND_COLOR, linewidth=1.3, linestyle="--", zorder=5,
                    label=f"media entre películas = {boot_row['mean']:.3g} (IC95% bootstrap n={int(boot_row['n_peliculas'])})")
        ax.legend(frameon=False, fontsize=7.5, loc="best")

    ax.set_xlabel("Película")
    ax.set_ylabel(label)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{fname_stem}.{ext}")
    plt.close(fig)

    boot_txt = ""
    if boot_row is not None and pd.notna(boot_row.get("mean", np.nan)):
        boot_txt = (
            f" La línea punteada y la banda son la media ± IC95% bootstrap "
            f"ENTRE películas (n={int(boot_row['n_peliculas'])}; ver "
            f"resumen_bootstrap.csv) -- ese es el número a citar, no un "
            f"promedio de las células individuales."
        )
    caption = (
        f"**{fname_stem}**: {medida_desc}. Violín + caja por PELÍCULA "
        f"({n_movies} películas, n={n_total} células en total). Cada caja "
        f"muestra la variabilidad célula-a-célula dentro de esa película; "
        f"las 16 cajas entre sí muestran la variabilidad entre campos."
        + boot_txt
        + (f" {filtro_desc}" if filtro_desc else "")
        + (" Eje Y en escala logarítmica por el amplio rango de valores." if log_y else "")
    )
    return {"fname": fname_stem, "n": n_total, "caption": caption}


def hist_figure(values, label, fname_stem, medida_desc, bins=40, xlim=None):
    values = np.asarray(pd.Series(values).dropna())
    n = len(values)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(values, bins=bins, color=VIOLIN_COLOR, edgecolor="white", linewidth=0.4)
    ax.set_xlabel(label)
    ax.set_ylabel("Frecuencia (n° de observaciones)")
    if xlim:
        ax.set_xlim(*xlim)
    ax.text(0.98, 0.95, f"n = {n}", transform=ax.transAxes, ha="right", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{fname_stem}.{ext}")
    plt.close(fig)
    caption = f"**{fname_stem}**: distribución poblacional pooled de {medida_desc}, n={n} observaciones (todas las películas juntas)."
    return {"fname": fname_stem, "n": n, "caption": caption}


def boot_row_dict(boot_df, metrica):
    r = boot_df[boot_df.metrica == metrica]
    return r.iloc[0].to_dict() if len(r) else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    por_celula = pd.read_csv(STATS_DIR / "por_celula.csv")
    boot_df = pd.read_csv(STATS_DIR / "resumen_bootstrap.csv")
    captions = []

    # ---- 1.3: alpha por tramo, K_alpha, tau_p, D, A (por PELÍCULA) ----
    alpha_pel_df = pd.read_csv(STATS_DIR / "alpha_por_pelicula_y_segmento.csv")
    segments = alpha_pel_df[["segmento", "tau_min_s", "tau_max_s"]].drop_duplicates().sort_values("segmento")
    for _, seg in segments.iterrows():
        seg_i = int(seg["segmento"])
        seg_df = alpha_pel_df[alpha_pel_df.segmento == seg_i][["movie", "alpha"]].dropna()
        boot_row = boot_row_dict(boot_df, f"alpha_tramo{seg_i}")
        captions.append(box_violin_por_pelicula(
            seg_df.rename(columns={"alpha": "alpha"}), "alpha",
            f"Exponente anómalo α, tramo {seg_i} (τ {seg['tau_min_s']:.0f}-{seg['tau_max_s']:.0f} s)",
            f"alpha_tramo{seg_i}_por_pelicula",
            f"Exponente α del ajuste log(EA-MSD)=α·log(τ)+C dentro del tramo {seg_i}, "
            f"un valor por película (no por célula: el fit es sobre la curva EA-MSD "
            f"poblacional de cada película).",
            boot_row=boot_row,
        ))

    prw_pel_df = pd.read_csv(STATS_DIR / "prw_por_pelicula.csv")
    for col, label, fname, desc in [
        ("tau_p_s", "Tiempo de persistencia τ_p (s)", "tau_p_por_pelicula",
         "Tiempo de persistencia del ajuste PRW (Persistent Random Walk) sobre la curva EA-MSD de cada película"),
        ("D_um2_s", "Coeficiente de difusión efectivo D (µm²/s)", "D_por_pelicula",
         "D = A/(4τ_p) del ajuste PRW por película"),
        ("A_um2", "Parámetro de escala A (µm²)", "A_por_pelicula",
         "Amplitud A del modelo PRW: MSD(τ)=A·(τ/τ_p−1+e^(−τ/τ_p)), por película"),
    ]:
        boot_row = boot_row_dict(boot_df, col)
        captions.append(box_violin_por_pelicula(
            prw_pel_df.dropna(subset=[col]), col, label, fname, desc, boot_row=boot_row,
        ))

    # ---- 1.4/1.5: rapidez, direccionalidad, morfología (por CÉLULA, agrupadas por película) ----
    for col, label, fname, desc, log_y in [
        ("rapidez_media_um_s", "Rapidez media (µm/s)", "rapidez_media_por_celula",
         "Rapidez media por célula (promedio de la rapidez instantánea a lo largo de su trayectoria)", False),
        ("indice_direccionalidad", "Índice de direccionalidad (adimensional, 0-1)", "indice_direccionalidad_por_celula",
         "Desplazamiento neto / longitud total recorrida, por célula (1 = línea recta, ->0 = muy tortuoso)", False),
        ("area_um2", "Área celular (µm²)", "area_por_celula",
         "Área promedio por célula a lo largo de su trayectoria (regionprops sobre máscaras de Cellpose)", False),
        ("aspect_ratio", "Relación de aspecto (eje mayor / eje menor)", "aspect_ratio_por_celula",
         "Elongación promedio por célula (1 = círculo perfecto, mayor = más elongada)", False),
        ("irregularidad", "Irregularidad de contorno (perímetro²/4π·área)", "irregularidad_por_celula",
         "Irregularidad promedio por célula (1 = círculo perfecto, mayor = contorno más irregular)", False),
    ]:
        boot_row = boot_row_dict(boot_df, col)
        captions.append(box_violin_por_pelicula(por_celula, col, label, fname, desc, boot_row=boot_row, log_y=log_y))

    # alpha/K_alpha individual por célula (filtrado por calidad de ajuste), igual agrupado por película
    alpha_ok = por_celula[por_celula.r2_alpha >= R2_MIN]
    captions.append(box_violin_por_pelicula(
        alpha_ok, "alpha", "Exponente anómalo α individual (por célula)", "alpha_individual_por_celula",
        "Exponente α del ajuste individual log(TA-MSD)=α·log(τ)+C por célula",
        filtro_desc=f"Solo células con ajuste individual de calidad R²≥{R2_MIN} "
                    f"(de {por_celula.alpha.notna().sum()} con suficientes frames para el ajuste)."))

    # ---- histogramas poblacionales pooled (1.4) ----
    rapidez_inst = pd.read_csv(STATS_DIR / "rapidez_instantanea.csv")
    captions.append(hist_figure(
        rapidez_inst["rapidez_um_s"], "Rapidez instantánea (µm/s)", "hist_rapidez_instantanea",
        "rapidez instantánea (paso a paso, gap-fijado, todas las películas)"))
    angulos = pd.read_csv(STATS_DIR / "angulos_giro.csv")
    captions.append(hist_figure(
        angulos["angulo_giro_deg"], "Ángulo de giro entre pasos consecutivos (grados)",
        "hist_angulos_giro", "ángulos de giro entre pasos consecutivos (gap-fijado: solo entre pasos genuinamente adyacentes en frame)",
        bins=36, xlim=(0, 180)))

    # ---- curva de MSD log-log: 16 curvas por película + gran media + tramos de régimen ----
    ea_pel_df = pd.read_csv(STATS_DIR / "msd_poblacion_por_pelicula.csv")
    pob_df = pd.read_csv(STATS_DIR / "msd_poblacion.csv")
    fig, ax = plt.subplots(figsize=(6, 4.8))
    for movie, g in ea_pel_df.groupby("movie"):
        ax.plot(g["tau_s"], g["ea_msd_um2"], color="0.75", linewidth=0.8, alpha=0.7, zorder=1)
    ax.plot([], [], color="0.75", linewidth=1.2, label=f"EA-MSD por película (n={ea_pel_df.movie.nunique()})")
    ax.errorbar(pob_df["tau_s"], pob_df["ea_msd_mean_um2"], yerr=pob_df["ea_msd_sem_um2"],
                fmt="-o", color=PALETTE[1], markersize=3, capsize=2, linewidth=1.5,
                label="gran media ± SEM entre películas", zorder=3)

    colors_seg = [PALETTE[2], PALETTE[3], PALETTE[4]]
    for i, seg in segments.iterrows():
        seg_i = int(seg["segmento"])
        alpha_mean = alpha_pel_df.loc[alpha_pel_df.segmento == seg_i, "alpha"].mean()
        sub = pob_df[(pob_df.tau_s >= seg["tau_min_s"]) & (pob_df.tau_s <= seg["tau_max_s"])]
        if len(sub) and pd.notna(alpha_mean):
            tau0, msd0 = sub["tau_s"].iloc[0], sub["ea_msd_mean_um2"].iloc[0]
            tt = np.array([sub["tau_s"].iloc[0], sub["tau_s"].iloc[-1]])
            ax.plot(tt, msd0 * (tt / tau0) ** alpha_mean, "--", color=colors_seg[(seg_i - 1) % 3],
                     linewidth=1.3, zorder=4, label=f"tramo {seg_i}: α={alpha_mean:.2f}")
        if i > 0:
            ax.axvline(seg["tau_min_s"], color="0.6", linestyle=":", linewidth=0.8, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("τ (s)")
    ax.set_ylabel("EA-MSD (µm²)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"msd_loglog.{ext}")
    plt.close(fig)
    captions.append({"fname": "msd_loglog", "n": ea_pel_df.movie.nunique(), "caption":
        "**msd_loglog**: EA-MSD en función del lag τ desde el nacimiento de cada trayectoria, "
        "escala log-log. Líneas grises finas: curva EA-MSD de cada una de las 16 películas "
        "(cada una ya truncada por su propio sesgo de supervivencia). Línea naranja: gran media "
        "± SEM entre películas. Líneas punteadas de color: ajuste log-log independiente por "
        "tramo (régimen) sobre la gran media, con el α promedio entre películas de ese tramo "
        "(ver alpha_por_pelicula_y_segmento.csv para el detalle película por película)."})

    # ---- VACF: 16 curvas por película + gran media ----
    vacf_pel_df = pd.read_csv(STATS_DIR / "vacf_por_pelicula.csv")
    vacf_df = pd.read_csv(STATS_DIR / "vacf.csv")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.axhline(0, color="0.7", linewidth=0.8)
    for movie, g in vacf_pel_df.groupby("movie"):
        g = g.sort_values("lag_s")
        norm = g["vacf_um2_s2"].iloc[0]
        if norm and norm > 0:
            ax.plot(g["lag_s"], g["vacf_um2_s2"] / norm, color="0.8", linewidth=0.7, alpha=0.6, zorder=1)
    ax.errorbar(vacf_df["lag_s"], vacf_df["vacf_norm"],
                yerr=vacf_df["vacf_sem_um2_s2"] / vacf_df["vacf_mean_um2_s2"].iloc[0],
                fmt="-o", color=PALETTE[0], markersize=3, capsize=2, linewidth=1.5,
                label="gran media ± SEM entre películas", zorder=3)
    ax.set_xlabel("Lag temporal (s)")
    ax.set_ylabel("VACF normalizada (adimensional)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"vacf.{ext}")
    plt.close(fig)
    captions.append({"fname": "vacf", "n": int(vacf_pel_df.movie.nunique()), "caption":
        "**vacf**: función de autocorrelación de velocidades normalizada (VACF(0)=1) vs. lag "
        "temporal (gap-fijado: pares de pasos agrupados por diferencia REAL de frame). Líneas "
        "grises: cada una de las 16 películas. Línea azul: gran media ± SEM entre películas."})

    # ---- índice de figuras ----
    lines = ["# Índice de figuras (resultados/figuras/)\n",
             "Una sola condición/dataset (brightfield MDA-MB-231, 16 películas de 100 frames, "
             "5 min/frame). Las figuras de violín/caja agrupan por PELÍCULA (n=16, la unidad de "
             "replicación real) en vez de pooler todas las células; la línea/banda punteada es "
             "la media ± IC95% bootstrap entre películas. Sigue sin haber comparación entre "
             "grupos ni pruebas de significancia porque no existe un segundo grupo experimental.\n"]
    for c in captions:
        lines.append(f"- `{c['fname']}.png/pdf/svg` (n={c['n']}): {c['caption']}\n")
    (OUT_DIR / "indice_figuras.md").write_text("\n".join(lines))

    print(f"{len(captions)} figuras generadas en {OUT_DIR}")
    for c in captions:
        print(f"  - {c['fname']} (n={c['n']})")


if __name__ == "__main__":
    main()
