"""
PASO 5 (v2): figuras del reporte. Cada figura es una función independiente;
si una falla (p.ej. falta un archivo) se reporta y se siguen generando las
demás.

Convenciones:
  - un punto = una réplica (película o experimento); la barra negra es la
    media y la línea vertical el IC95% bootstrap entre réplicas;
  - curvas finas grises = películas individuales; curva gruesa = media;
  - mapas de calor en escala secuencial de un solo tono.

Salida: resultados/v2/figuras/*.png y *.pdf
Uso: ../venv/bin/python 18_figuras.py --bf dist_tam --sirdna dist --camad dist_tam
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import CAMAD_ORDEN_CONDICIONES, DATASETS, res_dir  # noqa: E402
from lib import estilo as S  # noqa: E402
from lib.estilo import plt  # noqa: E402
from lib.inferencia import bootstrap_ic  # noqa: E402

FIG = res_dir("figuras")
NOMBRE_DS = {"bf": "Brightfield (células)", "sirdna": "SiR-DNA (núcleos)", "camad": "CAMAD"}


def est(ds, var, archivo):
    return pd.read_csv(res_dir(ds, "estadisticas", var) / archivo)


# ------------------------------------------------------------------ validación
def fig_val_seg_bf():
    d = res_dir("validacion_seg_bf")
    rk = pd.read_csv(d / "ranking_parametros.csv")
    el = json.loads((d / "eleccion.json").read_text())
    g = pd.read_csv(d / "grilla.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    piv = rk.pivot(index="cellprob", columns="flow", values="f1")
    # escala de color desde 0.78: la combinación sin control de flujo y cellprob
    # bajo da un F1 muy bajo que, si no, aplasta la escala del resto
    vals = piv.values
    vmin = max(0.78, np.nanmin(vals))
    im = axs[0].imshow(np.clip(vals, vmin, None), cmap=S.CMAP_SEQ, aspect="auto", origin="lower", vmin=vmin,
                       vmax=np.nanmax(vals))
    axs[0].set_xticks(range(len(piv.columns)), [("sin control" if c == 0 else f"{c}") for c in piv.columns])
    axs[0].set_yticks(range(len(piv.index)), piv.index)
    axs[0].set_xlabel("flow_threshold")
    axs[0].set_ylabel("cellprob_threshold")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            axs[0].text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if vals[i, j] > (vmin + np.nanmax(vals)) / 2 else S.TINTA)
    axs[0].set_title("F1 vs núcleos (media de 16 películas)")
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], fraction=0.046)
    # por película: default vs elegido
    agg = g.groupby(["cellprob", "flow", "movie"])[["tp", "fp", "fn", "fusiones", "n_nucleos",
                                                    "nucleos_sin_celula"]].sum().reset_index()
    agg["f1"] = 2 * agg.tp / (2 * agg.tp + agg.fp + agg.fusiones + agg.fn)
    agg["precision"] = agg.tp / (agg.tp + agg.fp + agg.fusiones)
    agg["recall"] = agg.tp / agg.n_nucleos
    # comparación: umbrales elegidos vs la alternativa relevante (si lo elegido
    # son los de por defecto, se compara contra el máximo F1 que empató)
    ce, fe = el["cellprob_threshold"], el["flow_threshold"]
    if (ce, fe) == (0.0, 0.4) and "maximo_f1" in el:
        ca, fa = el["maximo_f1"]["cellprob"], el["maximo_f1"]["flow"]
        lab_a, lab_b = f"máximo F1 ({ca:g}, {fa:g})", "elegido = por defecto (0, 0.4)"
    else:
        ca, fa = 0.0, 0.4
        lab_a, lab_b = "v1 (0, 0.4)", f"elegido ({ce:g}, {fe:g})"
    a = agg[(agg.cellprob == ca) & (agg.flow == fa)].set_index("movie")
    b = agg[(agg.cellprob == ce) & (agg.flow == fe)].set_index("movie")
    for k, met in enumerate(["f1", "precision", "recall"]):
        axs[1].plot([k - 0.15, k + 0.15], np.vstack([a[met], b[met]]), color=S.NEUTRO, lw=0.7, alpha=0.7)
        axs[1].scatter(np.full(len(a), k - 0.15), a[met], color=S.CAT[1], s=16, zorder=3,
                       label=lab_a if k == 0 else None)
        axs[1].scatter(np.full(len(b), k + 0.15), b[met], color=S.CAT[0], s=16, zorder=3,
                       label=lab_b if k == 0 else None)
    axs[1].set_xticks([0, 1, 2], ["F1", "precisión", "recall"])
    axs[1].set_title("Por película: dos opciones de umbrales")
    axs[1].legend(loc="lower left")
    # tipos de error
    tipos = pd.DataFrame({"a": [a.fp.sum(), a.fusiones.sum(), a.nucleos_sin_celula.sum()],
                          "b": [b.fp.sum(), b.fusiones.sum(), b.nucleos_sin_celula.sum()]},
                         index=["célula sin núcleo", "fusión (>=2 núcleos)", "núcleos perdidos"])
    x = np.arange(3)
    axs[2].bar(x - 0.2, tipos["a"], 0.38, color=S.CAT[1], label=lab_a)
    axs[2].bar(x + 0.2, tipos["b"], 0.38, color=S.CAT[0], label=lab_b)
    axs[2].set_xticks(x, tipos.index, rotation=15)
    axs[2].set_title("Errores (frames de la grilla, 16 películas)")
    axs[2].legend()
    S.guardar(fig, FIG / "val_segmentacion_bf")


def fig_val_seg_camad():
    d = res_dir("validacion_seg_camad")
    g = pd.read_csv(d / "grilla.csv")
    el = json.loads((d / "eleccion.json").read_text())
    por_exp = g.groupby(["escala", "cellprob", "flow", "exp"])[["recall", "iou_medio", "sobre_seg", "sub_seg"]].mean()
    med = por_exp.groupby(level=[0, 1, 2]).mean()
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for k, met in enumerate(["recall", "iou_medio"]):
        best = med.groupby(level=0)[met].max()
        axs[0].plot(best.index, best.values, "o-", color=S.CAT[k], label=f"{met} (mejores umbrales)")
    axs[0].set_xlabel("reducción de resolución (x)")
    axs[0].set_xticks(sorted(g.escala.unique()))
    axs[0].set_title("CAMAD: calidad vs escala")
    axs[0].legend()
    sel = por_exp.xs((el["escala"], el["cellprob_threshold"], el["flow_threshold"]), level=[0, 1, 2]).reset_index()
    v1 = por_exp.xs((1, 0.0, 0.4), level=[0, 1, 2]).reset_index() if (1, 0.0, 0.4, 1) in por_exp.index else None
    from lib.config import CAMAD_CONDICION
    sel["cond"] = sel.exp.map(CAMAD_CONDICION)
    conds = [c for c in CAMAD_ORDEN_CONDICIONES if c in sel.cond.unique()]
    S.puntos_con_media(axs[1], [sel[sel.cond == c].recall for c in conds], conds)
    axs[1].set_ylabel("recall (IoU >= 0.5)")
    axs[1].set_title(f"Recall por experimento (escala {el['escala']}x)")
    if v1 is not None:
        m = sel.merge(v1, on="exp", suffixes=("", "_v1"))
        axs[2].scatter(m.recall_v1, m.recall, color=S.CAT[0], s=22)
        axs[2].plot([0, 1], [0, 1], color=S.NEUTRO, lw=1, ls="--")
        for _, r in m.iterrows():
            axs[2].annotate(str(int(r.exp)), (r.recall_v1, r.recall), fontsize=6, color=S.TINTA2,
                            xytext=(2, 2), textcoords="offset points")
        axs[2].set_xlabel("recall v1 (resolución completa, umbrales por defecto)")
        axs[2].set_ylabel("recall v2 (elegido)")
        axs[2].set_title("Cada punto = un experimento")
    S.guardar(fig, FIG / "val_segmentacion_camad")


def fig_val_tracking():
    d = res_dir("validacion_tracking")
    df = pd.read_csv(d / "por_pelicula.csv")
    el = json.loads((d / "eleccion.json").read_text())
    vs = ["v1"] + [v for v in sorted(df.variante.unique()) if v != "v1"]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, met, tit in zip(axs, ["precision_eslabon", "recall_eslabon", "pureza", "completitud"],
                            ["Precisión de eslabones", "Recall de eslabones", "Pureza de trayectoria",
                             "Completitud de trayectoria"]):
        grupos = [df[df.variante == v][met] for v in vs]
        S.puntos_con_media(ax, grupos, vs, ic=[bootstrap_ic(x) for x in grupos])
        ax.set_title(tit)
    fig.suptitle(f"Tracking BF vs tracking de núcleos (16 películas). Elegida: {el['elegida']}")
    S.guardar(fig, FIG / "val_tracking")


# ------------------------------------------------------------------ simulaciones
def fig_simulaciones():
    d = res_dir("simulaciones")
    a = pd.read_csv(d / "a_reproduccion_liu2021.csv")
    b = pd.read_csv(d / "b_perfiles_ead_liu2024.csv")
    c = pd.read_csv(d / "c_sesgo_longitud.csv")
    r = pd.read_csv(d / "d_regimen_real.csv")
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
    x = np.arange(len(a))
    axs[0].bar(x - 0.2, a.se_media, 0.38, color=S.CAT[0], label="esta implementación")
    axs[0].bar(x + 0.2, a.se_paper_aprox, 0.38, color=S.NEUTRO, label="Liu 2021 Fig. 2e (aprox.)")
    axs[0].set_xticks(x, [f"P={p:g} min" for p in a.P_min])
    axs[0].set_ylabel("SE")
    axs[0].set_title("(a) Reproducción SE-FPS")
    axs[0].legend(loc="upper right")
    for i, (P, g) in enumerate(b.groupby("P_min")):
        axs[1].plot(g.tau_min, g.ead_media, color=S.CAT[i], label=f"P={P:g} min")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("lag (min)")
    axs[1].set_ylabel("EAD corregida")
    axs[1].set_title("(b) Perfiles EAD(τ) (cf. Liu 2024 Fig. 2D)")
    axs[1].legend()
    for i, (P, g) in enumerate(c.groupby("P_min")):
        lab = "aleatorio" if P < 0.1 else f"P={P:g} min"
        axs[2].plot(g.m_angulos, g.ead1_cruda, "--", color=S.CAT[i])
        axs[2].plot(g.m_angulos, g.ead1_corr, "-", color=S.CAT[i], label=lab)
    axs[2].set_xscale("log")
    axs[2].set_xlabel("ángulos por célula (m)")
    axs[2].set_ylabel("EAD1 (12 bins)")
    axs[2].set_title("(c) Sesgo: cruda (--) vs corregida (—)")
    axs[2].legend()
    for i, (s, g) in enumerate(r.groupby("sigma_um")):
        axs[3].errorbar(g.P_min, g.ead1_corr, yerr=g.ead1_corr_sd / np.sqrt(300) * 1.96, color=S.CAT[i],
                        marker="o", ms=3, label=f"EAD1, σ={s:g} µm")
        axs[3].plot(g.P_min, g.se_corr, ":", color=S.CAT[i])
    axs[3].set_xscale("log")
    axs[3].set_xlabel("persistencia P simulada (min)")
    axs[3].set_ylabel("valor corregido")
    axs[3].set_title("(d) Régimen real: Δ=5 min, 60 pasos (SE punteada)")
    axs[3].legend(fontsize=7)
    S.guardar(fig, FIG / "sim_validacion_metodos")


# ------------------------------------------------------------------ resultados por dataset
def fig_msd_vacf(ds, var):
    msd = est(ds, var, "msd.csv")
    vac = est(ds, var, "vacf.csv")
    fps = est(ds, var, "fps.csv")
    peli = est(ds, var, "por_pelicula.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for mv, g in msd.groupby("movie"):
        axs[0].plot(g.tau_s / 60, g.msd_um2, color=S.NEUTRO, lw=0.7, alpha=0.7)
    gm = msd.groupby("tau_s").msd_um2.mean()
    axs[0].plot(gm.index / 60, gm.values, color=S.CAT[0], lw=2.2, label="media entre réplicas")
    t = gm.index.values / 60
    ref = gm.values[np.isfinite(gm.values)][0]
    axs[0].plot(t, ref * (t / t[0]), ls="--", color=S.TINTA2, lw=1, label="α = 1")
    axs[0].plot(t, ref * (t / t[0]) ** 2, ls=":", color=S.TINTA2, lw=1, label="α = 2")
    # PRW con la mediana de parámetros
    from lib.motilidad import prw_msd
    Dm, Pm, sm = peli.prw_D_um2_min.median(), peli.prw_P_min.median(), peli.prw_sigma_um.median()
    axs[0].plot(t, prw_msd(t, Dm, Pm, sm ** 2), color=S.CAT[1], lw=1.4,
                label=f"PRW mediano: P={Pm:.1f} min, σ={sm:.2f} µm")
    axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    axs[0].set_ylim(bottom=np.nanmin(gm.values) / 3)
    axs[0].set_xlabel("τ (min)")
    axs[0].set_ylabel("MSD (µm²)")
    axs[0].set_title(f"TEA-MSD — {NOMBRE_DS[ds]}")
    axs[0].legend(fontsize=7)
    for mv, g in vac.groupby("movie"):
        axs[1].plot(g.tau_min, g.vacf, color=S.NEUTRO, lw=0.7, alpha=0.7)
    gv = vac.groupby("tau_min").vacf.mean()
    axs[1].plot(gv.index, gv.values, color=S.CAT[0], lw=2.2)
    axs[1].axhline(0, color=S.TINTA2, lw=0.8)
    axs[1].set_xlabel("τ (min)")
    axs[1].set_ylabel("VACF normalizada")
    axs[1].set_title("Autocorrelación de velocidad")
    for mv, g in fps.groupby("movie"):
        axs[2].plot(g.f_1_min[1:], g.fps_um2_min[1:], color=S.NEUTRO, lw=0.7, alpha=0.7)
    gf = fps[fps.f_1_min > 0].groupby("f_1_min").fps_um2_min.mean()
    axs[2].plot(gf.index, gf.values, color=S.CAT[0], lw=2.2)
    axs[2].set_xscale("log")
    axs[2].set_yscale("log")
    axs[2].set_xlabel("frecuencia (1/min)")
    axs[2].set_ylabel("FPS (µm²/min)")
    axs[2].set_title("Espectro de potencias de la velocidad")
    S.guardar(fig, FIG / f"{ds}_msd_vacf_fps")


def fig_ead(ds, var):
    ead = est(ds, var, "ead_tau.csv")
    cel = est(ds, var, "por_celula.csv")
    pdf = np.load(res_dir(ds, "estadisticas", var) / "pdf3d_angulos.npz")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    M = np.nanmean([pdf[k] for k in pdf.files], axis=0)
    tau_min = ead.groupby("tau").tau_min.first().values
    im = axs[0].imshow(M, aspect="auto", origin="lower", cmap=S.CMAP_SEQ,
                       extent=[-180, 180, tau_min[0] - (tau_min[1] - tau_min[0]) / 2,
                               tau_min[-1] + (tau_min[1] - tau_min[0]) / 2])
    axs[0].set_xlabel("ángulo entre v(t) y v(t+τ) (grados)")
    axs[0].set_ylabel("τ (min)")
    axs[0].set_title("PDF de ángulos vs lag (media de réplicas)")
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], fraction=0.046, label="densidad (1/rad)")
    for mv, g in ead.groupby("movie"):
        axs[1].plot(g.tau_min, g.ead_ens_corr, color=S.NEUTRO, lw=0.7, alpha=0.7)
    gm = ead.groupby("tau_min").ead_ens_corr.mean()
    axs[1].plot(gm.index, gm.values, color=S.CAT[0], lw=2.2, label="ensamble (36 bins)")
    gc = ead.groupby("tau_min").ead_cel_media_corr.mean()
    gs = ead.groupby("tau_min").ead_cel_sd_corr.mean()
    axs[1].errorbar(gc.index, gc.values, yerr=gs.values, color=S.CAT[1], lw=1.4, capsize=2,
                    label="media por célula ± SD (12 bins)")
    axs[1].axhline(1, color=S.TINTA2, lw=0.8, ls="--")
    axs[1].set_xlabel("τ (min)")
    axs[1].set_ylabel("EAD corregida")
    axs[1].set_title("EAD(τ): 1 = sin dirección preferida")
    axs[1].legend(fontsize=7, loc="lower right")
    c = cel.dropna(subset=["ead1_corr"])
    tl = c.tl1_min.fillna(c.tl1_min.max() + 5)
    hb = axs[2].hexbin(tl, c.ead1_corr, gridsize=(12, 25), cmap=S.CMAP_SEQ, mincnt=1)
    axs[2].set_xlabel("TL1 (min)  [censurados a la derecha]")
    axs[2].set_ylabel("EAD1 corregida")
    axs[2].set_title("Distribución conjunta (TL1, EAD1) por célula")
    axs[2].grid(False)
    fig.colorbar(hb, ax=axs[2], fraction=0.046, label="células")
    S.guardar(fig, FIG / f"{ds}_ead")


def fig_heatmaps_tiempo(ds, var, movie=None):
    se = pd.read_csv(res_dir(ds, "estadisticas", var) / "se_wavelet_tiempo.csv.gz")
    ea = pd.read_csv(res_dir(ds, "estadisticas", var) / "ead_tiempo.csv.gz")
    peli = est(ds, var, "por_pelicula.csv")
    if movie is None:
        movie = int(peli.sort_values("n_segmentos").movie.iloc[len(peli) // 2])
    fig, axs = plt.subplots(2, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    for fila, (df, col, nombre) in enumerate(((se, "se_corr", "SE(t) wavelet corregida"),
                                              (ea, "ead_corr", "EAD(t) corregida, ventana 1 h"))):
        d = df[df.movie == movie].copy()
        d["cel"] = d.track_id.astype(str) + "_" + d.seg.astype(str)
        piv = d.pivot_table(index="cel", columns="t_s", values=col)
        orden = piv.notna().idxmax(axis=1).sort_values().index
        piv = piv.loc[orden]
        vmin = float(np.floor(np.nanpercentile(piv.values, 2) * 10) / 10)
        im = axs[fila, 0].imshow(piv.values, aspect="auto", cmap=S.CMAP_SEQ, vmin=vmin, vmax=1.05,
                                 interpolation="nearest",
                                 extent=[piv.columns.min() / 3600, piv.columns.max() / 3600, len(piv), 0])
        axs[fila, 0].set_xlabel("tiempo (h)")
        axs[fila, 0].set_ylabel("célula")
        axs[fila, 0].set_title(f"{nombre} — réplica {movie} ({len(piv)} células)")
        axs[fila, 0].grid(False)
        fig.colorbar(im, ax=axs[fila, 0], fraction=0.03)
        for mv, g in df.groupby("movie"):
            gg = g.groupby("t_s")[col].agg(["mean", "size"])
            gg = gg[gg["size"] >= 10]
            axs[fila, 1].plot(gg.index / 3600, gg["mean"], color=S.CAT[0] if mv == movie else S.NEUTRO,
                              lw=2 if mv == movie else 0.7, alpha=1 if mv == movie else 0.7)
        axs[fila, 1].set_xlabel("tiempo (h)")
        axs[fila, 1].set_ylabel("promedio entre células (corregida)")
        axs[fila, 1].axhline(1, color=S.TINTA2, lw=0.8, ls="--")
        axs[fila, 1].set_title("Promedio por réplica (azul = la del mapa)")
    S.guardar(fig, FIG / f"{ds}_mapas_temporales")


def fig_colectivo(ds, var):
    corr = est(ds, var, "corr_espacial.csv")
    peli = est(ds, var, "por_pelicula.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    for mv, g in corr.groupby("movie"):
        axs[0].plot(g.r_um, g.corr_direccion, color=S.NEUTRO, lw=0.7, alpha=0.8)
    gm = corr.groupby("r_um").corr_direccion.mean()
    axs[0].plot(gm.index, gm.values, color=S.CAT[0], lw=2.2)
    axs[0].axhline(0, color=S.TINTA2, lw=0.8)
    axs[0].set_xlabel("distancia entre células (µm)")
    axs[0].set_ylabel("⟨v̂ᵢ · v̂ⱼ⟩")
    axs[0].set_title("Alineamiento de direcciones vs distancia")
    x = np.array([0, 1])
    for _, r in peli.iterrows():
        axs[1].plot(x, [r.corr_se_vecinos, r.corr_se_lejanos], color=S.NEUTRO, lw=0.8)
    axs[1].scatter(np.zeros(len(peli)), peli.corr_se_vecinos, color=S.CAT[0], zorder=3, s=18)
    axs[1].scatter(np.ones(len(peli)), peli.corr_se_lejanos, color=S.CAT[1], zorder=3, s=18)
    axs[1].set_xticks(x, ["vecinas (< 60 µm)", "lejanas (> 200 µm)"])
    axs[1].set_ylabel("Spearman medio entre SE(t)")
    axs[1].set_title("Correlación de SE(t) entre pares (Liu 2021)")
    sc = axs[2].scatter(peli.rapidez_um_min, peli.rho_rapidez_cosgiro, c=peli.densidad_cel_mm2, cmap=S.CMAP_SEQ,
                        s=30, edgecolor=S.TINTA2, linewidth=0.5)
    fig.colorbar(sc, ax=axs[2], fraction=0.046, label="densidad (células/mm²)")
    axs[2].axhline(0, color=S.TINTA2, lw=0.8)
    axs[2].set_xlabel("rapidez media (µm/min)")
    axs[2].set_ylabel("ρ(rapidez, cos giro)")
    axs[2].set_title("Acoplamiento rapidez–persistencia por réplica")
    S.guardar(fig, FIG / f"{ds}_colectivo_acoplamiento")


def fig_pasos(ds, var):
    """Distribución de v^2 (Liu 2024 Fig. 1B) y rapidez vs giro (Fig. 1C)."""
    m = est(ds, var, "pasos_muestra.csv")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    v2 = m.rapidez_um_min ** 2
    axs[0].hist(v2, bins=60, color=S.CAT[0], edgecolor="white", linewidth=0.3)
    axs[0].set_yscale("log")
    axs[0].set_xlabel("v² ((µm/min)²)")
    axs[0].set_ylabel("pasos")
    axs[0].set_title("Distribución de la velocidad al cuadrado")
    ang = np.arccos(np.clip(m.cos_giro, -1, 1))
    hb = axs[1].hexbin(np.degrees(ang), m.rapidez_um_min, gridsize=30, cmap=S.CMAP_SEQ, mincnt=1)
    axs[1].set_xlabel("|ángulo de giro| (grados)")
    axs[1].set_ylabel("rapidez (µm/min)")
    axs[1].set_title("Rapidez vs giro (todas las réplicas)")
    axs[1].grid(False)
    fig.colorbar(hb, ax=axs[1], fraction=0.046, label="pasos")
    q = pd.qcut(m.rapidez_um_min, 8, duplicates="drop")
    g = m.groupby(q, observed=True).agg(v=("rapidez_um_min", "median"), c=("cos_giro", "mean"),
                                        n=("cos_giro", "size"), sd=("cos_giro", "std"))
    axs[2].errorbar(g.v, g.c, yerr=1.96 * g.sd / np.sqrt(g.n), color=S.CAT[0], marker="o", capsize=2)
    axs[2].axhline(0, color=S.TINTA2, lw=0.8)
    axs[2].set_xlabel("rapidez (µm/min, octavos)")
    axs[2].set_ylabel("⟨cos giro⟩ ± IC95%")
    axs[2].set_title("Las células rápidas giran menos (UCSP)")
    S.guardar(fig, FIG / f"{ds}_pasos_rapidez_giro")


def fig_densidad(ds, var):
    peli = est(ds, var, "por_pelicula.csv")
    mets = [("rapidez_um_min", "rapidez (µm/min)"), ("prw_P_min", "P PRW (min)"),
            ("ead1_ens_corr", "EAD1 ensamble corr."), ("se_ventana_corr", "SE ventana corr."),
            ("direccionalidad_1h", "direccionalidad (1 h)"), ("corr_dir_0_50um", "alineamiento 0-50 µm")]
    from scipy.stats import spearmanr
    fig, axs = plt.subplots(1, len(mets), figsize=(18, 3.2))
    for ax, (m, lab) in zip(axs, mets):
        ax.scatter(peli.densidad_cel_mm2, peli[m], color=S.CAT[0], s=22)
        for _, r in peli.iterrows():
            ax.annotate(str(int(r.movie)), (r.densidad_cel_mm2, r[m]), fontsize=6, color=S.TINTA2,
                        xytext=(2, 2), textcoords="offset points")
        rr = spearmanr(peli.densidad_cel_mm2, peli[m], nan_policy="omit")
        ax.set_title(f"{lab}\nρ={rr.statistic:.2f}, p={rr.pvalue:.3f}", fontsize=8)
        ax.set_xlabel("densidad (células/mm²)")
    S.guardar(fig, FIG / f"{ds}_vs_densidad")


def fig_bf_vs_nuc(var_bf, var_nuc):
    a = est("bf", var_bf, "por_pelicula.csv").set_index("movie")
    b = est("sirdna", var_nuc, "por_pelicula.csv").set_index("movie")
    mets = [("rapidez_um_min", "rapidez (µm/min)"), ("prw_P_min", "P PRW (min)"), ("prw_sigma_um", "σ localización (µm)"),
            ("ead1_ens_corr", "EAD1 ensamble corr."), ("se_ventana_corr", "SE ventana corr."),
            ("direccionalidad_1h", "direccionalidad (1 h)"), ("tl1_ens_min", "TL1 ensamble (min)"),
            ("corr_dir_0_50um", "alineamiento 0-50 µm")]
    from lib.inferencia import ccc_lin
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, (m, lab) in zip(axs.ravel(), mets):
        x, y = b.loc[a.index, m], a[m]
        ax.scatter(x, y, color=S.CAT[0], s=20)
        lo = np.nanmin([x.min(), y.min()])
        hi = np.nanmax([x.max(), y.max()])
        ax.plot([lo, hi], [lo, hi], color=S.NEUTRO, ls="--", lw=1)
        ax.set_xlabel("núcleos (SiR-DNA)")
        ax.set_ylabel("brightfield")
        ax.set_title(f"{lab}\nCCC={ccc_lin(x, y):.2f}", fontsize=8)
    fig.suptitle("¿Cambian los estadísticos si se usa el pipeline sin marcador? (1 punto = 1 película)")
    S.guardar(fig, FIG / "bf_vs_nucleos")


# ------------------------------------------------------------------ CAMAD
def fig_camad_condiciones(var):
    peli = est("camad", var, "por_pelicula.csv")
    mets = [("rapidez_um_min", "rapidez (µm/min)"), ("direccionalidad_1h", "direccionalidad (1 h)"),
            ("prw_P_min", "P PRW (min)"), ("ead1_ens_corr", "EAD1 ensamble corr."),
            ("se_ventana_corr", "SE ventana corr."), ("tl1_cel_mediana_min", "TL1 mediana (min)"),
            ("area_um2", "área (µm²)"), ("circularidad", "circularidad"),
            ("densidad_cel_mm2", "densidad (células/mm²)"), ("n_segmentos", "trayectorias analizadas")]
    conds = [c for c in CAMAD_ORDEN_CONDICIONES if c in peli.condicion.unique()]
    etiquetas = [c.replace("Matriz 231 (exp8-9)", "Matriz 231\n(exp8-9)*") for c in conds]
    fig, axs = plt.subplots(2, 5, figsize=(18, 7))
    for ax, (m, lab) in zip(axs.ravel(), mets):
        grupos = [peli[peli.condicion == c][m] for c in conds]
        S.puntos_con_media(ax, grupos, etiquetas, ic=[bootstrap_ic(g) if len(g.dropna()) > 1 else None for g in grupos])
        ax.set_title(lab)
        ax.axvline(len(conds) - 1.5, color=S.GRILLA, lw=1)
    fig.suptitle("CAMAD: un punto = un experimento (n = 2-4 por sustrato). *exp8-9: identidad celular incierta, fuera de la comparación",
                 y=1.0)
    S.guardar(fig, FIG / "camad_condiciones")


def fig_camad_tiempo(var):
    tr = pd.read_csv(res_dir("camad", "tracking", var) / "tracks.csv.gz",
                     usecols=["movie", "track_id", "frame", "x_um", "y_um", "area_um2", "borde"])
    from lib.config import CAMAD_CONDICION
    dt = DATASETS["camad"]["dt_s"]
    tr = tr.sort_values(["track_id", "frame"])
    # rapidez en pasos de 10 frames (5 min)
    filas = []
    for (mv, tid), g in tr.groupby(["movie", "track_id"]):
        g = g.set_index("frame")
        fr = np.arange(g.index.min(), g.index.max() + 1, 10)
        fr = fr[np.isin(fr, g.index)]
        if len(fr) < 2:
            continue
        xy = g.loc[fr, ["x_um", "y_um"]].to_numpy()
        ok = np.diff(fr) == 10
        v = np.linalg.norm(np.diff(xy, axis=0), axis=1)[ok] / 5.0
        t = fr[:-1][ok] * dt / 3600
        filas.append(pd.DataFrame({"movie": mv, "t_h": t, "rapidez": v}))
    rap = pd.concat(filas)
    rap["bin"] = (rap.t_h * 2).astype(int) / 2
    area = tr[~tr.borde].assign(t_h=lambda d: d.frame * dt / 3600)
    area["bin"] = (area.t_h * 2).astype(int) / 2
    ea = pd.read_csv(res_dir("camad", "estadisticas", var) / "ead_tiempo.csv.gz")
    ea["bin"] = (ea.t_s / 1800).astype(int) / 2
    conds = [c for c in CAMAD_ORDEN_CONDICIONES]
    fig, axs = plt.subplots(3, len(conds), figsize=(18, 8), sharey="row", sharex=True)
    for j, c in enumerate(conds):
        exps = [e for e, cc in CAMAD_CONDICION.items() if cc == c]
        for i, (df, col, lab) in enumerate(((rap, "rapidez", "rapidez (µm/min)"),
                                            (area, "area_um2", "área (µm²)"),
                                            (ea, "ead_corr", "EAD(t) corr."))):
            for k, e in enumerate(sorted(exps)):
                g = df[df.movie == e].groupby("bin")[col].agg(["mean", "size"])
                g = g[g["size"] >= 5]
                axs[i, j].plot(g.index, g["mean"], color=S.CAT[k], lw=1.4, label=f"exp{e}")
            if j == 0:
                axs[i, j].set_ylabel(lab)
        axs[0, j].set_title(c, fontsize=9)
        axs[0, j].legend(fontsize=6)
        axs[2, j].set_xlabel("tiempo desde el inicio (h)")
    fig.suptitle("CAMAD: evolución durante las 5 h (adhesión y esparcimiento en medio sin suero)")
    S.guardar(fig, FIG / "camad_tiempo")


def fig_sensibilidad(ds, var):
    s = est(ds, var, "sensibilidad_paso.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (m, lab) in zip(axs, [("ead1_cel_corr", "EAD1 por célula corr."), ("ead1_ens_corr", "EAD1 ensamble corr."),
                                  ("cos_giro_medio", "⟨cos giro⟩")]):
        for mv, g in s.groupby("movie"):
            ax.plot(g.paso_min, g[m], color=S.NEUTRO, lw=0.7, alpha=0.8)
        gm = s.groupby("paso_min")[m].mean()
        ax.plot(gm.index, gm.values, "o-", color=S.CAT[0], lw=2)
        ax.set_xscale("log")
        ax.set_xticks(sorted(s.paso_min.unique()), [f"{x:g}" for x in sorted(s.paso_min.unique())])
        ax.minorticks_off()
        ax.set_xlabel("paso de análisis (min)")
        ax.set_title(lab)
    fig.suptitle(f"{NOMBRE_DS[ds]}: la persistencia medida depende del intervalo de análisis")
    S.guardar(fig, FIG / f"{ds}_sensibilidad_paso")


def fig_camad_ead_condicion(var):
    ead = est("camad", var, "ead_tau.csv")
    conds = [c for c in CAMAD_ORDEN_CONDICIONES if c in ead.condicion.unique()]
    fig, axs = plt.subplots(1, len(conds), figsize=(18, 3.4), sharey=True)
    for ax, c in zip(axs, conds):
        g = ead[ead.condicion == c]
        for k, (mv, gg) in enumerate(g.groupby("movie")):
            ax.plot(gg.tau_min, gg.ead_ens_corr, color=S.CAT[k], lw=1.4, label=f"exp{mv}")
        ax.axhline(1, color=S.TINTA2, lw=0.8, ls="--")
        ax.set_title(c, fontsize=9)
        ax.set_xlabel("τ (min)")
        ax.legend(fontsize=6)
    axs[0].set_ylabel("EAD ensamble corr.")
    S.guardar(fig, FIG / "camad_ead_tau")


def fig_trayectorias(ds, var, movies):
    from lib.fuentes import listar_frames, leer_frame
    tr = pd.read_csv(res_dir(ds, "tracking", var) / "tracks.csv.gz", usecols=["movie", "track_id", "frame", "x", "y"])
    fig, axs = plt.subplots(1, len(movies), figsize=(5 * len(movies), 4.6))
    axs = np.atleast_1d(axs)
    items = listar_frames(ds)
    from lib.config import CAMAD_CONDICION
    for ax, mv in zip(axs, movies):
        it = [i for i in items if i["movie"] == mv][0]
        img = leer_frame(it).astype(float)
        lo, hi = np.percentile(img, (1, 99.5))
        ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
        g = tr[tr.movie == mv]
        largos = g.groupby("track_id").size()
        for k, tid in enumerate(largos[largos >= 20].index):
            gg = g[g.track_id == tid].sort_values("frame")
            ax.plot(gg.x, gg.y, lw=0.8, color=S.CAT[k % 3], alpha=0.9)
        ax.axis("off")
        t = f"{NOMBRE_DS[ds]} — réplica {mv}"
        if ds == "camad":
            t = f"exp{mv}: {CAMAD_CONDICION[mv]}"
        ax.set_title(t, fontsize=9)
    S.guardar(fig, FIG / f"{ds}_trayectorias")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf")
    ap.add_argument("--sirdna")
    ap.add_argument("--camad")
    ap.add_argument("--solo", default="")
    args = ap.parse_args()
    tareas = [("val_seg_bf", fig_val_seg_bf, ()), ("val_seg_camad", fig_val_seg_camad, ()),
              ("val_tracking", fig_val_tracking, ()), ("simulaciones", fig_simulaciones, ())]
    for ds in ("bf", "sirdna", "camad"):
        var = getattr(args, ds)
        if not var:
            continue
        tareas += [(f"{ds}_msd", fig_msd_vacf, (ds, var)), (f"{ds}_ead", fig_ead, (ds, var)),
                   (f"{ds}_mapas", fig_heatmaps_tiempo, (ds, var)), (f"{ds}_colectivo", fig_colectivo, (ds, var)),
                   (f"{ds}_sens", fig_sensibilidad, (ds, var)), (f"{ds}_pasos", fig_pasos, (ds, var))]
        if ds != "camad":
            tareas.append((f"{ds}_densidad", fig_densidad, (ds, var)))
            tareas.append((f"{ds}_tray", fig_trayectorias, (ds, var, [1, 8, 15])))
    if args.bf and args.sirdna:
        tareas.append(("bf_vs_nuc", fig_bf_vs_nuc, (args.bf, args.sirdna)))
    if args.camad:
        tareas += [("camad_cond", fig_camad_condiciones, (args.camad,)), ("camad_tiempo", fig_camad_tiempo, (args.camad,)),
                   ("camad_ead", fig_camad_ead_condicion, (args.camad,)),
                   ("camad_tray", fig_trayectorias, ("camad", args.camad, [14, 1, 12, 7, 8]))]
    solo = set(args.solo.split(",")) if args.solo else None
    for nombre, fn, a in tareas:
        if solo and nombre not in solo:
            continue
        try:
            fn(*a)
            print("OK  ", nombre, flush=True)
        except Exception as e:
            print("FALLA", nombre, repr(e), flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
