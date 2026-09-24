"""
Mapa de formas (morfoespacio) y "diagrama de fase" forma-movimiento.

Idea (literatura de la transición de atascamiento, "jamming"): el índice de
forma adimensional q = perímetro / sqrt(área) resume qué tan alargada e
irregular es una célula (círculo: 3.54; hexágono regular: 3.72). En el
modelo de vértices de tejidos epiteliales (Bi et al., Nat Phys 2015) y en
epitelio de vía aérea (Park et al., Nat Mater 2015) el valor q* = 3.81
separa un estado "sólido" (células compactas, casi inmóviles) de uno
"fluido" (células alargadas que migran). Esa frontera se dedujo para
monocapas confluentes; aquí las células no forman una monocapa (salvo la
película 15), así que se usa como referencia, no como ley.

Qué se calcula, con las máscaras ya segmentadas (sin volver a segmentar):
  1. Para cada célula en cada paso de 5 min: índice de forma q, alargamiento,
     solidez, circularidad, área, orientación del eje mayor; rapidez del
     paso siguiente; coseno del giro; ángulo entre la dirección de
     movimiento y el eje mayor; densidad local (vecinas a < 50 um).
  2. Morfoespacio (q vs solidez) coloreado por rapidez, persistencia y
     fracción de pasos a lo largo del eje mayor.
  3. "Diagrama de fase": densidad local vs q, coloreado por rapidez.
  4. Rapidez y persistencia en función de q, por película (réplicas).
  5. ¿Se mueven a lo largo de su eje mayor? Distribución del ángulo
     movimiento-eje según el alargamiento.
  6. Dinámica: correlación cruzada entre alargamiento y rapidez a distintos
     desfases (¿la célula se alarga ANTES de acelerar?).
  7. CAMAD: cómo se desplaza la población en el morfoespacio durante las 5 h
     de adhesión según el sustrato.
Todas las comparaciones usan las películas/experimentos como réplicas.

Salidas: resultados/v2/morfoespacio/*.csv, figuras morfo_*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import CAMAD_CONDICION, CAMAD_ORDEN_CONDICIONES, DATASETS, res_dir  # noqa: E402
from lib import estilo as S  # noqa: E402
from lib.estilo import plt  # noqa: E402
from lib.inferencia import bootstrap_ic  # noqa: E402

OUT = res_dir("morfoespacio")
FIG = res_dir("figuras")
Q_CRIT = 3.81
R_DENS = 50.0


def pasos(ds, var, paso):
    tr = pd.read_csv(res_dir(ds, "tracking", var) / "tracks.csv.gz")
    tr = tr[~tr.borde].copy()
    tr["q"] = tr.perimeter / np.sqrt(tr.area)
    dt_min = DATASETS[ds]["dt_s"] / 60 * paso
    filas = []
    # densidad local por frame
    dens = {}
    for (mv, fr), g in tr.groupby(["movie", "frame"]):
        xy = g[["x_um", "y_um"]].to_numpy()
        n = cKDTree(xy).query_ball_point(xy, R_DENS, return_length=True) - 1
        dens.update({(mv, fr, lab): d / (np.pi * R_DENS ** 2 / 1e6) for lab, d in zip(g.label, n)})
    for tid, g in tr.groupby("track_id"):
        g = g.set_index("frame").sort_index()
        fr = g.index.to_numpy()
        for f0 in fr:
            f1, fm = f0 + paso, f0 - paso
            if f1 not in g.index:
                continue
            a, b = g.loc[f0], g.loc[f1]
            v = np.array([b.x_um - a.x_um, b.y_um - a.y_um])
            sp = np.linalg.norm(v) / dt_min
            cosg = np.nan
            if fm in g.index:
                c = g.loc[fm]
                u = np.array([a.x_um - c.x_um, a.y_um - c.y_um])
                nu = np.linalg.norm(u) * np.linalg.norm(v)
                cosg = float(u @ v / nu) if nu > 0 else np.nan
            # ángulo entre movimiento y eje mayor (orientación de skimage: respecto del eje de filas)
            th_eje = a.orientation
            eje = np.array([np.sin(th_eje), np.cos(th_eje)])   # (dx, dy) en coords imagen (x=col, y=fila)
            nv = np.linalg.norm(v)
            cos_eje = abs(float(eje @ v / nv)) if nv > 0 else np.nan
            filas.append((a.movie, tid, f0, a.q, a.aspect_ratio, a.solidity, a.circularidad, a.area_um2, sp, cosg,
                          cos_eje, dens.get((a.movie, f0, a.label), np.nan)))
    return pd.DataFrame(filas, columns=["movie", "track_id", "frame", "q", "alargamiento", "solidez", "circularidad",
                                        "area_um2", "rapidez", "cos_giro", "cos_eje", "densidad_local"])


def _pasos_pelicula(args):
    ds, var, paso, mv = args
    tr = pd.read_csv(res_dir(ds, "tracking", var) / "tracks.csv.gz")
    return pasos_df(tr[tr.movie == mv], ds, paso)


def pasos_df(tr, ds, paso):
    tr = tr[~tr.borde].copy()
    tr["q"] = tr.perimeter / np.sqrt(tr.area)
    dt_min = DATASETS[ds]["dt_s"] / 60 * paso
    dens = {}
    for fr, g in tr.groupby("frame"):
        xy = g[["x_um", "y_um"]].to_numpy()
        n = cKDTree(xy).query_ball_point(xy, R_DENS, return_length=True) - 1
        for lab, d in zip(g.label, n):
            dens[(fr, lab)] = d / (np.pi * R_DENS ** 2 / 1e6)
    tr["densidad_local"] = [dens.get((f, l), np.nan) for f, l in zip(tr.frame, tr.label)]
    tr = tr.sort_values(["track_id", "frame"])
    out = []
    for tid, g in tr.groupby("track_id"):
        g = g.set_index("frame")
        idx = g.index
        sig = g.reindex(idx + paso)
        ant = g.reindex(idx - paso)
        vx = sig.x_um.to_numpy() - g.x_um.to_numpy()
        vy = sig.y_um.to_numpy() - g.y_um.to_numpy()
        ux = g.x_um.to_numpy() - ant.x_um.to_numpy()
        uy = g.y_um.to_numpy() - ant.y_um.to_numpy()
        nv = np.hypot(vx, vy)
        nu = np.hypot(ux, uy)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosg = (ux * vx + uy * vy) / (nu * nv)
            th = g.orientation.to_numpy()
            cos_eje = np.abs(np.sin(th) * vx + np.cos(th) * vy) / nv
        d = pd.DataFrame({"movie": g.movie.to_numpy(), "track_id": tid, "frame": idx.to_numpy(), "q": g.q.to_numpy(),
                          "alargamiento": g.aspect_ratio.to_numpy(), "solidez": g.solidity.to_numpy(),
                          "circularidad": g.circularidad.to_numpy(), "area_um2": g.area_um2.to_numpy(),
                          "rapidez": nv / dt_min, "cos_giro": cosg, "cos_eje": cos_eje,
                          "densidad_local": g.densidad_local.to_numpy()})
        out.append(d[np.isfinite(d.rapidez)])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def correlacion_cruzada(p, paso_frames, max_lag=6, min_len=24):
    """Correlación entre alargamiento(t) y rapidez(t + k): k > 0 = la forma
    antecede al cambio de rapidez. Por película, promediando sobre células."""
    filas = []
    for mv, gm in p.groupby("movie"):
        acc = {k: [] for k in range(-max_lag, max_lag + 1)}
        for _, g in gm.groupby("track_id"):
            g = g.sort_values("frame")
            fr = g.frame.to_numpy()
            if len(g) < min_len or np.any(np.diff(fr) != paso_frames):
                continue
            a = (g.alargamiento - g.alargamiento.mean()) / (g.alargamiento.std() + 1e-9)
            v = (g.rapidez - g.rapidez.mean()) / (g.rapidez.std() + 1e-9)
            a, v = a.to_numpy(), v.to_numpy()
            n = len(a)
            for k in acc:
                if k >= 0:
                    x, y = a[:n - k], v[k:]
                else:
                    x, y = a[-k:], v[:n + k]
                acc[k].append(np.mean(x * y))
        for k, vals in acc.items():
            if vals:
                filas.append({"movie": mv, "lag": k, "corr": np.mean(vals), "n_celulas": len(vals)})
    return pd.DataFrame(filas)


def hexmap(ax, x, y, c, gridsize, cmap, label, vmin=None, vmax=None, minimo=15, fig=None):
    hb = ax.hexbin(x, y, C=c, reduce_C_function=np.median, gridsize=gridsize, cmap=cmap, mincnt=minimo,
                   vmin=vmin, vmax=vmax)
    ax.grid(False)
    if fig is not None:
        fig.colorbar(hb, ax=ax, fraction=0.046, label=label)
    return hb


def main():
    import importlib.util
    from concurrent.futures import ProcessPoolExecutor
    var_bf = (res_dir().parent / "v2" / "variante_bf.txt")
    vbf = (Path(res_dir()) / "variante_bf.txt").read_text().strip() if (Path(res_dir()) / "variante_bf.txt").exists() else "dist_tam"
    vca = (Path(res_dir()) / "variante_camad.txt").read_text().strip() if (Path(res_dir()) / "variante_camad.txt").exists() else "camad_huecos"
    tareas = [("bf", vbf, 1, m) for m in range(1, 17)] + [("camad", vca, 10, m) for m in range(1, 17)]
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(_pasos_pelicula, tareas))
    bf = pd.concat(res[:16], ignore_index=True)
    ca = pd.concat(res[16:], ignore_index=True)
    ca["condicion"] = ca.movie.map(CAMAD_CONDICION)
    ca["t_h"] = ca.frame * DATASETS["camad"]["dt_s"] / 3600
    bf.to_csv(OUT / "pasos_bf.csv.gz", index=False)
    ca.to_csv(OUT / "pasos_camad.csv.gz", index=False)

    # calibración del índice de forma: células casi redondas
    red = bf[bf.circularidad > 0.9].q.median()
    print(f"q mediano de células casi redondas (circularidad > 0.9): {red:.2f} (círculo ideal 3.54)")

    # ---------------- estadística por película (réplicas)
    filas = []
    for mv, g in bf.groupby("movie"):
        r = lambda a, b: stats.spearmanr(g[a], g[b], nan_policy="omit").statistic
        filas.append({"movie": mv, "q_mediana": g.q.median(), "frac_fluida": (g.q > Q_CRIT).mean(),
                      "rho_q_rapidez": r("q", "rapidez"), "rho_q_cosgiro": r("q", "cos_giro"),
                      "rho_alarg_coseje": r("alargamiento", "cos_eje"),
                      "cos_eje_medio": g.cos_eje.mean(), "rapidez_media": g.rapidez.mean(),
                      "densidad_local_media": g.densidad_local.mean(),
                      "rapidez_fluida": g[g.q > Q_CRIT].rapidez.mean(), "rapidez_solida": g[g.q <= Q_CRIT].rapidez.mean()})
    pm = pd.DataFrame(filas)
    pm.to_csv(OUT / "por_pelicula_bf.csv", index=False)
    pr = []
    for c, h, alt in (("rho_q_rapidez", "rapidez aumenta con q", "greater"), ("rho_q_cosgiro", "persistencia aumenta con q", "greater"),
                      ("rho_alarg_coseje", "más alargadas -> se mueven más a lo largo de su eje", "greater")):
        pr.append({"prueba": c, "hipotesis": h, "mediana": pm[c].median(), "p": stats.wilcoxon(pm[c], alternative=alt).pvalue})
    d = pm.rapidez_fluida - pm.rapidez_solida
    pr.append({"prueba": "rapidez(q>3.81) - rapidez(q<=3.81)", "hipotesis": "las 'fluidas' son más rápidas",
               "mediana": d.median(), "p": stats.wilcoxon(d, alternative="greater").pvalue})
    # cos_eje vs uniforme (E|cos| para ángulo uniforme = 2/pi)
    pr.append({"prueba": "<|cos(mov, eje)|> - 2/pi", "hipotesis": "se mueven a lo largo del eje mayor más que al azar",
               "mediana": (pm.cos_eje_medio - 2 / np.pi).median(),
               "p": stats.wilcoxon(pm.cos_eje_medio - 2 / np.pi, alternative="greater").pvalue})
    r_dens = stats.spearmanr(pm.densidad_local_media, pm.q_mediana)
    pr.append({"prueba": "rho(densidad, q mediana) entre películas", "hipotesis": "más densidad -> formas más compactas",
               "mediana": r_dens.statistic, "p": r_dens.pvalue})
    pr = pd.DataFrame(pr)
    from lib.inferencia import holm
    pr["p_holm"] = holm(pr.p)
    pr.to_csv(OUT / "pruebas_bf.csv", index=False)
    print(pm.round(3).to_string(index=False))
    print(pr.round(4).to_string(index=False))

    cc = correlacion_cruzada(bf, 1)
    cc.to_csv(OUT / "correlacion_cruzada_bf.csv", index=False)
    ccm = cc.groupby("lag")["corr"].agg(["mean", "std", "count"])
    print(ccm.round(4).to_string())

    # ---------------- figuras BF
    b = bf.dropna(subset=["q", "solidez", "rapidez"])
    b = b[(b.q < np.nanpercentile(b.q, 99.5)) & (b.solidez > np.nanpercentile(b.solidez, 0.5))]
    fig, axs = plt.subplots(1, 4, figsize=(21, 4.6))
    hb = axs[0].hexbin(b.q, b.solidez, gridsize=45, cmap=S.CMAP_SEQ, mincnt=5, bins="log")
    fig.colorbar(hb, ax=axs[0], fraction=0.046, label="número de observaciones (log)")
    axs[0].grid(False)
    axs[0].set_title("Morfoespacio: cuántas células hay en cada forma")
    hexmap(axs[1], b.q, b.solidez, b.rapidez, 40, S.CMAP_SEQ, "rapidez mediana (µm/min)", fig=fig)
    axs[1].set_title("Rapidez según la forma")
    bb = b.dropna(subset=["cos_giro"])
    hexmap(axs[2], bb.q, bb.solidez, bb.cos_giro, 40, S.CMAP_DIV, "⟨cos giro⟩ (persistencia)", vmin=-0.4, vmax=0.4, fig=fig)
    axs[2].set_title("Persistencia según la forma")
    be = b.dropna(subset=["cos_eje"])
    hexmap(axs[3], be.q, be.solidez, be.cos_eje, 40, S.CMAP_SEQ, "⟨|cos(movimiento, eje mayor)|⟩", fig=fig)
    axs[3].set_title("¿Se mueve a lo largo de su eje mayor?")
    for ax in axs:
        ax.axvline(Q_CRIT, color=S.TINTA, ls="--", lw=1)
        ax.set_xlabel("índice de forma q = perímetro / √área")
        ax.set_ylabel("solidez (1 = sin entrantes)")
    fig.suptitle("Brightfield: mapa de formas (cada célula en cada paso de 5 min). Línea: q* = 3.81 (transición sólido–fluido de la literatura)")
    S.guardar(fig, FIG / "morfo_bf_mapa")

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    bd = b.dropna(subset=["densidad_local"])
    hexmap(axs[0], bd.densidad_local, bd.q, bd.rapidez, 35, S.CMAP_SEQ, "rapidez mediana (µm/min)", fig=fig)
    axs[0].axhline(Q_CRIT, color=S.TINTA, ls="--", lw=1)
    axs[0].set_xlabel("densidad local (células/mm² en un radio de 50 µm)")
    axs[0].set_ylabel("índice de forma q")
    axs[0].set_title("«Diagrama de fase»: densidad y forma vs rapidez")
    bins = np.quantile(b.q, np.linspace(0, 1, 13))
    for mv, g in b.groupby("movie"):
        gg = g.groupby(pd.cut(g.q, bins), observed=True).rapidez.mean()
        axs[1].plot([iv.mid for iv in gg.index], gg.values, color=S.NEUTRO, lw=0.8)
    gm = b.groupby(pd.cut(b.q, bins), observed=True).rapidez.mean()
    axs[1].plot([iv.mid for iv in gm.index], gm.values, color=S.CAT[0], lw=2.4, marker="o")
    axs[1].axvline(Q_CRIT, color=S.TINTA, ls="--", lw=1)
    axs[1].set_xlabel("índice de forma q")
    axs[1].set_ylabel("rapidez media (µm/min)")
    axs[1].set_title("Rapidez vs forma (gris: películas)")
    lg = ccm.index.to_numpy() * 5
    axs[2].errorbar(lg, ccm["mean"], yerr=1.96 * ccm["std"] / np.sqrt(ccm["count"]), color=S.CAT[0], marker="o", capsize=2)
    for mv, g in cc.groupby("movie"):
        axs[2].plot(g.lag * 5, g["corr"], color=S.NEUTRO, lw=0.6)
    axs[2].axhline(0, color=S.TINTA2, lw=0.8)
    axs[2].axvline(0, color=S.TINTA2, lw=0.8)
    axs[2].set_xlabel("desfase (min): positivo = la forma antecede a la rapidez")
    axs[2].set_ylabel("correlación alargamiento(t) · rapidez(t + desfase)")
    axs[2].set_title("¿La célula se alarga antes de acelerar?")
    S.guardar(fig, FIG / "morfo_bf_fase")

    # ángulo movimiento-eje según alargamiento
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    cats = [(1, 1.5, "1–1.5 (casi redondas)"), (1.5, 2.5, "1.5–2.5"), (2.5, 4, "2.5–4"), (4, 50, "> 4 (muy alargadas)")]
    for k, (lo, hi, lab) in enumerate(cats):
        g = be[(be.alargamiento >= lo) & (be.alargamiento < hi)]
        ang = np.degrees(np.arccos(np.clip(g.cos_eje, 0, 1)))
        axs[0].hist(ang, bins=18, range=(0, 90), density=True, histtype="step", lw=2, color=S.CAT[k], label=lab)
    axs[0].axhline(1 / 90, color=S.TINTA2, ls="--", lw=1, label="al azar")
    axs[0].set_xlabel("ángulo entre la dirección de movimiento y el eje mayor (grados)")
    axs[0].set_ylabel("densidad")
    axs[0].set_title("Dirección de movimiento respecto de la forma")
    axs[0].legend(title="alargamiento", fontsize=7)
    x = np.arange(len(pm))
    axs[1].scatter(pm.frac_fluida, pm.rapidez_media, color=S.CAT[0], s=30)
    for _, r in pm.iterrows():
        axs[1].annotate(str(int(r.movie)), (r.frac_fluida, r.rapidez_media), fontsize=7, color=S.TINTA2,
                        xytext=(3, 2), textcoords="offset points")
    rr = stats.spearmanr(pm.frac_fluida, pm.rapidez_media)
    axs[1].set_xlabel("fracción de observaciones con q > 3.81 («fluidas»)")
    axs[1].set_ylabel("rapidez media de la película (µm/min)")
    axs[1].set_title(f"Por película: ρ = {rr.statistic:.2f}, p = {rr.pvalue:.3f}")
    S.guardar(fig, FIG / "morfo_bf_eje")

    # ---------------- CAMAD: trayectoria de la población en el morfoespacio
    c = ca.dropna(subset=["q", "solidez"])
    c = c[(c.q < np.nanpercentile(c.q, 99.5))]
    conds = [x for x in CAMAD_ORDEN_CONDICIONES if x in set(c.condicion)]
    tb = [(0, 1, "0–1 h"), (2, 3, "2–3 h"), (4, 5, "4–5 h")]
    fig, axs = plt.subplots(len(tb), len(conds), figsize=(3.2 * len(conds), 8.2), sharex=True, sharey=True)
    for j, cond in enumerate(conds):
        for i, (a, b_, lab) in enumerate(tb):
            g = c[(c.condicion == cond) & (c.t_h >= a) & (c.t_h < b_)]
            ax = axs[i, j]
            if len(g) > 20:
                ax.hexbin(g.q, g.solidez, gridsize=22, cmap=S.CMAP_SEQ, mincnt=1, extent=(3.4, 7, 0.4, 1.0))
            ax.axvline(Q_CRIT, color=S.TINTA, ls="--", lw=0.8)
            ax.grid(False)
            if i == 0:
                ax.set_title(cond, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{lab}\nsolidez")
            if i == len(tb) - 1:
                ax.set_xlabel("índice de forma q")
            ax.text(0.97, 0.05, f"q med {g.q.median():.2f}" if len(g) else "", transform=ax.transAxes, ha="right",
                    fontsize=7, color=S.TINTA2)
    fig.suptitle("CAMAD: cómo se desplaza la población en el mapa de formas durante la adhesión (filas: tiempo)")
    S.guardar(fig, FIG / "morfo_camad_tiempo")
    ct = c.groupby(["movie", "condicion", pd.cut(c.t_h, [0, 1, 2, 3, 4, 5.1])], observed=True).agg(
        q_mediana=("q", "median"), frac_fluida=("q", lambda s: (s > Q_CRIT).mean()), rapidez=("rapidez", "mean")).reset_index()
    ct.to_csv(OUT / "camad_por_tiempo.csv", index=False)
    print(ct.groupby(["condicion", "t_h"], observed=True)[["q_mediana", "frac_fluida", "rapidez"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
