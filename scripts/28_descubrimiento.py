"""
Búsqueda exploratoria de relaciones no evidentes (pedido: "que una IA busque
relaciones que no estamos viendo").

Lo que hace posible esto: en el dataset brightfield tenemos, para CADA célula
y en cada instante, su contorno (brightfield) Y su núcleo (SiR-DNA). Eso
permite cruzar movimiento, forma celular, forma nuclear, contenido de ADN
(indicador de la fase del ciclo celular), posición del núcleo dentro de la
célula (polaridad) y contacto con vecinas.

Para no "descubrir" casualidades (con cientos de pruebas, alguna sale
significativa por azar):
  1. DESCUBRIMIENTO en las películas impares y VALIDACIÓN en las pares: una
     relación solo se reporta si aparece en ambas mitades, con el mismo signo.
  2. Las películas son la réplica (correlación dentro de cada película,
     prueba de Wilcoxon de las 8 correlaciones contra 0) y se controla la
     tasa de falsos descubrimientos (Benjamini-Hochberg) en el descubrimiento.
  3. Las relaciones triviales (una variable definida a partir de la otra,
     p. ej. alargamiento y circularidad) se marcan y no cuentan como hallazgo.
  4. Modelo predictivo (gradient boosting) evaluado en películas que no vio,
     con importancia por permutación: ¿qué variables predicen la rapidez y la
     persistencia del paso siguiente, incluidas relaciones no lineales?

Salidas: resultados/v2/descubrimiento/*.csv, figuras desc_*.png
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, res_dir  # noqa: E402
from lib.fuentes import listar_frames, leer_frame, cargar_mascara  # noqa: E402
from lib.metricas_seg import asignar_nucleos  # noqa: E402

OUT = res_dir("descubrimiento")
UM = DATASETS["bf"]["um_per_px"]


def nucleos_pelicula(movie):
    """Para cada célula BF con exactamente un núcleo: rasgos del núcleo."""
    items_n = {i["frame"]: i for i in listar_frames("sirdna") if i["movie"] == movie}
    filas = []
    for f, it in items_n.items():
        cel = cargar_mascara("bf", movie, f)
        nuc = cargar_mascara("sirdna", movie, f)
        img = leer_frame(it).astype(float)
        cel_de_nuc, n_por_cel = asignar_nucleos(cel, nuc)
        p = pd.DataFrame(regionprops_table(nuc, intensity_image=img, properties=(
            "label", "centroid", "area", "eccentricity", "solidity", "major_axis_length", "minor_axis_length",
            "intensity_mean")))
        p["label_bf"] = cel_de_nuc[p.label.to_numpy() - 1]
        p = p[p.label_bf > 0]
        p = p[n_por_cel[p.label_bf.to_numpy() - 1] == 1]
        fondo = np.percentile(img, 5)
        p["adn_integrado"] = (p.intensity_mean - fondo) * p.area
        p["frame"] = f
        filas.append(p)
    d = pd.concat(filas, ignore_index=True).rename(columns={
        "centroid-0": "ny", "centroid-1": "nx", "area": "nuc_area", "eccentricity": "nuc_excentricidad",
        "solidity": "nuc_solidez", "intensity_mean": "nuc_intensidad"})
    d["nuc_alargamiento"] = d.major_axis_length / d.minor_axis_length.replace(0, np.nan)
    d["movie"] = movie
    # contenido de ADN normalizado por película (la mediana ~ G1): ~1 en G1, ~2 en G2/M
    d["adn_rel"] = d.adn_integrado / d.groupby("frame").adn_integrado.transform("median")
    return d.drop(columns=["label", "major_axis_length", "minor_axis_length"])


def tabla_pasos():
    pasos = pd.read_csv(res_dir("morfoespacio") / "pasos_bf.csv.gz")
    tr = pd.read_csv(res_dir("bf", "tracking", "dist_tam") / "tracks.csv.gz",
                     usecols=["movie", "track_id", "frame", "label", "x", "y", "area", "major_axis_length"])
    with ProcessPoolExecutor(16) as ex:
        nuc = pd.concat(list(ex.map(nucleos_pelicula, range(1, 17))), ignore_index=True)
    nuc.to_csv(OUT / "nucleos_por_celula.csv.gz", index=False)
    t = tr.merge(nuc, left_on=["movie", "frame", "label"], right_on=["movie", "frame", "label_bf"], how="left")
    t = t.merge(pasos, on=["movie", "track_id", "frame"], how="inner")
    # polaridad: vector centro de la célula -> centro del núcleo (um), relativo al tamaño de la célula
    t["off_x"] = (t.nx - t.x) * UM
    t["off_y"] = (t.ny - t.y) * UM
    t["off_rel"] = np.hypot(t.off_x, t.off_y) / (t.major_axis_length * UM / 2)
    t["nuc_frac_area"] = t.nuc_area / t.area
    # dirección del paso siguiente
    t = t.sort_values(["track_id", "frame"])
    g = t.groupby("track_id")
    vx = g.x.shift(-1) - t.x
    vy = g.y.shift(-1) - t.y
    nv = np.hypot(vx, vy)
    no = np.hypot(t.off_x, t.off_y)
    t["cos_nucleo_mov"] = (t.off_x * vx * UM + t.off_y * vy * UM) / (no * nv * UM)
    t.loc[(nv == 0) | (no == 0), "cos_nucleo_mov"] = np.nan
    # vecina más cercana y contacto
    filas = []
    for (mv, fr), gg in t.groupby(["movie", "frame"]):
        xy = gg[["x", "y"]].to_numpy() * UM
        if len(xy) < 2:
            continue
        d, j = cKDTree(xy).query(xy, k=2)
        r_eq = np.sqrt(gg.area_um2.to_numpy() / np.pi)
        filas.append(pd.DataFrame({"idx": gg.index, "dist_vecina_um": d[:, 1],
                                   "contacto": d[:, 1] < 1.1 * (r_eq + r_eq[j[:, 1]])}))
    v = pd.concat(filas).set_index("idx")
    t = t.join(v)
    t["t_h"] = t.frame * DATASETS["bf"]["dt_s"] / 3600
    t["rapidez_sig"] = g.rapidez.shift(-1)          # rapidez del paso siguiente
    t["cos_giro_sig"] = g.cos_giro.shift(-1)
    t.to_csv(OUT / "pasos_combinados.csv.gz", index=False)
    return t


VARS = ["rapidez", "cos_giro", "cos_eje", "q", "alargamiento", "solidez", "area_um2", "densidad_local",
        "dist_vecina_um", "nuc_area", "nuc_alargamiento", "nuc_solidez", "nuc_intensidad", "adn_rel", "off_rel",
        "nuc_frac_area", "cos_nucleo_mov", "t_h"]
TRIVIALES = {frozenset(x) for x in [("q", "solidez"), ("q", "alargamiento"), ("alargamiento", "solidez"), ("q", "area_um2"),
             ("densidad_local", "dist_vecina_um"), ("nuc_area", "nuc_frac_area"), ("area_um2", "nuc_frac_area"),
             ("nuc_intensidad", "adn_rel"), ("nuc_area", "adn_rel"), ("alargamiento", "cos_eje"),
             ("rapidez", "cos_eje"), ("nuc_alargamiento", "nuc_solidez")]}


def cribado(t):
    """Correlación de Spearman dentro de cada película para cada par de
    variables; descubrimiento en películas impares, validación en pares."""
    from itertools import combinations
    filas = []
    for a, b in combinations(VARS, 2):
        rhos = {}
        for mv, g in t.groupby("movie"):
            g = g[[a, b]].dropna()
            if len(g) > 50:
                rhos[mv] = stats.spearmanr(g[a], g[b]).statistic
        imp = np.array([v for k, v in rhos.items() if k % 2 == 1])
        par = np.array([v for k, v in rhos.items() if k % 2 == 0])
        if len(imp) < 6 or len(par) < 6:
            continue
        filas.append({"a": a, "b": b, "rho_desc": np.median(imp), "p_desc": stats.wilcoxon(imp).pvalue,
                      "rho_val": np.median(par), "p_val": stats.wilcoxon(par).pvalue,
                      "rho_todas": np.median(list(rhos.values())), "trivial": frozenset((a, b)) in TRIVIALES})
    c = pd.DataFrame(filas)
    # Benjamini-Hochberg en el descubrimiento
    p = c.p_desc.to_numpy()
    o = np.argsort(p)
    q = np.empty_like(p)
    q[o] = np.minimum.accumulate((p[o] * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
    c["q_desc"] = np.minimum(q, 1)
    c["replica"] = (c.q_desc < 0.05) & (c.p_val < 0.05) & (np.sign(c.rho_desc) == np.sign(c.rho_val))
    return c.sort_values("rho_todas", key=np.abs, ascending=False)


def modelo(t, objetivo):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import GroupKFold
    feats = ["q", "alargamiento", "solidez", "area_um2", "densidad_local", "dist_vecina_um", "nuc_area",
             "nuc_alargamiento", "nuc_solidez", "adn_rel", "off_rel", "nuc_frac_area", "cos_nucleo_mov", "t_h",
             "rapidez", "cos_giro"]
    feats = [f for f in feats if f != objetivo]
    d = t[feats + [objetivo, "movie"]].dropna()
    d = d.sample(min(len(d), 120000), random_state=0)
    X, y, gr = d[feats].to_numpy(), d[objetivo].to_numpy(), d.movie.to_numpy()
    r2, imps = [], []
    for tr_i, te_i in GroupKFold(4).split(X, y, gr):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X[tr_i], y[tr_i])
        pred = m.predict(X[te_i])
        r2.append(1 - ((y[te_i] - pred) ** 2).sum() / ((y[te_i] - y[te_i].mean()) ** 2).sum())
        pi = permutation_importance(m, X[te_i][:20000], y[te_i][:20000], n_repeats=3, random_state=0)
        imps.append(pi.importances_mean)
    imp = pd.DataFrame({"variable": feats, "importancia": np.mean(imps, 0)}).sort_values("importancia", ascending=False)
    return float(np.mean(r2)), imp


def main():
    t = tabla_pasos()
    print("pasos:", len(t), " con núcleo:", t.nuc_area.notna().sum())
    c = cribado(t)
    c.to_csv(OUT / "cribado_correlaciones.csv", index=False)
    rep = c[c.replica & ~c.trivial]
    print("\n=== relaciones que replican (no triviales), ordenadas por |rho| ===")
    print(rep.round(3).head(40).to_string(index=False))
    print(f"\n{len(c)} pares probados; {c.replica.sum()} replican ({rep.shape[0]} no triviales)")
    res = {}
    for obj in ("rapidez_sig", "cos_giro_sig"):
        r2, imp = modelo(t, obj)
        imp.to_csv(OUT / f"importancia_{obj}.csv", index=False)
        res[obj] = r2
        print(f"\nmodelo para {obj}: R2 en películas no vistas = {r2:.3f}")
        print(imp.round(4).head(10).to_string(index=False))
    pd.Series(res).to_csv(OUT / "r2_modelos.csv")


if __name__ == "__main__":
    main()
