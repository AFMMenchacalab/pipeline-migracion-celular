"""
PASO 3 (v2): todos los estadísticos de motilidad y persistencia, POR
PELÍCULA (o experimento), en paralelo (un proceso por película).

Unidad de réplica: la película / experimento (igual que v1 desde el arreglo
#6). Todo lo que se calcula por célula queda en por_celula.csv para
figuras, pero cada número poblacional se calcula dentro de cada película
y después se agrega entre películas (17_inferencia.py).

Qué se calcula (y de dónde sale):
  Clásicos (v1, con mejoras documentadas en lib/motilidad.py):
    TEA-MSD, alfa corto/largo, PRW con ruido de localización (D, P, sigma,
    S), VACF, espectro de potencias (FPS), rapidez, direccionalidad a
    ventana fija de 1 h, distribución de ángulos de giro, acoplamiento
    rapidez-persistencia (Maiuri et al. 2015, citado en Liu 2024),
    correlación espacial de direcciones entre células (migración colectiva),
    morfología.
  Entropía (lib/entropia.py):
    SE del FPS (Liu 2021): por célula en la trayectoria completa (como el
      paper) y en ventanas de largo fijo con corrección por ruido blanco.
    SE(t) por wavelet (Liu 2021): mapa célula x tiempo, promedio por
      célula y curva promedio en el tiempo.
    EAD(tau) (Liu 2024): por célula (12 bins) y del ensamble de la película
      (ángulos de todas las células juntas, 36 bins), cruda y corregida.
    TL1 y EAD1 por célula (para la distribución conjunta, JPD).
    EAD(t) por ventana deslizante (1 h) -> mapa célula x tiempo y
      porcentajes en [0,0.3), [0.3,0.6), [0.6,1] como en el suplemento de
      Liu 2024.
    Correlación de SE(t) entre pares de células vecinas vs lejanas
      (Liu 2021 Fig. 8, migración correlacionada).
  Sensibilidad de EAD1, SE y cos(giro) al paso de análisis (CAMAD 30 s ...
    15 min; BF y núcleos 5 ... 20 min).

Uso:
    ../venv/bin/python 16_estadisticas.py --dataset bf --variante dist_tam
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, CAMAD_CONDICION, CAMAD_LINEA, res_dir  # noqa: E402
from lib import entropia as E  # noqa: E402
from lib import motilidad as M  # noqa: E402
from lib.trayectorias import tabla_segmentos  # noqa: E402

PARAMS = {
    # paso: frames entre puntos del análisis (Delta = paso*dt = 5 min en ambos)
    "bf": dict(paso=1, msd_max_lag=60, alfa_corto=(5, 30), alfa_largo=(60, 240)),
    "sirdna": dict(paso=1, msd_max_lag=60, alfa_corto=(5, 30), alfa_largo=(60, 240)),
    "camad": dict(paso=10, msd_max_lag=300, alfa_corto=(0.5, 5), alfa_largo=(30, 120)),
}
COMUN = dict(
    min_vel=12,            # mínimo de velocidades por segmento para análisis por célula (1 h)
    tau_max_cel=12,        # EAD por célula hasta 1 h de lag
    bins_cel=12,           # 30 grados por bin (pocos ángulos por célula)
    tau_max_ens=36,        # EAD del ensamble hasta 3 h
    bins_ens=36,           # 10 grados por bin (como Liu 2024)
    umbral_tl1=0.95,       # umbral sobre EAD CORREGIDA por célula
    umbral_tl1_ens=0.99,   # umbral sobre EAD corregida del ensamble
    ventana_ead=12,        # EAD(t): 12 ángulos = 1 h
    bins_ventana=8,
    L_se=16,               # SE en ventanas de 16 velocidades (80 min)
    min_vel_wavelet=24,    # SE(t) wavelet: >= 2 h
    ventana_dr=12,         # direccionalidad en ventanas de 1 h
    bins_corr_um=np.arange(0, 325, 25.0),
    vecino_um=60.0, lejano_um=200.0,
)
# Sensibilidad al paso de análisis. Las simulaciones (19_simulaciones_metodos.py)
# muestran que con el ruido de localización del centroide (sigma ~ 2 um en
# BF) y pasos de 5 min, la EAD1 queda cerca de 1 aunque la persistencia real
# sea larga: el ruido domina el ángulo entre pasos cortos. Con pasos más
# largos el desplazamiento crece y el ruido pesa menos (a costa de menos
# pasos por célula). Se mide explícitamente en todos los datasets.
PASOS_SENSIB = {"camad": [1, 2, 4, 10, 20, 30], "bf": [1, 2, 3, 4], "sirdna": [1, 2, 3, 4]}


def morfologia_celula(g):
    g = g[~g["borde"]]
    if len(g) == 0:
        return {}
    return {"area_um2": g.area_um2.median(), "aspect_ratio": g.aspect_ratio.median(),
            "circularidad": g.circularidad.median(), "solidez": g.solidity.median()}


def analizar_pelicula(args):
    ds, movie, tr, det_frames, area_campo_mm2 = args
    P = {**PARAMS[ds], **COMUN}
    dt = DATASETS[ds]["dt_s"]
    paso = P["paso"]
    D_s = dt * paso                  # intervalo de análisis (s)
    D_min = D_s / 60
    t0 = time.time()

    # --- segmentos a resolución completa (MSD) y al paso de análisis
    segs_full = tabla_segmentos(tr, paso=1, min_puntos=4)
    segs = tabla_segmentos(tr, paso=paso, min_puntos=P["min_vel"] + 1)

    # ---------------------------------------------------------------- MSD / PRW
    msd, npares, ncel = M.tea_msd([s["xy"] for s in segs_full], P["msd_max_lag"])
    tau = np.arange(1, P["msd_max_lag"] + 1) * dt
    prw = M.ajustar_prw(tau, msd)
    a_c = M.alfa_rango(tau / 60, msd, *P["alfa_corto"])
    a_l = M.alfa_rango(tau / 60, msd, *P["alfa_largo"])
    curva_msd = pd.DataFrame({"movie": movie, "tau_s": tau, "msd_um2": msd, "pares": npares, "celulas": ncel})

    # ---------------------------------------------------------------- por célula
    vels = []
    celulas = []
    se_t_filas = []
    ead_t_filas = []
    ang_pool = {t: [] for t in range(1, P["tau_max_ens"] + 1)}
    rap_ang = []
    wav_series = {}
    for s in segs:
        xy, fr = s["xy"], s["frames"]
        v = np.diff(xy, axis=0) / D_s          # um/s
        vels.append(v)
        rap = np.linalg.norm(v, axis=1) * 60   # um/min
        g = tr[tr.track_id == s["track_id"]]
        g = g[(g.frame >= fr[0]) & (g.frame <= fr[-1])]
        c = {"movie": movie, "track_id": s["track_id"], "seg": s["seg"],
             "t_inicio_s": fr[0] * dt, "duracion_s": (fr[-1] - fr[0]) * dt, "n_vel": len(v),
             "x0_um": xy[0, 0], "y0_um": xy[0, 1],
             "rapidez_um_min": rap.mean(), "rapidez_mediana_um_min": np.median(rap),
             "direccionalidad_1h": M.direccionalidad_ventana(xy, P["ventana_dr"]),
             "desplazamiento_neto_um": float(np.linalg.norm(xy[-1] - xy[0]))}
        c.update(morfologia_celula(g))
        # SE Fourier
        c["se_fps_completa"] = E.se_fps(v)
        c["se_fps_completa_corr"] = c["se_fps_completa"] / E.se_nula(len(v))
        sev, nw = E.se_ventanas(v, P["L_se"])
        c["se_ventana"] = sev
        c["se_ventana_corr"] = sev / E.se_nula(P["L_se"]) if nw else np.nan
        # EAD por célula
        cr, co, ms = E.ead_perfil(v, P["tau_max_cel"], P["bins_cel"])
        c["ead1"] = cr[0]
        c["ead1_corr"] = co[0]
        c["tl1_min"] = E.tl1(co, P["umbral_tl1"]) * D_min
        c["tl1_censurado"] = bool(np.isnan(c["tl1_min"]))
        for t in range(P["tau_max_cel"]):
            c[f"ead_tau{t + 1}_corr"] = co[t]
        # ángulos para el ensamble y acoplamiento rapidez-giro
        for t in ang_pool:
            ang_pool[t].append(E.angulos_lag(v, t))
        th1 = E.angulos_lag(v, 1)
        if len(th1) == len(v) - 1:
            rap_ang.append(np.column_stack([(rap[:-1] + rap[1:]) / 2, np.cos(th1)]))
        # SE(t) wavelet
        if len(v) >= P["min_vel_wavelet"]:
            sw = E.se_wavelet(v, D_s)
            nul = E.se_wavelet_nula(len(v), D_s)
            c["se_wavelet_media"] = np.nanmean(sw)
            c["se_wavelet_media_corr"] = np.nanmean(sw / nul)
            tt = (fr[:-1] + fr[1:]) / 2 * dt   # tiempo del medio de cada paso
            for ti, val, vn in zip(tt, sw, sw / nul):
                if np.isfinite(val):
                    se_t_filas.append((movie, s["track_id"], s["seg"], ti, val, vn))
            wav_series[(s["track_id"], s["seg"])] = (fr[0], sw, xy)
        # EAD(t) ventana deslizante
        cen, ecr, eco = E.ead_temporal(v, P["ventana_ead"], P["bins_ventana"])
        for ci, a1, a2 in zip(cen, ecr, eco):
            ead_t_filas.append((movie, s["track_id"], s["seg"], (fr[0] + ci * paso) * dt, a1, a2))
        celulas.append(c)
    cel = pd.DataFrame(celulas)

    # ---------------------------------------------------------------- ensamble
    ead_ens = []
    for t, lst in ang_pool.items():
        th = np.concatenate(lst) if lst else np.empty(0)
        m = len(th)
        if m < 30:
            ead_ens.append((movie, t, t * D_min, np.nan, np.nan, m, np.nan, np.nan))
            continue
        e = E.entropia_norm(E.histograma_angulos(th, P["bins_ens"]))
        col = f"ead_tau{t}_corr"
        media_cel = cel[col].mean() if col in cel else np.nan
        sd_cel = cel[col].std() if col in cel else np.nan
        ead_ens.append((movie, t, t * D_min, e, e / E.ead_nula(m, P["bins_ens"]), m, media_cel, sd_cel))
    ead_ens = pd.DataFrame(ead_ens, columns=["movie", "tau", "tau_min", "ead_ens", "ead_ens_corr", "n_angulos",
                                             "ead_cel_media_corr", "ead_cel_sd_corr"])
    # PDF 3D de ángulos (tau x ángulo) para la figura tipo Liu 2024 Fig. 2A
    pdf3d = np.array([np.histogram(np.concatenate(ang_pool[t]) if ang_pool[t] else [], bins=P["bins_ens"],
                                   range=(-np.pi, np.pi), density=True)[0]
                      for t in ang_pool])

    vac, _ = M.vacf_ensamble(vels, 24)
    f, fps = M.fps_ensamble(vels, P["L_se"], D_s / 60)   # frecuencia en 1/min
    curva_vacf = pd.DataFrame({"movie": movie, "tau_min": np.arange(len(vac)) * D_min, "vacf": vac})
    curva_fps = pd.DataFrame({"movie": movie, "f_1_min": f, "fps_um2_min": fps}) if f is not None else None

    # muestra de pasos (rapidez, cos giro, ángulo) para la figura tipo Liu 2024 Fig. 1B-C
    muestra = pd.DataFrame(columns=["movie", "rapidez_um_min", "cos_giro", "angulo_rad"])
    if rap_ang:
        ra_all = np.concatenate(rap_ang)
        idx = np.random.default_rng(movie).permutation(len(ra_all))[:3000]
        muestra = pd.DataFrame({"movie": movie, "rapidez_um_min": ra_all[idx, 0], "cos_giro": ra_all[idx, 1]})
    # acoplamiento rapidez-persistencia
    if rap_ang:
        ra = np.concatenate(rap_ang)
        rho_rap_giro = spearmanr(ra[:, 0], ra[:, 1]).statistic if len(ra) > 20 else np.nan
    else:
        rho_rap_giro = np.nan
    ang1 = np.concatenate(ang_pool[1]) if ang_pool[1] else np.empty(0)

    # correlación espacial de direcciones (pares simultáneos)
    bins = P["bins_corr_um"]
    sc = np.zeros(len(bins) - 1)
    cc = np.zeros(len(bins) - 1)
    filas_v = []
    for s in segs:
        v = np.diff(s["xy"], axis=0) / D_s
        mid = (s["xy"][:-1] + s["xy"][1:]) / 2
        for fr, p, vv in zip(s["frames"][:-1], mid, v):
            filas_v.append((fr, p[0], p[1], vv[0], vv[1]))
    if filas_v:
        fv = pd.DataFrame(filas_v, columns=["frame", "x", "y", "vx", "vy"])
        for _, g in fv.groupby("frame"):
            s_, c_ = M.correlacion_espacial_velocidad(g[["x", "y", "vx", "vy"]].to_numpy(), bins)
            sc += s_
            cc += c_
    corr_esp = pd.DataFrame({"movie": movie, "r_um": (bins[:-1] + bins[1:]) / 2,
                             "corr_direccion": np.where(cc > 50, sc / np.maximum(cc, 1), np.nan), "pares": cc})

    # correlación de SE(t) entre pares vecinos / lejanos
    rng = np.random.default_rng(movie)
    claves = list(wav_series)
    cerca, lejos = [], []
    if len(claves) > 1:
        ini = np.array([wav_series[k][0] for k in claves])
        for i in range(len(claves)):
            fi, si, xyi = wav_series[claves[i]]
            for j in rng.permutation(np.arange(i + 1, len(claves)))[:60]:
                fj, sj, xyj = wav_series[claves[j]]
                a = max(fi, fj)
                b = min(fi + len(si), fj + len(sj))
                if b - a < 12:
                    continue
                x1 = si[(a - fi) // paso:(b - fi) // paso]
                x2 = sj[(a - fj) // paso:(b - fj) // paso]
                n = min(len(x1), len(x2))
                x1, x2 = x1[:n], x2[:n]
                ok = np.isfinite(x1) & np.isfinite(x2)
                if ok.sum() < 10:
                    continue
                d = np.linalg.norm(xyi[(a - fi) // paso:(a - fi) // paso + n].mean(0)
                                   - xyj[(a - fj) // paso:(a - fj) // paso + n].mean(0))
                r = spearmanr(x1[ok], x2[ok]).statistic
                if not np.isfinite(r):
                    continue
                if d < P["vecino_um"]:
                    cerca.append(r)
                elif d > P["lejano_um"]:
                    lejos.append(r)
        del ini

    # EAD(t): porcentajes por intervalo (Liu 2024 suplemento)
    ead_t = pd.DataFrame(ead_t_filas, columns=["movie", "track_id", "seg", "t_s", "ead", "ead_corr"])
    se_t = pd.DataFrame(se_t_filas, columns=["movie", "track_id", "seg", "t_s", "se", "se_corr"])

    def pct(x):
        x = x.dropna()
        if len(x) == 0:
            return (np.nan,) * 3
        return tuple(np.histogram(x.clip(0, 1), bins=[0, 0.3, 0.6, 1.0001])[0] / len(x) * 100)

    p_cr = pct(ead_t.ead) if len(ead_t) else (np.nan,) * 3
    p_co = pct(ead_t.ead_corr) if len(ead_t) else (np.nan,) * 3

    e1 = ead_ens.set_index("tau")
    tl1_ens = E.tl1(e1["ead_ens_corr"].to_numpy(), P["umbral_tl1_ens"]) * D_min
    peli = {
        "movie": movie, "n_segmentos": len(cel), "n_tracks": tr.track_id.nunique(),
        "densidad_cel_mm2": det_frames / area_campo_mm2,
        "rapidez_um_min": cel.rapidez_um_min.mean() if len(cel) else np.nan,
        "direccionalidad_1h": cel.direccionalidad_1h.mean() if len(cel) else np.nan,
        "prw_D_um2_min": prw["D"] * 60, "prw_P_min": prw["P"] / 60, "prw_sigma_um": prw["sigma"],
        "prw_S_um_min": prw["S"] * 60, "prw_r2": prw["r2"],
        "alfa_corto": a_c, "alfa_largo": a_l,
        "vacf_lag1": vac[1] if len(vac) > 1 else np.nan,
        "rho_rapidez_cosgiro": rho_rap_giro,
        "cos_giro_medio": float(np.mean(np.cos(ang1))) if len(ang1) else np.nan,
        "se_fps_completa": cel.se_fps_completa.mean(), "se_fps_completa_corr": cel.se_fps_completa_corr.mean(),
        "se_ventana": cel.se_ventana.mean(), "se_ventana_corr": cel.se_ventana_corr.mean(),
        "se_wavelet": cel.get("se_wavelet_media", pd.Series(dtype=float)).mean(),
        "se_wavelet_corr": cel.get("se_wavelet_media_corr", pd.Series(dtype=float)).mean(),
        "ead1_cel": cel.ead1.mean(), "ead1_cel_corr": cel.ead1_corr.mean(),
        "tl1_cel_mediana_min": cel.tl1_min.median(), "tl1_frac_censurado": cel.tl1_censurado.mean(),
        "ead1_ens": e1.loc[1, "ead_ens"], "ead1_ens_corr": e1.loc[1, "ead_ens_corr"],
        "tl1_ens_min": tl1_ens,
        "ead_t_pct_0_03": p_cr[0], "ead_t_pct_03_06": p_cr[1], "ead_t_pct_06_1": p_cr[2],
        "ead_t_corr_pct_0_03": p_co[0], "ead_t_corr_pct_03_06": p_co[1], "ead_t_corr_pct_06_1": p_co[2],
        "corr_dir_0_50um": np.nanmean(corr_esp.corr_direccion.iloc[:2]),
        "corr_dir_100_200um": np.nanmean(corr_esp.corr_direccion.iloc[4:8]),
        "corr_se_vecinos": float(np.mean(cerca)) if cerca else np.nan,
        "corr_se_lejanos": float(np.mean(lejos)) if lejos else np.nan,
        "n_pares_vecinos": len(cerca), "n_pares_lejanos": len(lejos),
        "area_um2": cel.get("area_um2", pd.Series(dtype=float)).mean(),
        "aspect_ratio": cel.get("aspect_ratio", pd.Series(dtype=float)).mean(),
        "circularidad": cel.get("circularidad", pd.Series(dtype=float)).mean(),
        "solidez": cel.get("solidez", pd.Series(dtype=float)).mean(),
        "segundos": time.time() - t0,
    }

    # ---------------------------------------------------------------- sensibilidad al paso (CAMAD)
    sens = []
    if True:
        for pz in PASOS_SENSIB[ds]:
            sg = tabla_segmentos(tr, paso=pz, min_puntos=10)
            e1s, ses, pool = [], [], []
            for s in sg:
                v = np.diff(s["xy"], axis=0) / (dt * pz)
                cr, co, _ = E.ead_perfil(v, 1, COMUN["bins_cel"])
                e1s.append(co[0])
                sv, nw = E.se_ventanas(v, 12)
                if nw:
                    ses.append(sv / E.se_nula(12))
                pool.append(E.angulos_lag(v, 1))
            th = np.concatenate(pool) if pool else np.empty(0)
            sens.append({"movie": movie, "paso_frames": pz, "paso_min": pz * dt / 60, "n_seg": len(sg),
                         "ead1_cel_corr": np.nanmean(e1s) if e1s else np.nan,
                         "se_ventana12_corr": np.nanmean(ses) if ses else np.nan,
                         "ead1_ens_corr": (E.entropia_norm(E.histograma_angulos(th, 36)) / E.ead_nula(len(th), 36))
                         if len(th) > 30 else np.nan,
                         "cos_giro_medio": float(np.mean(np.cos(th))) if len(th) else np.nan})
    return dict(peli=peli, cel=cel, msd=curva_msd, vacf=curva_vacf, fps=curva_fps, ead=ead_ens, pasos=muestra,
                corr=corr_esp, se_t=se_t, ead_t=ead_t, pdf3d=pdf3d, sens=pd.DataFrame(sens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variante", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    ds = args.dataset
    t0 = time.time()
    tr = pd.read_csv(res_dir(ds, "tracking", args.variante) / "tracks.csv.gz")
    det = pd.read_csv(res_dir(ds) / "detecciones.csv.gz", usecols=["movie", "frame"])
    det_por_frame = det.groupby("movie").size() / det.groupby("movie")["frame"].nunique()
    # área del campo en mm2 (tamaño de máscara x calibración)
    from lib.fuentes import listar_frames, cargar_mascara
    it = listar_frames(ds)[0]
    h, w = cargar_mascara(ds, it["movie"], it["frame"]).shape
    area_mm2 = h * w * DATASETS[ds]["um_per_px"] ** 2 / 1e6
    tareas = [(ds, mv, g, det_por_frame[mv], area_mm2) for mv, g in tr.groupby("movie")]
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(analizar_pelicula, tareas))
    out = res_dir(ds, "estadisticas", args.variante)
    peli = pd.DataFrame([r["peli"] for r in res])
    if ds == "camad":
        peli["condicion"] = peli.movie.map(CAMAD_CONDICION)
        peli["linea"] = peli.movie.map(CAMAD_LINEA)
    peli.to_csv(out / "por_pelicula.csv", index=False)
    for k in ["cel", "msd", "vacf", "fps", "ead", "corr", "sens", "pasos"]:
        partes = [r[k] for r in res if r[k] is not None and len(r[k])]
        if partes:
            d = pd.concat(partes, ignore_index=True)
            if ds == "camad" and "movie" in d:
                d["condicion"] = d.movie.map(CAMAD_CONDICION)
            nombre = {"cel": "por_celula", "msd": "msd", "vacf": "vacf", "fps": "fps", "ead": "ead_tau",
                      "corr": "corr_espacial", "sens": "sensibilidad_paso", "pasos": "pasos_muestra"}[k]
            d.to_csv(out / f"{nombre}.csv", index=False)
    pd.concat([r["se_t"] for r in res], ignore_index=True).to_csv(out / "se_wavelet_tiempo.csv.gz", index=False)
    pd.concat([r["ead_t"] for r in res], ignore_index=True).to_csv(out / "ead_tiempo.csv.gz", index=False)
    np.savez_compressed(out / "pdf3d_angulos.npz", **{f"m{r['peli']['movie']:02d}": r["pdf3d"] for r in res})
    (out / "parametros.json").write_text(json.dumps(
        {**{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in COMUN.items()}, **PARAMS[ds]}, indent=2))
    print(peli.drop(columns=["segundos"]).round(3).to_string())
    print(f"[{ds}/{args.variante}] listo en {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
