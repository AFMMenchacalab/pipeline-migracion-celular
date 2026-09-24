"""
PASO 4 del pipeline: estadísticos de validación y dinámica de movimiento a
partir de las trayectorias de resultados/tracking/tracks.csv.

Dataset: brightfield MDA-MB-231 (Zenodo 10.5281/zenodo.10074471), 16 películas
independientes de 100 frames cada una, intervalo real ~300.02 s (~5 min),
calibración 0.6496 um/px (leída de metadatos ImageJ del TIFF). Es la ÚNICA
condición trackeada hasta ahora -> no hay grupos experimentales para comparar
entre sí, solo 16 campos (réplicas biológicas) de la misma condición.

===========================================================================
 ARREGLOS DE LA REVISIÓN DEL 2026-09-20 (ver ESTADO.md, sección "POR
 ARREGLAR") APLICADOS EN ESTA VERSIÓN:
===========================================================================
#4 Bug de huecos: TA-MSD, EA-MSD, VACF y ángulos de giro ahora usan la
   diferencia REAL de `frame` como lag/tau, nunca el índice de fila del
   array. Antes, una trayectoria con un hueco (gap-closing) desalineaba
   todo lo que venía después de ese hueco.
#6 Unidad de replicación correcta: la célula individual NO es una réplica
   independiente (todas las células de una misma película comparten campo,
   pocillo, iluminación...). Por eso TODA la estadística poblacional (curva
   EA-MSD, régimen de difusión α, ajuste PRW, VACF) se calcula primero
   DENTRO de cada una de las 16 películas por separado, y solo después se
   agregan esas 16 estimaciones independientes (resumen_por_pelicula.csv es
   el dataset real, n=16 — no n=miles de células).
#3 Barras de error: en vez del error analítico del OLS/curve_fit sobre la
   curva pooled (que asume puntos de MSD independientes cuando en realidad
   comparten células entre sí), el error reportado es la dispersión ENTRE
   PELÍCULAS de sus 16 estimaciones independientes, vía bootstrap sobre esas
   16 películas (resample con reemplazo de películas, no de células ni de
   puntos del MSD). Esto ya incorpora el arreglo #6: al ser películas
   genuinamente independientes, el bootstrap sobre ellas es estadísticamente
   válido (a diferencia de un bootstrap sobre células pooled, que seguiría
   subestimando el error).
#2 Sesgo de supervivencia: la cohorte de células que siguen vivas (siguen
   teniendo una detección) se achica a medida que τ crece. Cada curva
   EA-MSD (por película) se trunca en el primer τ donde la cohorte cae por
   debajo de SURVIVORSHIP_MIN_FRAC del tamaño que tenía en τ=1, antes de
   ajustar nada sobre ella.
#1 Régimen por tramos: el exponente α ya NO se ajusta como un único número
   global sobre todo el rango de τ. Se detectan automáticamente (por fuerza
   bruta, minimizando la suma de residuos) hasta 3 tramos en la curva
   log-log de la "gran media" (promedio de las 16 curvas EA-MSD por
   película), y α se ajusta por separado en cada tramo, en cada película.
#5 Gráficas de barra en escala log -> se resuelve en 05_figuras.py (ahí es
   donde se dibujan), pero este script deja todo lo necesario: por_celula.csv
   y resumen_por_pelicula.csv con columna `movie` para agrupar por película
   en vez de una sola barra pooled.

Salidas en resultados/estadisticas/:
    calibracion.txt
    validacion.md
    msd_ta.csv                        TA-MSD por célula (gap-fijado)
    msd_poblacion_por_pelicula.csv    EA-MSD por película (gap-fijado, truncado por supervivencia)
    msd_poblacion.csv                 gran media ± SEM de EA-MSD ENTRE las 16 películas
    alpha_por_pelicula_y_segmento.csv α, K_α por película y por tramo de régimen
    prw_por_pelicula.csv              ajuste PRW (A, τ_p, D) por película
    vacf_por_pelicula.csv             VACF por película (gap-fijado)
    vacf.csv                          gran media ± SEM de VACF entre películas
    angulos_giro.csv                  ángulos de giro (pooled, gap-fijado)
    rapidez_instantanea.csv           rapidez paso a paso (pooled, gap-fijado, con columna movie)
    morfologia_por_frame.csv          área/aspect ratio/irregularidad por célula y frame
    por_celula.csv                    una fila por célula (con columna movie) para las
                                       gráficas de caja/violín agrupadas por película
    resumen_por_pelicula.csv          UNA FILA POR PELÍCULA (n=16): el dataset real
                                       sobre el que se calcula toda inferencia poblacional
    resumen_bootstrap.csv             media/SEM/IC95% (bootstrap entre películas) de cada
                                       métrica de resumen_por_pelicula.csv -- son los números
                                       "oficiales" a citar
    resumen.md (en resultados/)       resumen narrativo con los números clave
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import stats
from scipy.optimize import curve_fit
from skimage.measure import regionprops_table

ROOT = Path(__file__).resolve().parent.parent
TRACKS_CSV = ROOT / "resultados" / "tracking" / "tracks.csv"
MASKS_DIR = ROOT / "resultados" / "segmentacion"
DATASET_DIR = ROOT / "datasets" / "brightfield-mdamb231" / "1.1-training-source-BF-1600"
OUT_DIR = ROOT / "resultados" / "estadisticas"

FRAMES_PER_MOVIE = 100  # mismo valor y misma evidencia que en 02_tracking.py

# umbrales mínimos de longitud de track para cada tipo de ajuste INDIVIDUAL (por célula)
MIN_LEN_ALPHA_FIT = 15   # para ajuste log-log individual (alpha, K_alpha)
MIN_LEN_PRW_FIT = 20     # para ajuste PRW individual (tau_p, D, A)

MIN_CELLS_PER_TAU_MOVIE = 10   # piso de cohorte para un punto de EA-MSD/VACF DENTRO de una película
SURVIVORSHIP_MIN_FRAC = 0.5    # arreglo #2: cortar la curva EA-MSD cuando la cohorte cae < 50% de la de tau=1
MIN_MOVIES_PER_TAU = 8         # piso de películas contribuyendo para reportar un punto de la "gran media" (mitad de 16)
MIN_SEG_POINTS = 4             # mínimo de puntos de tau por tramo en el ajuste de régimen

N_BOOT = 2000
RNG = np.random.default_rng(0)


def read_calibration():
    # Se lee de una sola imagen: la resolución (um/px) y el intervalo entre
    # frames son del mismo microscopio/protocolo para las 16 películas (el
    # finterval varía en la 5ta cifra decimal entre películas -- ruido de
    # redondeo del instrumento, no una calibración distinta).
    with tifffile.TiffFile(DATASET_DIR / "0001.tif") as tif:
        tags = tif.pages[0].tags
        xres = tags["XResolution"].value  # (num, den) píxeles por micron
        px_per_um = xres[0] / xres[1]
        um_per_px = 1 / px_per_um
        desc = tags["ImageDescription"].value
    finterval_s = None
    for line in desc.splitlines():
        if line.startswith("finterval="):
            finterval_s = float(line.split("=")[1])
    return um_per_px, finterval_s


def prw_model(tau, A, tau_p):
    return A * (tau / tau_p - 1 + np.exp(-tau / tau_p))


def loglog_fit(tau, msd):
    """log(msd) = alpha*log(tau) + log(K_alpha). Devuelve alpha, K_alpha, r2."""
    mask = (tau > 0) & (msd > 0) & np.isfinite(msd)
    tau, msd = tau[mask], msd[mask]
    if len(tau) < 3:
        return None
    logt, logm = np.log(tau), np.log(msd)
    res = stats.linregress(logt, logm)
    return {
        "alpha": res.slope,
        "K_alpha": np.exp(res.intercept),
        "r2": res.rvalue ** 2,
        "n_puntos": len(tau),
    }


def fit_prw(tau, msd):
    mask = (tau > 0) & (msd > 0) & np.isfinite(msd)
    tau, msd = tau[mask], msd[mask]
    if len(tau) < 4:
        return None
    A0 = msd[-1] / tau[-1] if tau[-1] > 0 else 1.0
    tau_p0 = tau[len(tau) // 2]
    try:
        popt, pcov = curve_fit(
            prw_model, tau, msd, p0=[max(A0, 1e-6), max(tau_p0, 1e-3)],
            maxfev=20000, bounds=([1e-8, 1e-6], [np.inf, np.inf]),
        )
    except RuntimeError:
        return None
    A, tau_p = popt
    D = A / (4 * tau_p)
    pred = prw_model(tau, A, tau_p)
    ss_res = np.sum((msd - pred) ** 2)
    ss_tot = np.sum((msd - msd.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"A": A, "tau_p": tau_p, "D": D, "r2": r2, "n_puntos": len(tau)}


def compute_ta_msd_single(frames, xy, max_frac=4, min_len=8):
    """TA-MSD para UNA trayectoria (arreglo #4: usa diferencia real de frame
    como tau, no el índice de fila -- así una trayectoria con huecos no
    desalinea los tau posteriores al hueco). Vectorizado con numpy: matriz
    de todos los pares (i,j) de una vez en vez de un loop por tau."""
    n = len(xy)
    if n < min_len:
        return None
    span = int(frames[-1] - frames[0])
    tau_max = max(1, span // max_frac)

    dframe = frames[None, :] - frames[:, None]           # (n,n) diferencia real de frame: dframe[i,j] = frames[j]-frames[i]
    diff = xy[:, None, :] - xy[None, :, :]
    sqdisp = np.sum(diff ** 2, axis=-1)                   # (n,n) desplazamiento al cuadrado
    iu = np.triu_indices(n, k=1)                          # solo pares i<j (frames estrictamente crecientes)
    taus_all, sq_all = dframe[iu], sqdisp[iu]
    keep = taus_all <= tau_max
    taus_all, sq_all = taus_all[keep], sq_all[keep]
    if len(taus_all) == 0:
        return None

    order = np.argsort(taus_all)
    taus_sorted, sq_sorted = taus_all[order], sq_all[order]
    uniq_taus, idx_start = np.unique(taus_sorted, return_index=True)
    idx_end = np.append(idx_start[1:], len(sq_sorted))
    msds = np.array([sq_sorted[a:b].mean() for a, b in zip(idx_start, idx_end)])
    return uniq_taus.astype(int), msds


def compute_ea_msd_for_movie(movie_df):
    """EA-MSD(tau) = <|r_i(t_i^nacimiento + tau) - r_i(t_i^nacimiento)|^2>_i,
    tau relativo al nacimiento de cada trayectoria (arreglo #4: busca la
    posición en el frame real t_i^nacimiento+tau, no en la fila "tau" de la
    tabla -- eso es lo que rompía huecos). Corta la curva en el primer tau
    donde la cohorte cae debajo de SURVIVORSHIP_MIN_FRAC de la cohorte en
    tau=1 (arreglo #2, sesgo de supervivencia)."""
    frame_maps, start_frames, max_span = {}, {}, 0
    for track_id, g in movie_df.groupby("track_id"):
        g = g.sort_values("frame")
        frames = g["frame"].values
        xy = g[["x_um", "y_um"]].values
        frame_maps[track_id] = dict(zip(frames.tolist(), map(tuple, xy)))
        start_frames[track_id] = int(frames[0])
        max_span = max(max_span, int(frames[-1] - frames[0]))

    rows, n0 = [], None
    for tau in range(1, max_span + 1):
        sqs = []
        for track_id, fmap in frame_maps.items():
            sf = start_frames[track_id]
            r0, rt = fmap.get(sf), fmap.get(sf + tau)
            if r0 is not None and rt is not None:
                sqs.append((rt[0] - r0[0]) ** 2 + (rt[1] - r0[1]) ** 2)
        if tau == 1:
            n0 = len(sqs)
        if len(sqs) < MIN_CELLS_PER_TAU_MOVIE:
            continue
        if n0 and len(sqs) < SURVIVORSHIP_MIN_FRAC * n0:
            break  # arreglo #2: no seguir más allá de donde la cohorte se derrumbó
        rows.append({"tau_frames": tau, "n_celulas": len(sqs), "ea_msd_um2": float(np.mean(sqs))})
    return pd.DataFrame(rows), (n0 or 0)


def compute_vacf_speed_angles_for_movie(movie_df, finterval_s, max_lag):
    """VACF, rapidez instantánea y ángulos de giro para una película,
    respetando huecos reales de frame (arreglo #4). Vectorizado por
    trayectoria: para cada célula arma la matriz de todos los pares
    (paso_a, paso_b) de una sola vez con numpy en vez de un loop anidado en
    Python puro (con ~8000 células x hasta 100 frames, el loop puro tardaba
    demasiado)."""
    vacf_sum = np.zeros(max_lag + 1)
    vacf_count = np.zeros(max_lag + 1, dtype=np.int64)
    speed_rows, turning_angles = [], []

    for track_id, g in movie_df.groupby("track_id"):
        g = g.sort_values("frame")
        frames = g["frame"].values
        xy = g[["x_um", "y_um"]].values
        if len(xy) < 2:
            continue

        step_frames_all = frames[:-1]
        step_mask = np.diff(frames) == 1  # solo pasos entre frames REALMENTE consecutivos
        if not step_mask.any():
            continue
        v_all = np.diff(xy, axis=0) / finterval_s
        v = v_all[step_mask]
        f = step_frames_all[step_mask]  # frame de inicio de cada paso válido

        speeds = np.linalg.norm(v, axis=1)
        for s in speeds:
            speed_rows.append({"track_id": track_id, "rapidez_um_s": float(s)})

        m = len(v)
        if m > 0:
            lag_mat = f[None, :] - f[:, None]
            dot_mat = v @ v.T
            iu = np.triu_indices(m, k=0)  # incluye lag=0 (diagonal) y todo par b>=a
            lags, dots = lag_mat[iu], dot_mat[iu]
            keep = lags <= max_lag
            np.add.at(vacf_sum, lags[keep], dots[keep])
            np.add.at(vacf_count, lags[keep], 1)

        # ángulos de giro: solo entre pasos que además son consecutivos en
        # frame REAL (el paso que termina en f[i]+1 es exactamente el que
        # empieza el siguiente) -- si hubo un hueco entre medio, se descarta.
        for i in range(m - 1):
            if f[i] + 1 != f[i + 1]:
                continue
            v1, v2 = v[i], v[i + 1]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                turning_angles.append(float(np.degrees(np.arccos(cos_a))))

    return vacf_sum, vacf_count, speed_rows, turning_angles


def find_regime_breakpoints(tau_s, msd, max_segments=3, min_points=MIN_SEG_POINTS):
    """Arreglo #1: en vez de un único ajuste log-log global, busca por fuerza
    bruta los puntos de quiebre que minimizan la suma de residuos cuadrados
    de `k` ajustes lineales independientes en escala log-log (uno por
    régimen), probando todas las combinaciones válidas de cortes con al
    menos `min_points` puntos por tramo. Si no hay puntos suficientes para 3
    tramos, degrada a 2 o a 1 (ajuste único, como antes). Devuelve los
    límites de tau (en segundos, inclusive) de cada tramo."""
    mask = (tau_s > 0) & (msd > 0) & np.isfinite(msd)
    tau_s, msd = tau_s[mask], msd[mask]
    order = np.argsort(tau_s)
    tau_s, msd = tau_s[order], msd[order]
    logt, logm = np.log(tau_s), np.log(msd)
    n = len(logt)

    def sse(lo, hi):
        if hi - lo < 2:
            return 0.0
        res = stats.linregress(logt[lo:hi], logm[lo:hi])
        pred = res.intercept + res.slope * logt[lo:hi]
        return float(np.sum((logm[lo:hi] - pred) ** 2))

    for k in range(max_segments, 0, -1):
        if n < k * min_points:
            continue
        if k == 1:
            bounds = [0, n]
        else:
            candidates = range(min_points, n - min_points + 1)
            best = None
            for cuts in itertools.combinations(candidates, k - 1):
                b = [0, *cuts, n]
                if any(b[i + 1] - b[i] < min_points for i in range(k)):
                    continue
                total = sum(sse(b[i], b[i + 1]) for i in range(k))
                if best is None or total < best[0]:
                    best = (total, b)
            if best is None:
                continue
            bounds = best[1]
        # límites en tau_s (segundos), no en índice: así cada película los
        # puede aplicar a su propia grilla de tau aunque no coincida exacto
        # con la de la "gran media".
        segments = []
        for i in range(len(bounds) - 1):
            lo_idx, hi_idx = bounds[i], bounds[i + 1] - 1
            segments.append((float(tau_s[lo_idx]), float(tau_s[hi_idx])))
        return segments
    return [(float(tau_s[0]), float(tau_s[-1]))]


def bootstrap_mean_ci(values, n_boot=N_BOOT, rng=RNG):
    """Arreglo #3 (+#6): bootstrap sobre las 16 PELÍCULAS (no sobre células
    ni sobre puntos del MSD). Como las películas sí son unidades
    independientes, este bootstrap es estadísticamente válido -- a
    diferencia de un bootstrap sobre las miles de células pooled, que
    seguiría subestimando el error porque las células de una misma película
    no son independientes entre sí."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    if n == 0:
        return {"mean": np.nan, "sem_clasico": np.nan, "boot_se": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "n_peliculas": 0}
    if n < 2:
        return {"mean": values[0], "sem_clasico": np.nan, "boot_se": np.nan, "ci95_lo": np.nan, "ci95_hi": np.nan, "n_peliculas": 1}
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return {
        "mean": float(values.mean()),
        "sem_clasico": float(values.std(ddof=1) / np.sqrt(n)),
        "boot_se": float(boot_means.std(ddof=1)),
        "ci95_lo": float(np.percentile(boot_means, 2.5)),
        "ci95_hi": float(np.percentile(boot_means, 97.5)),
        "n_peliculas": n,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    um_per_px, finterval_s = read_calibration()
    (OUT_DIR / "calibracion.txt").write_text(
        f"um_per_px = {um_per_px:.6f}\n"
        f"finterval_s = {finterval_s:.6f}\n"
        f"finterval_min = {finterval_s/60:.6f}\n"
        f"fuente = metadatos ImageJ embebidos en datasets/brightfield-mdamb231/.../0001.tif "
        f"(tag XResolution + ImageDescription:finterval)\n"
    )
    print(f"Calibración: {um_per_px:.4f} um/px, {finterval_s:.2f} s/frame ({finterval_s/60:.2f} min/frame)")

    df = pd.read_csv(TRACKS_CSV)
    if "movie" not in df.columns:
        raise SystemExit("tracks.csv no tiene columna 'movie' -- correr de nuevo 02_tracking.py (versión por película).")
    df = df.sort_values(["movie", "track_id", "frame"]).reset_index(drop=True)
    df["x_um"] = df["x"] * um_per_px
    df["y_um"] = df["y"] * um_per_px
    movies = sorted(df["movie"].unique())
    n_movies = len(movies)

    # ================= 1.1 VALIDACIÓN =================
    track_lengths = df.groupby("track_id").size()
    n_tracks = len(track_lengths)
    frac_completa = (track_lengths == FRAMES_PER_MOVIE).mean()

    gap_links, total_links = 0, 0
    for _, g in df.groupby("track_id"):
        frames = g["frame"].values
        if len(frames) < 2:
            continue
        deltas = np.diff(frames)
        total_links += len(deltas)
        gap_links += int(np.sum(deltas > 1))
    frac_gap = gap_links / total_links if total_links else np.nan

    validacion_md = f"""# 1.1 Validación de segmentación/tracking

**Dataset:** brightfield MDA-MB-231, {n_movies} películas de {FRAMES_PER_MOVIE} frames
cada una, {n_tracks} trayectorias (>=3 frames) en total.

- **Error de localización (ε):** no aplica — el dataset no tiene objetos
  inmóviles/fijos de referencia (todas las detecciones son células en
  movimiento activo). No se calculó.
- **Comparación contra ground truth (FP, FN, ID switches, fragmentación):**
  no disponible — el dataset LFCT, que traía anotación manual de
  tracking, no se pudo descargar (los enlaces de Zenodo y del sitio del
  proyecto devolvían 404 al momento de intentar la descarga). No hay
  ninguna otra fuente de ground truth en el proyecto. No se calculó.
- **Tasa de error de asociación entre cuadros consecutivos:** no se puede
  calcular una tasa de error *validada* sin ground truth. Como proxy de
  autoconsistencia se reporta (pooled sobre las {n_movies} películas):
  - Fracción de trayectorias que abarcan la película completa
    ({FRAMES_PER_MOVIE} frames): **{frac_completa*100:.1f}%** ({int((track_lengths==FRAMES_PER_MOVIE).sum())}/{n_tracks})
  - Fracción de eslabones frame-a-frame que requirieron *gap closing*:
    **{frac_gap*100:.2f}%** ({gap_links}/{total_links} eslabones)

  Esto NO es una tasa de error de asociación validada, solo un indicador de
  cuánto tuvo que apoyarse el tracking en el cierre de huecos. Los tau/lag
  usados en el resto de este análisis usan la diferencia REAL de frame
  (no el índice de fila), así que estos huecos ya no desalinean las curvas
  de MSD/VACF/ángulos (ver docstring del script, arreglo #4).
"""
    (OUT_DIR / "validacion.md").write_text(validacion_md)
    print("1.1 validación -> validacion.md")

    # ================= 1.2 TA-MSD por célula (gap-fijado) =================
    ta_rows, per_cell_ta = [], {}
    for track_id, g in df.groupby("track_id"):
        frames = g["frame"].values
        xy = g[["x_um", "y_um"]].values
        res = compute_ta_msd_single(frames, xy, min_len=8)
        if res is None:
            continue
        taus, msds = res
        per_cell_ta[track_id] = (taus, msds)
        for tau, msd in zip(taus, msds):
            ta_rows.append({"track_id": track_id, "tau_frames": tau, "tau_s": tau * finterval_s, "msd_um2": msd})
    ta_df = pd.DataFrame(ta_rows)
    ta_df.to_csv(OUT_DIR / "msd_ta.csv", index=False)
    print(f"1.2 TA-MSD -> msd_ta.csv ({n_tracks} tracks totales, {len(per_cell_ta)} con TA-MSD válido)")

    # ================= 1.2/1.3 EA-MSD por película + régimen (arreglos #1,#2,#4,#6) =================
    ea_por_pelicula = []
    ea_curvas = {}  # movie -> (tau_s array, ea_msd array) para el ajuste por tramos
    for movie in movies:
        movie_df = df[df["movie"] == movie]
        ea_df_m, n0 = compute_ea_msd_for_movie(movie_df)
        if len(ea_df_m):
            ea_df_m["movie"] = movie
            ea_df_m["tau_s"] = ea_df_m["tau_frames"] * finterval_s
            ea_df_m["n_celulas_tau1"] = n0
            ea_por_pelicula.append(ea_df_m)
            ea_curvas[movie] = (ea_df_m["tau_s"].values, ea_df_m["ea_msd_um2"].values)
    ea_pel_df = pd.concat(ea_por_pelicula, ignore_index=True)
    ea_pel_df.to_csv(OUT_DIR / "msd_poblacion_por_pelicula.csv", index=False)
    print(f"1.2 EA-MSD por película -> msd_poblacion_por_pelicula.csv ({n_movies} películas)")

    # gran media entre películas, tau por tau (solo tau con >= MIN_MOVIES_PER_TAU películas)
    grand_rows = []
    max_tau_frames = int(ea_pel_df["tau_frames"].max())
    for tau in range(1, max_tau_frames + 1):
        vals = ea_pel_df.loc[ea_pel_df["tau_frames"] == tau, "ea_msd_um2"].values
        if len(vals) >= MIN_MOVIES_PER_TAU:
            grand_rows.append({
                "tau_frames": tau, "tau_s": tau * finterval_s,
                "ea_msd_mean_um2": float(vals.mean()),
                "ea_msd_sem_um2": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                "n_peliculas": len(vals),
            })
    msd_poblacion_df = pd.DataFrame(grand_rows)
    msd_poblacion_df.to_csv(OUT_DIR / "msd_poblacion.csv", index=False)
    print(f"1.2 gran media EA-MSD entre películas -> msd_poblacion.csv ({len(msd_poblacion_df)} valores de tau)")

    # arreglo #1: tramos de régimen detectados sobre la gran media
    segments = find_regime_breakpoints(msd_poblacion_df["tau_s"].values, msd_poblacion_df["ea_msd_mean_um2"].values)
    print(f"1.3 régimen: {len(segments)} tramo(s) detectados (tau_s): " + ", ".join(f"[{lo:.0f}-{hi:.0f}]" for lo, hi in segments))

    # arreglo #1+#6: alpha por PELÍCULA y por TRAMO (no un único fit pooled)
    alpha_rows = []
    for movie in movies:
        tau_s_m, msd_m = ea_curvas.get(movie, (np.array([]), np.array([])))
        for seg_i, (lo, hi) in enumerate(segments, start=1):
            mask = (tau_s_m >= lo) & (tau_s_m <= hi)
            fit = loglog_fit(tau_s_m[mask], msd_m[mask]) if mask.sum() >= 3 else None
            alpha_rows.append({
                "movie": movie, "segmento": seg_i, "tau_min_s": lo, "tau_max_s": hi,
                "alpha": fit["alpha"] if fit else np.nan,
                "K_alpha_um2_s-alpha": fit["K_alpha"] if fit else np.nan,
                "r2": fit["r2"] if fit else np.nan,
                "n_puntos": fit["n_puntos"] if fit else int(mask.sum()),
            })
    alpha_pel_df = pd.DataFrame(alpha_rows)
    alpha_pel_df.to_csv(OUT_DIR / "alpha_por_pelicula_y_segmento.csv", index=False)
    print(f"1.3 alpha por película y tramo -> alpha_por_pelicula_y_segmento.csv")

    # arreglo #3+#6: PRW por PELÍCULA (ajuste sobre toda la curva truncada por supervivencia)
    #
    # Degeneración conocida del modelo PRW: MSD(tau) = A*(tau/tau_p - 1 + exp(-tau/tau_p))
    # tiende a A*tau/tau_p*... -> una recta pura (difusión simple) cuando tau_p -> 0,
    # manteniendo D=A/(4*tau_p) finito. Si una película no muestra curvatura de
    # persistencia detectable dentro del rango de tau disponible (dinámica cercana
    # a difusiva desde el principio), curve_fit puede "resolver" eso empujando A y
    # tau_p los dos hacia el piso de sus bounds (A~1e-5, tau_p~1e-4 s) en vez de
    # converger a un tau_p realista -- con buen R² igual, porque en ese régimen el
    # modelo YA ES casi una recta. Un tau_p por debajo de un frame (finterval_s) no
    # es un tiempo de persistencia medible con este muestreo: se marca inválido acá
    # y se excluye de tau_p_s/A_um2/D_um2_s en resumen_por_pelicula.csv (r2 y
    # n_puntos quedan igual en prw_por_pelicula.csv para que se pueda auditar).
    prw_rows = []
    n_degenerados = 0
    for movie in movies:
        tau_s_m, msd_m = ea_curvas.get(movie, (np.array([]), np.array([])))
        fit = fit_prw(tau_s_m, msd_m)
        valido = bool(fit) and fit["tau_p"] >= finterval_s
        if fit and not valido:
            n_degenerados += 1
        prw_rows.append({
            "movie": movie,
            "A_um2": fit["A"] if valido else np.nan,
            "tau_p_s": fit["tau_p"] if valido else np.nan,
            "D_um2_s": fit["D"] if valido else np.nan,
            "r2": fit["r2"] if fit else np.nan,
            "n_puntos": fit["n_puntos"] if fit else 0,
            "valido": valido,
        })
    prw_pel_df = pd.DataFrame(prw_rows)
    prw_pel_df.to_csv(OUT_DIR / "prw_por_pelicula.csv", index=False)
    print(f"1.3 PRW por película -> prw_por_pelicula.csv ({n_degenerados} película(s) con ajuste degenerado, tau_p<{finterval_s:.0f}s, excluidas del resumen)")

    # ================= 1.4 VACF, rapidez, ángulos (por película, gap-fijado) =================
    vacf_por_pelicula, speed_rows_all, turning_angles_all = [], [], []
    max_lag = FRAMES_PER_MOVIE - 1
    for movie in movies:
        movie_df = df[df["movie"] == movie]
        vsum, vcount, srows, tangles = compute_vacf_speed_angles_for_movie(movie_df, finterval_s, max_lag)
        for lag in range(max_lag + 1):
            if vcount[lag] >= MIN_CELLS_PER_TAU_MOVIE:
                vacf_por_pelicula.append({
                    "movie": movie, "lag_frames": lag, "lag_s": lag * finterval_s,
                    "vacf_um2_s2": vsum[lag] / vcount[lag], "n_pares": int(vcount[lag]),
                })
        for r in srows:
            r["movie"] = movie
        speed_rows_all.extend(srows)
        turning_angles_all.extend(tangles)

    vacf_pel_df = pd.DataFrame(vacf_por_pelicula)
    vacf_pel_df.to_csv(OUT_DIR / "vacf_por_pelicula.csv", index=False)
    print(f"1.4 VACF por película -> vacf_por_pelicula.csv")

    grand_vacf_rows = []
    for lag in range(max_lag + 1):
        vals = vacf_pel_df.loc[vacf_pel_df["lag_frames"] == lag, "vacf_um2_s2"].values
        if len(vals) >= MIN_MOVIES_PER_TAU:
            grand_vacf_rows.append({
                "lag_frames": lag, "lag_s": lag * finterval_s,
                "vacf_mean_um2_s2": float(vals.mean()),
                "vacf_sem_um2_s2": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan,
                "n_peliculas": len(vals),
            })
    vacf_df = pd.DataFrame(grand_vacf_rows)
    if len(vacf_df):
        vacf_df["vacf_norm"] = vacf_df["vacf_mean_um2_s2"] / vacf_df["vacf_mean_um2_s2"].iloc[0]
    vacf_df.to_csv(OUT_DIR / "vacf.csv", index=False)
    print(f"1.4 gran media VACF entre películas -> vacf.csv ({len(vacf_df)} lags)")

    angulos_df = pd.DataFrame({"angulo_giro_deg": turning_angles_all})
    angulos_df.to_csv(OUT_DIR / "angulos_giro.csv", index=False)
    print(f"1.4 ángulos de giro -> angulos_giro.csv ({len(angulos_df)} ángulos, gap-fijado)")

    speed_df = pd.DataFrame(speed_rows_all)
    speed_df.to_csv(OUT_DIR / "rapidez_instantanea.csv", index=False)
    print(f"1.4 rapidez instantánea -> rapidez_instantanea.csv ({len(speed_df)} pasos, gap-fijado)")

    # ================= 1.5 Morfología =================
    morfo_rows = []
    mask_files = {p.stem.replace("_masks", ""): p for p in MASKS_DIR.glob("*_masks.npy")}
    masks_cache = {}
    for (frame, mask_label, movie), sub in df.groupby(["frame", "mask_label", "movie"]):
        fkey = f"{int(sub['file'].iloc[0]):04d}"
        if fkey not in mask_files:
            continue
        if fkey not in masks_cache:
            masks_cache[fkey] = np.load(mask_files[fkey])
        masks = masks_cache[fkey]
        region = masks == mask_label
        if not region.any():
            continue
        props = regionprops_table(region.astype(np.uint8), properties=("area", "perimeter", "major_axis_length", "minor_axis_length"))
        if props["minor_axis_length"][0] == 0:
            continue
        area_um2 = props["area"][0] * (um_per_px ** 2)
        perim_um = props["perimeter"][0] * um_per_px
        aspect_ratio = props["major_axis_length"][0] / props["minor_axis_length"][0]
        irregularidad = (perim_um ** 2) / (4 * np.pi * area_um2) if area_um2 > 0 else np.nan
        morfo_rows.append({
            "track_id": sub["track_id"].iloc[0], "movie": movie, "frame": frame, "area_um2": area_um2,
            "aspect_ratio": aspect_ratio, "irregularidad": irregularidad,
        })
    morfo_df = pd.DataFrame(morfo_rows)
    morfo_df.to_csv(OUT_DIR / "morfologia_por_frame.csv", index=False)
    print(f"1.5 morfología -> morfologia_por_frame.csv ({len(morfo_df)} filas)")

    # ================= por_celula.csv (con columna movie, para cajas/violines por película) =================
    cell_rows = []
    for track_id, g in df.groupby("track_id"):
        g = g.sort_values("frame")
        n = len(g)
        row = {"track_id": track_id, "movie": g["movie"].iloc[0], "n_frames": n}

        speeds_i = speed_df[speed_df.track_id == track_id]["rapidez_um_s"] if len(speed_df) else pd.Series(dtype=float)
        row["rapidez_media_um_s"] = speeds_i.mean() if len(speeds_i) else np.nan

        disp_neto = np.linalg.norm(g[["x_um", "y_um"]].iloc[-1].values - g[["x_um", "y_um"]].iloc[0].values)
        pasos = np.diff(g[["x_um", "y_um"]].values, axis=0)
        long_total = np.sum(np.linalg.norm(pasos, axis=1))
        row["indice_direccionalidad"] = disp_neto / long_total if long_total > 0 else np.nan

        morfo_i = morfo_df[morfo_df.track_id == track_id]
        row["area_um2"] = morfo_i["area_um2"].mean() if len(morfo_i) else np.nan
        row["aspect_ratio"] = morfo_i["aspect_ratio"].mean() if len(morfo_i) else np.nan
        row["irregularidad"] = morfo_i["irregularidad"].mean() if len(morfo_i) else np.nan

        if track_id in per_cell_ta and n >= MIN_LEN_ALPHA_FIT:
            taus, msds = per_cell_ta[track_id]
            fit = loglog_fit(taus * finterval_s, msds)
            if fit:
                row["alpha"] = fit["alpha"]
                row["K_alpha"] = fit["K_alpha"]
                row["r2_alpha"] = fit["r2"]
        if track_id in per_cell_ta and n >= MIN_LEN_PRW_FIT:
            taus, msds = per_cell_ta[track_id]
            fit = fit_prw(taus * finterval_s, msds)
            if fit:
                row["tau_p_s"] = fit["tau_p"]
                row["D_um2_s"] = fit["D"]
                row["A_um2"] = fit["A"]
                row["r2_prw"] = fit["r2"]
        cell_rows.append(row)

    por_celula_df = pd.DataFrame(cell_rows)
    por_celula_df.to_csv(OUT_DIR / "por_celula.csv", index=False)
    print(f"por_celula.csv -> {len(por_celula_df)} células")

    # ================= resumen_por_pelicula.csv: EL dataset real (n=16) =================
    agg = por_celula_df.groupby("movie").agg(
        n_celulas=("track_id", "count"),
        rapidez_media_um_s=("rapidez_media_um_s", "mean"),
        indice_direccionalidad=("indice_direccionalidad", "mean"),
        area_um2=("area_um2", "mean"),
        aspect_ratio=("aspect_ratio", "mean"),
        irregularidad=("irregularidad", "mean"),
    ).reset_index()

    alpha_wide = alpha_pel_df.pivot(index="movie", columns="segmento", values="alpha")
    alpha_wide.columns = [f"alpha_tramo{c}" for c in alpha_wide.columns]
    alpha_wide = alpha_wide.reset_index()

    resumen_pel_df = agg.merge(alpha_wide, on="movie", how="left").merge(
        prw_pel_df[["movie", "tau_p_s", "D_um2_s", "A_um2"]], on="movie", how="left"
    )
    resumen_pel_df.to_csv(OUT_DIR / "resumen_por_pelicula.csv", index=False)
    print(f"resumen_por_pelicula.csv -> {len(resumen_pel_df)} películas (EL dataset real para inferencia poblacional)")

    # ================= resumen_bootstrap.csv: números "oficiales" (bootstrap entre películas) =================
    boot_rows = []
    for col in resumen_pel_df.columns:
        if col == "movie":
            continue
        stats_boot = bootstrap_mean_ci(resumen_pel_df[col].values)
        boot_rows.append({"metrica": col, **stats_boot})
    resumen_boot_df = pd.DataFrame(boot_rows)
    resumen_boot_df.to_csv(OUT_DIR / "resumen_bootstrap.csv", index=False)
    print("resumen_bootstrap.csv -> media/SEM/IC95% bootstrap (n=16 películas) por métrica")
    print(resumen_boot_df.to_string(index=False))

    # ================= resumen.md =================
    def fmt_boot(metrica):
        r = resumen_boot_df[resumen_boot_df.metrica == metrica]
        if not len(r) or pd.isna(r.iloc[0]["mean"]):
            return "sin datos suficientes"
        r = r.iloc[0]
        return f"{r['mean']:.4g} (IC95% bootstrap [{r['ci95_lo']:.4g}, {r['ci95_hi']:.4g}], n={int(r['n_peliculas'])} películas)"

    resumen = f"""# Resumen de estadísticos de dinámica de movimiento

**Dataset usado:** brightfield MDA-MB-231 (Zenodo 10.5281/zenodo.10074471,
`Training-source-BF.zip`), {n_movies} películas de {FRAMES_PER_MOVIE} frames,
intervalo real {finterval_s:.1f} s (~{finterval_s/60:.1f} min), calibración
{um_per_px:.4f} um/px. Es la ÚNICA condición disponible con tracking completo:
no hay comparación entre condiciones experimentales, pero sí {n_movies} campos
(réplicas biológicas independientes) de la misma condición -- por eso TODA la
estadística poblacional de abajo se calcula por película y se agrega entre
películas (n={n_movies}), no pooleando las {n_tracks} células como si fueran
independientes entre sí (ver docstring del script para el detalle de los 6
arreglos aplicados sobre la versión anterior de este análisis).

## 1.1 Validación
Ver `resultados/estadisticas/validacion.md`. Proxy de autoconsistencia:
{frac_completa*100:.1f}% de tracks completos (100 frames), {frac_gap*100:.2f}%
de eslabones vía gap-closing. Sin ground truth disponible (LFCT no se pudo
descargar) no hay una tasa de error de asociación validada.

## 1.2 MSD
- TA-MSD calculado para {len(per_cell_ta)}/{n_tracks} células (gap-fijado: usa
  diferencia real de frame, no índice de fila).
- EA-MSD calculado POR PELÍCULA, truncado en el primer τ donde la cohorte cae
  por debajo del {SURVIVORSHIP_MIN_FRAC*100:.0f}% de su tamaño en τ=1 (arreglo
  del sesgo de supervivencia). La "gran media" en `msd_poblacion.csv` es el
  promedio de esas {n_movies} curvas independientes.

## 1.3 Régimen de migración (α, por tramos detectados automáticamente)
Tramos detectados sobre la gran media (τ en s): {", ".join(f"[{lo:.0f}, {hi:.0f}]" for lo, hi in segments)}.
Para cada tramo, α se ajustó independientemente EN CADA PELÍCULA y se agregó
con bootstrap entre películas (n={n_movies}):
"""
    for seg_i in range(1, len(segments) + 1):
        resumen += f"- **Tramo {seg_i}:** α = {fmt_boot(f'alpha_tramo{seg_i}')}\n"
    resumen += f"""
**Persistent Random Walk (PRW)**, ajustado por película sobre la curva
completa (truncada por supervivencia) y agregado con bootstrap entre
películas:
- τ_p (tiempo de persistencia) = {fmt_boot('tau_p_s')} s
- D (coeficiente de difusión efectivo, D=A/4τ_p) = {fmt_boot('D_um2_s')} um²/s
- A (amplitud) = {fmt_boot('A_um2')} um²

## 1.4 Dinámica complementaria
- Rapidez instantánea: n={len(speed_df)} pasos (gap-fijado), media pooled =
  {speed_df['rapidez_um_s'].mean():.4f} um/s. Rapidez media POR PELÍCULA
  (n={n_movies}) = {fmt_boot('rapidez_media_um_s')} um/s.
- Índice de direccionalidad por película (n={n_movies}) = {fmt_boot('indice_direccionalidad')}.
- VACF y ángulos de giro (`vacf.csv`, `angulos_giro.csv`): gap-fijados, VACF
  agregada entre películas igual que el MSD.

## 1.5 Morfología (por película, n={n_movies})
- Área = {fmt_boot('area_um2')} um²
- Aspect ratio = {fmt_boot('aspect_ratio')}
- Irregularidad = {fmt_boot('irregularidad')}

## Archivos clave para citar números
`resultados/estadisticas/resumen_por_pelicula.csv` (las {n_movies} películas,
una fila cada una — el dataset real) y `resumen_bootstrap.csv` (media, SEM
clásico, SE y IC95% por bootstrap entre películas, para cada métrica de
arriba). Los ajustes individuales por célula (para las cajas/violines,
n={len(por_celula_df)} células) siguen en `por_celula.csv`, ahora con columna
`movie` para agrupar correctamente en las figuras.
"""
    (ROOT / "resultados" / "resumen.md").write_text(resumen)
    print("\nresumen.md escrito en resultados/")


if __name__ == "__main__":
    main()
