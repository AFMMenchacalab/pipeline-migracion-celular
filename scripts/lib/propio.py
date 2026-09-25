"""
Tracking y estadística de un experimento del microscopio propio, a partir
de las máscaras que deja la segmentación en vivo (lib/en_vivo.py).

Usa las mismas piezas y parámetros que el pipeline validado en campo claro
(variante `dist_tam` de 14_tracking.py: enlace por distancia con
penalización de cambio de tamaño, cierre de huecos de hasta 3 ciclos), y
las mismas medidas del reporte: rapidez, direccionalidad, α de cada célula,
EAD₁ corregida, MSD del ensamble con ajuste de caminata persistente.

El tiempo se toma de la fecha de cada ciclo (img_<AAAAMMDD_HHMMSS>), no se
supone: si un ciclo falta, queda un hueco en vez de juntar dos tiempos.

Salida en ~/microscopio_cache/analisis/propio/<experimento>/<cam>/:
  tracks.csv, por_celula.csv, resumen.json, trayectorias.png, msd.png
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS
from . import entropia as E
from . import motilidad as M

PROPS = ("label", "centroid", "area", "bbox", "eccentricity", "solidity",
         "major_axis_length", "minor_axis_length", "perimeter")
AREA_MIN_UM2 = 50.0
MIN_PUNTOS = 6              # trayectorias más cortas no entran en la estadística


def serie(salida, exp, cam):
    """[(fecha, datetime, ruta npz)] ordenado por tiempo."""
    d = Path(salida) / exp / cam
    filas = []
    for p in sorted(d.glob("*.npz")):
        try:
            filas.append((p.stem, datetime.strptime(p.stem, "%Y%m%d_%H%M%S"), p))
        except ValueError:
            continue
    return filas


def detecciones(filas, area_min_um2=AREA_MIN_UM2):
    from skimage.measure import regionprops_table
    t0 = filas[0][1]
    ts = np.array([(f[1] - t0).total_seconds() for f in filas])
    dt_s = float(np.median(np.diff(ts))) if len(ts) > 1 else 60.0
    um = None
    partes = []
    for fecha, cuando, ruta in filas:
        z = np.load(ruta)
        m = z["masks"]
        um = float(z["um_por_px"]) if "um_por_px" in z.files else um
        if m.max() == 0:
            continue
        d = pd.DataFrame(regionprops_table(m, properties=PROPS)).rename(
            columns={"centroid-0": "y", "centroid-1": "x"})
        H, W = m.shape
        d["borde"] = (d["bbox-0"] == 0) | (d["bbox-1"] == 0) | (d["bbox-2"] == H) | (d["bbox-3"] == W)
        d = d.drop(columns=[c for c in d.columns if c.startswith("bbox")])
        d["t_s"] = (cuando - t0).total_seconds()
        d["frame"] = int(round(d["t_s"].iloc[0] / dt_s))    # hueco si falta un ciclo
        d["fecha"] = fecha
        partes.append(d)
    det = pd.concat(partes, ignore_index=True)
    det["movie"] = 1
    det = det[det["area"] * um ** 2 >= area_min_um2].reset_index(drop=True)
    return det, um, dt_s, m.shape


def analizar(salida, exp, cam, destino, entrada_dir=None, progreso=None):
    """Tracking + estadística de una cámara de un experimento."""
    from .tracking import trackear
    aviso = progreso or (lambda *_: None)
    filas = serie(salida, exp, cam)
    if len(filas) < MIN_PUNTOS:
        raise ValueError(f"hacen falta al menos {MIN_PUNTOS} ciclos segmentados "
                         f"(hay {len(filas)})")
    aviso("detectando células")
    det, um, dt_s, forma = detecciones(filas)
    dt_min = dt_s / 60
    # registrar el experimento como un dataset más, para reusar lib/tracking
    DATASETS["propio"] = {"um_per_px": um, "dt_s": dt_s, "area_min_um2": AREA_MIN_UM2}
    link = max(6.0, 25.0 * dt_min / 5.0)       # misma regla que 14_tracking.py
    aviso("uniendo trayectorias")
    tr, _ = trackear(det, "propio", max_link_um=link, gap_frames=3, gap_link_um=link * 1.5,
                     size_weight=1.0, min_len=3)
    tr["track_id"] -= 1_000_000                 # trackear() suma película × 10⁶; aquí hay una sola
    tr["tree_id"] -= 1_000_000
    tr["x_um"], tr["y_um"] = tr["x"] * um, tr["y"] * um
    tr["area_um2"] = tr["area"] * um ** 2
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    tr.to_csv(destino / "tracks.csv", index=False)

    aviso("calculando estadística")
    celulas, segs = [], []
    for tid, g in tr.groupby("track_id"):
        g = g.sort_values("frame")
        if len(g) < MIN_PUNTOS:
            continue
        xy = g[["x_um", "y_um"]].to_numpy()
        t = g["t_s"].to_numpy() / 60
        pasos = np.hypot(*np.diff(xy, axis=0).T)
        recorrido = pasos.sum()
        neto = float(np.hypot(*(xy[-1] - xy[0])))
        alfa, _ = M.alfa_celula(xy, dt_min)
        v = np.diff(xy, axis=0)
        ead1 = np.nan
        if len(v) >= 9:
            _, co, _ = E.ead_perfil(v, 1, 12)
            ead1 = co[0]
        celulas.append({"track_id": int(tid), "inicio_min": round(t[0], 1),
                        "duracion_min": round(t[-1] - t[0], 1), "puntos": len(g),
                        "rapidez_um_min": recorrido / max(t[-1] - t[0], 1e-9),
                        "direccionalidad": neto / max(recorrido, 1e-9),
                        "desplazamiento_neto_um": neto, "alfa": alfa, "ead1_corr": ead1,
                        "area_um2": g["area_um2"].mean(),
                        "alargamiento": (g["major_axis_length"] / g["minor_axis_length"].replace(0, np.nan)).mean()})
        segs.append(xy)
    pc = pd.DataFrame(celulas)
    pc.to_csv(destino / "por_celula.csv", index=False)

    resumen = {"experimento": exp, "camara": cam, "ciclos": len(filas),
               "intervalo_min": round(dt_min, 2), "um_por_px": um,
               "detecciones": int(len(det)), "trayectorias": int(tr["track_id"].nunique()),
               "trayectorias_analizadas": int(len(pc)),
               "celulas_por_ciclo": round(len(det) / len(filas), 1)}
    if len(pc):
        resumen.update({
            "rapidez_um_min": float(pc["rapidez_um_min"].median()),
            "direccionalidad": float(pc["direccionalidad"].median()),
            "alfa_mediana": float(pc["alfa"].median()),
            "ead1_mediana": float(pc["ead1_corr"].median()),
        })
        max_lag = max(4, min(60, int(np.percentile([len(s) for s in segs], 90)) // 2))
        msd, _, _ = M.tea_msd(segs, max_lag, min_pares=20)
        tau = np.arange(1, max_lag + 1) * dt_min
        prw = M.ajustar_prw(tau, msd)
        resumen["prw"] = {k: (None if not np.isfinite(v) else float(v)) for k, v in prw.items()}
        ok = np.isfinite(msd) & (msd > 0)
        n = min(6, ok.sum())
        if n >= 3:
            resumen["alfa_ensamble"] = float(np.polyfit(np.log(tau[ok][:n]), np.log(msd[ok][:n]), 1)[0])
        figura_msd(tau, msd, prw, destino / "msd.png")
    figura_trayectorias(tr, forma, filas[-1][0], entrada_dir, exp, cam, destino / "trayectorias.png")
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    aviso("listo")
    return resumen


def _fondo(entrada_dir, exp, cam, fecha, forma):
    """Imagen de fondo para dibujar las trayectorias: la suma de las fotos
    del último ciclo, al tamaño de las máscaras. Si no está, fondo gris."""
    import cv2
    import tifffile
    if entrada_dir:
        fotos = sorted((Path(entrada_dir) / exp / cam).glob(f"img_{fecha}*.tif"))
        if fotos:
            img = np.mean([tifffile.imread(f).astype(np.float32) for f in fotos], axis=0)
            if img.ndim == 3:
                img = img.mean(-1)
            img = cv2.resize(img, (forma[1], forma[0]), interpolation=cv2.INTER_AREA)
            lo, hi = np.percentile(img, (1, 99))
            return (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    return np.full(forma, 40, np.uint8)


def figura_trayectorias(tr, forma, fecha, entrada_dir, exp, cam, ruta, lado_max=1200):
    import cv2
    g8 = _fondo(entrada_dir, exp, cam, fecha, forma)
    rgb = np.dstack([g8] * 3)
    esc = min(1.0, lado_max / max(forma))
    if esc < 1:
        rgb = cv2.resize(rgb, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    # paleta categórica fija (mismo orden que las figuras del reporte), en BGR
    paleta = [(214, 120, 40), (58, 103, 230), (100, 160, 40), (180, 90, 200),
              (40, 190, 230), (200, 200, 60), (90, 90, 220), (160, 160, 160)]
    grosor = max(1, int(round(2 * esc / 0.6)))
    for i, (tid, g) in enumerate(tr.groupby("track_id")):
        g = g.sort_values("frame")
        pts = np.round(g[["x", "y"]].to_numpy() * esc).astype(np.int32)
        if len(pts) < 2:
            continue
        c = paleta[i % len(paleta)]
        cv2.polylines(rgb, [pts.reshape(-1, 1, 2)], False, c, grosor, cv2.LINE_AA)
        cv2.circle(rgb, tuple(int(v) for v in pts[-1]), grosor + 2, c, -1, cv2.LINE_AA)
    tmp = Path(ruta).with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), rgb)
    tmp.replace(ruta)


def figura_msd(tau, msd, prw, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=150)
    ok = np.isfinite(msd) & (msd > 0)
    ax.loglog(tau[ok], msd[ok], "o", ms=4, color="#2563eb", label="MSD medido")
    if np.isfinite(prw.get("D", np.nan)):
        t = np.logspace(np.log10(tau[ok][0]), np.log10(tau[ok][-1]), 100)
        ax.loglog(t, M.prw_msd(t, prw["D"], prw["P"], prw["sigma"] ** 2), color="#ea580c", lw=2,
                  label=f"caminata persistente (P = {prw['P']:.1f} min)")
    x0, y0 = tau[ok][0], msd[ok][0]
    for a, estilo in ((1, ":"), (2, "--")):
        ax.loglog(tau[ok], y0 * (tau[ok] / x0) ** a, estilo, color="#94a3b8", lw=1)
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, FuncFormatter
    # etiquetas en números simples (no 3×10¹), solo en valores redondos dentro del rango
    marcas = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000)
              if tau[ok][0] * 0.9 <= v <= tau[ok][-1] * 1.1]
    ax.xaxis.set_major_locator(FixedLocator(marcas))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("τ (min)")
    ax.set_ylabel("MSD (µm²)")
    ax.set_title("Desplazamiento cuadrático medio\n(punteada: α = 1; guiones: α = 2)", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    tmp = Path(ruta).with_suffix(".tmp.png")
    fig.savefig(tmp)
    plt.close(fig)
    tmp.replace(ruta)
