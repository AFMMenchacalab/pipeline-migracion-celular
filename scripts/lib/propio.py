"""
Tracking y estadística de un experimento del microscopio propio, a partir
de las máscaras que deja la segmentación en vivo (lib/en_vivo.py).

Usa las mismas piezas y parámetros que el pipeline validado en campo claro
(variante `dist_tam` de 14_tracking.py: enlace por distancia con
penalización de cambio de tamaño, cierre de huecos de hasta 3 ciclos), y
las mismas medidas del reporte: rapidez, direccionalidad, α de cada célula,
EAD₁ corregida, MSD del ensamble con ajuste de caminata persistente.

El tiempo se toma de la fecha de cada ciclo (img_<AAAAMMDD_HHMMSS> o, con
los nombres de MicroscopeOS desde el 2026-10-02, 0001_<AAAA-MM-DD_HH-MM-SS>),
no se supone: si un ciclo falta, queda un hueco en vez de juntar dos tiempos.

Cúmulos: cada detección lleva `grupo`, cuántas células tiene el cúmulo en
el que está (1 = suelta). Se cuentan juntas las células que se tocan y
también la zona con células que se ve en el DPC aunque Cellpose no la haya
separado (una lámina sin segmentar no deja "suelta" a la célula de su
borde); el tamaño es el mayor entre las máscaras que caen en el cúmulo y
su área dividida por el área típica de una célula. Con `max_grupo` el
tracking se hace igual con todas las células (para no enlazar mal cuando
una entra o sale de un cúmulo) y la estadística usa solo los tramos de
cada trayectoria en los que la célula está en un grupo de hasta
`max_grupo` células.

Salida en ~/microscopio_cache/analisis/propio/<experimento>/<cam>/ (o en
<cam>/hasta_<N>/ con max_grupo):
  tracks.csv, por_celula.csv, resumen.json, trayectorias.png, msd.png
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASETS
from . import entropia as E
from . import motilidad as M
from .importar import fecha_de

PROPS = ("label", "centroid", "area", "bbox", "eccentricity", "solidity",
         "major_axis_length", "minor_axis_length", "perimeter")
AREA_MIN_UM2 = 50.0
# Restos: trayectorias cuya área mediana es menor que esto no son células
# (polvo y basura del medio: manchas compactas y casi inmóviles de 80–150 µm²
# en prueba 7). La mediana de toda la trayectoria, no cada detección, para
# no perder células que se redondean un rato al dividirse.
AREA_MIN_CELULA_UM2 = 150.0
MIN_PUNTOS = 6              # trayectorias más cortas no entran en la estadística
# Zona con células en el DPC: textura (magnitud suavizada) sobre el fondo
ZONA_SIGMA_UM = 2.5
ZONA_MADS = 4
ZONA_AREA_MIN_UM2 = 80
CONTACTO_UM = 1.5           # dos máscaras a menos de esto se tocan


def zona_celulas(lr, tb, um):
    """Máscara de las zonas con células según la textura del DPC."""
    from scipy import ndimage as ndi
    from skimage import morphology
    t = ndi.gaussian_filter(np.hypot(lr, tb), ZONA_SIGMA_UM / um)
    med = np.median(t)
    mad = np.median(np.abs(t - med)) * 1.4826
    z = t > med + ZONA_MADS * mad
    z = ndi.binary_closing(z, morphology.disk(max(1, int(round(2 / um)))))
    z = ndi.binary_fill_holes(z)
    return morphology.remove_small_objects(z, max_size=int(ZONA_AREA_MIN_UM2 / um ** 2))


def grupos_ciclo(masks, um, area_celula_um2, zona=None):
    """{label: tamaño del cúmulo en células} de un ciclo."""
    from scipy import ndimage as ndi
    union = ndi.binary_dilation(masks > 0, iterations=max(1, int(round(CONTACTO_UM / um))))
    if zona is not None:
        union |= zona
    comp, nc = ndi.label(union)
    lab, cc = masks.ravel(), comp.ravel()
    sel = lab > 0
    pares = np.unique(np.stack([cc[sel], lab[sel]]), axis=1)
    n_masks = np.bincount(pares[0], minlength=nc + 1)
    area = np.bincount(cc, minlength=nc + 1) * um ** 2
    tam = np.maximum(n_masks, np.round(area / area_celula_um2))
    salida = {}
    for c, l in pares.T:
        salida[int(l)] = max(salida.get(int(l), 0), int(tam[c]))
    return salida


def grupos_experimento(filas, exp, cam, entrada_dir, area_celula_um2, progreso=None):
    """DataFrame (fecha, label, grupo) de todos los ciclos. Si están los DPC
    de la Pi (_dpcLR/_dpcTB) se usa también la zona con células."""
    from . import en_vivo as EV
    fotos = {}
    if entrada_dir is not None:
        fotos = {f: fo for e, c, f, fo, su, ok in EV.ciclos(entrada_dir) if e == exp and c == cam}
    partes = []
    for i, (fecha, _, ruta) in enumerate(filas):
        z = np.load(ruta)
        m = z["masks"]
        um = float(z["um_por_px"])
        red = int(z["reducir"]) if "reducir" in z.files else 1
        zona = None
        fo = fotos.get(fecha, {})
        if "_dpcLR" in fo and "_dpcTB" in fo:
            lr, tb = EV.leer(fo["_dpcLR"], red), EV.leer(fo["_dpcTB"], red)
            if lr.shape == m.shape:
                zona = zona_celulas(lr, tb, um)
        g = grupos_ciclo(m, um, area_celula_um2, zona)
        partes.append(pd.DataFrame({"fecha": fecha, "label": list(g), "grupo": list(g.values())}))
        if progreso and i % 50 == 0:
            progreso(f"midiendo cúmulos ({i}/{len(filas)})")
    return pd.concat(partes, ignore_index=True)


def tramos(tr, max_grupo):
    """Corta cada trayectoria en los tramos seguidos en los que la célula
    está en un grupo de hasta max_grupo; cada tramo es una trayectoria nueva
    (track_id original × 1000 + número de tramo, columna track_original)."""
    tr = tr.sort_values(["track_id", "frame"]).copy()
    ok = tr["grupo"] <= max_grupo
    corte = (ok != ok.groupby(tr["track_id"]).shift()).astype(int)
    n = corte.groupby(tr["track_id"]).cumsum()
    tr["track_original"] = tr["track_id"]
    tr["track_id"] = tr["track_id"] * 1000 + n
    return tr[ok]


def serie(salida, exp, cam):
    """[(fecha, datetime, ruta npz)] ordenado por tiempo."""
    d = Path(salida) / exp / cam
    filas = []
    for p in d.glob("*.npz"):
        cuando = fecha_de(p.stem)
        if cuando is not None:
            filas.append((p.stem, cuando, p))
    return sorted(filas, key=lambda f: f[1])


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


def analizar(salida, exp, cam, destino, entrada_dir=None, progreso=None, max_grupo=None):
    """Tracking + estadística de una cámara de un experimento. Con max_grupo
    la estadística usa solo células sueltas o en grupos de hasta ese tamaño."""
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
    aviso("midiendo cúmulos")
    grupos = grupos_experimento(filas, exp, cam, entrada_dir, float(det["area"].median()) * um ** 2, aviso)
    tr = tr.merge(grupos, on=["fecha", "label"], how="left")
    tr["grupo"] = tr["grupo"].fillna(1).astype(int)
    area_med = tr.groupby("track_id")["area_um2"].transform("median")
    restos = int(tr.loc[area_med < AREA_MIN_CELULA_UM2, "track_id"].nunique())
    tr = tr[area_med >= AREA_MIN_CELULA_UM2]
    if max_grupo:
        tr = tramos(tr, max_grupo)
        if tr.empty:
            raise ValueError(f"ninguna célula estuvo en grupos de hasta {max_grupo}")
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
               "celulas_por_ciclo": round(len(det) / len(filas), 1),
               "max_grupo": max_grupo,
               "trayectorias_descartadas_restos": restos,
               "fraccion_detecciones_sueltas": round(float((grupos["grupo"] == 1).mean()), 3),
               "fraccion_detecciones_hasta_3": round(float((grupos["grupo"] <= 3).mean()), 3)}
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
    (destino / "resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    aviso("listo")
    return resumen


def _fondo(entrada_dir, exp, cam, fecha, forma):
    """Imagen de fondo para dibujar las trayectorias: la suma de las fotos
    del último ciclo (o el campo claro que guardó la Pi), al tamaño de las
    máscaras. Si no está, fondo gris."""
    import cv2
    import tifffile
    from .en_vivo import grupos
    carpeta = Path(entrada_dir) / exp / cam if entrada_dir else None
    if carpeta and carpeta.is_dir():
        hay = grupos(carpeta).get(fecha, {})
        fotos = ([hay["_suma"]] if "_suma" in hay else
                 [hay[s] for s in ("_L", "_R", "_T", "_B", "") if s in hay])
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
