"""
Reporte HTML de tracking y morfología de un experimento del microscopio
propio, a partir de lo que deja el análisis de la interfaz (lib/propio.py).

Un solo archivo, sin dependencias: las gráficas van como SVG dentro de la
página (el texto toma los colores del tema claro u oscuro) y las fotos como
JPEG embebido.

Dos variantes:
  todas las células            analisis/propio/<exp>/<cam>/
  --max-grupo N                analisis/propio/<exp>/<cam>/hasta_N/
                               (solo células sueltas o en grupos de hasta N)

Uso:
  venv/bin/python scripts/35_reporte_experimento.py 2026-10-02_1618_prueba_7
  venv/bin/python scripts/35_reporte_experimento.py 2026-10-02_1618_prueba_7 --max-grupo 3
Salida (por defecto): analisis/propio/<exp>/reporte_todas.html o reporte_hasta_N.html
"""
import argparse
import base64
import datetime
import html
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT, cache_dir  # noqa: E402
from lib import estilo  # noqa: E402,F401  (fija rcParams)
from lib import en_vivo as EV  # noqa: E402
from lib import propio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Colores "centinela": la figura se dibuja con ellos y en el SVG se cambian
# por variables CSS del tema (tinta, tinta secundaria, grilla).
TINTA, TINTA2, GRILLA = "#010203", "#040506", "#070809"
CSSVAR = {TINTA: "var(--tinta)", TINTA2: "var(--tinta2)", GRILLA: "var(--grilla)"}
CAM_COLOR = {"cam0": estilo.CAT[0], "cam1": estilo.CAT[1], "cam2": estilo.CAT[2], "cam3": estilo.CAT[3]}
GRUPO_CAT = [(1, 1, "suelta"), (2, 3, "2–3"), (4, 10, "4–10"), (11, 10 ** 9, "más de 10")]
VENTANA_H = 6
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]

plt.rcParams.update({
    "svg.fonttype": "none", "font.family": "sans-serif",
    "font.sans-serif": ["Public Sans", "DejaVu Sans"],
    "text.color": TINTA, "axes.labelcolor": TINTA, "axes.edgecolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "grid.color": GRILLA,
    "axes.facecolor": "none", "figure.facecolor": "none", "savefig.facecolor": "none",
    "savefig.transparent": True, "font.size": 10, "axes.titlesize": 10.5,
    "legend.labelcolor": TINTA,
})


# ------------------------------------------------------------------ utilidades
def f(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(x):
        return "–"
    return f"{x:,.{d}f}".replace(",", " ").replace("-", "−")


def svg(fig):
    b = io.StringIO()
    fig.savefig(b, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = b.getvalue()
    s = s[s.index("<svg"):]
    for k, v in CSSVAR.items():
        s = s.replace(k, v).replace(k.upper(), v)
    s = re.sub(r'<svg ([^>]*?)width="[^"]*" height="[^"]*"', r'<svg \1', s, count=1)
    return f'<div class="fig">{s}</div>'


def jpg(rgb, calidad=82):
    import cv2
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, calidad])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def etiqueta_directa(ax, x, y, texto, color):
    ax.annotate(texto, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=9, color=TINTA, fontweight="bold")
    ax.plot([x], [y], "o", ms=5, color=color, mec="none")


def fecha_larga(d):
    return f"{d.day} de {MESES[d.month - 1]} de {d.year}, {d:%H:%M}"


# ------------------------------------------------------------------ datos
def cargar(exp, cams, max_grupo):
    an = cache_dir("analisis", "propio") / exp
    datos = {}
    for cam in cams:
        base = an / cam
        d = base / f"hasta_{max_grupo}" if max_grupo else base
        if not (d / "resumen.json").exists():
            sys.exit(f"falta el análisis de {cam} en {d}: correrlo desde la interfaz o con lib/propio.analizar")
        tr = pd.read_csv(d / "tracks.csv")
        todas = pd.read_csv(base / "tracks.csv") if max_grupo else tr
        datos[cam] = {"tr": tr, "todas": todas, "pc": pd.read_csv(d / "por_celula.csv"),
                      "res": json.loads((d / "resumen.json").read_text(encoding="utf-8"))}
    return datos


def preparar(tr, dt_min):
    """Pasos (velocidad instantánea) y forma de cada detección."""
    tr = tr.sort_values(["track_id", "frame"]).copy()
    um = 1.0
    g = tr.groupby("track_id")
    df = g["frame"].diff()
    tr["dx"] = g["x_um"].diff() / df
    tr["dy"] = g["y_um"].diff() / df
    tr["v"] = np.hypot(tr["dx"], tr["dy"]) / dt_min
    tr["t_h"] = tr["t_s"] / 3600
    per_um = tr["perimeter"] * np.sqrt(tr["area_um2"] / tr["area"].clip(lower=1)) * um
    tr["perimetro_um"] = per_um
    tr["circularidad"] = (4 * np.pi * tr["area_um2"] / per_um.clip(lower=1e-6) ** 2).clip(upper=1)
    tr["alargamiento"] = tr["major_axis_length"] / tr["minor_axis_length"].replace(0, np.nan)
    return tr


def msd_frames(tr, max_lag):
    """MSD promediado en tiempo y ensamble usando la diferencia real de
    frame (con huecos no se juntan dos tiempos distintos)."""
    s = np.zeros(max_lag + 1)
    c = np.zeros(max_lag + 1)
    for _, gg in tr.groupby("track_id"):
        fr = gg["frame"].to_numpy()
        xy = gg[["x_um", "y_um"]].to_numpy()
        for i in range(len(fr) - 1):
            lag = fr[i + 1:] - fr[i]
            ok = lag <= max_lag
            if not ok.any():
                continue
            d2 = ((xy[i + 1:][ok] - xy[i]) ** 2).sum(1)
            np.add.at(s, lag[ok], d2)
            np.add.at(c, lag[ok], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / c
    return np.arange(1, max_lag + 1), m[1:], c[1:]


def por_ventana(tr, col, horas=VENTANA_H):
    b = (tr["t_h"] // horas).astype(int)
    q = tr.groupby(b)[col].quantile([0.25, 0.5, 0.75]).unstack()
    n = tr.groupby(b)[col].count()
    q = q[n >= 20]
    return (q.index.to_numpy() + 0.5) * horas, q[0.25].to_numpy(), q[0.5].to_numpy(), q[0.75].to_numpy()


def resumir(D, T, Tf, Tall, MSD, cams, dt_min):
    """Números de cada cámara (motilidad por célula y paso a paso, forma sin
    las detecciones del borde, fracción de detecciones en grupos chicos)."""
    def alfa_ens(c):
        lag, m, n = MSD[c]
        ok = np.isfinite(m) & (n >= 20) & (m > 0)
        k = min(6, ok.sum())
        return np.polyfit(np.log(lag[ok][:k] * dt_min), np.log(m[ok][:k]), 1)[0] if k >= 3 else np.nan

    def deriva(c):
        g = T[c].dropna(subset=["dx"]).groupby("frame")[["dx", "dy"]].mean()
        return float(np.hypot(g["dx"].sum(), g["dy"].sum()))

    R = {}
    for c in cams:
        pc, r, t = D[c]["pc"], D[c]["res"], T[c]
        R[c] = {
            "ciclos": r["ciclos"], "cel_ciclo": r["celulas_por_ciclo"], "tray": len(pc),
            "dur": pc["duracion_min"].median() / 60, "dur4": int((pc["duracion_min"] >= 240).sum()),
            "v": pc["rapidez_um_min"].median(), "v25": pc["rapidez_um_min"].quantile(.25),
            "v75": pc["rapidez_um_min"].quantile(.75), "dir": pc["direccionalidad"].median(),
            "alfa": pc["alfa"].median(), "alfa_ens": alfa_ens(c), "ead": pc["ead1_corr"].median(),
            "neto": pc["desplazamiento_neto_um"].median(), "deriva": deriva(c),
            "area": Tf[c]["area_um2"].median(), "circ": Tf[c]["circularidad"].median(),
            "alarg": Tf[c]["alargamiento"].median(), "sol": Tf[c]["solidity"].median(),
            "vinst": t["v"].median(), "restos": r.get("trayectorias_descartadas_restos"),
            "sueltas": float((Tall[c]["grupo"] == 1).mean()), "hasta3": float((Tall[c]["grupo"] <= 3).mean()),
        }
    return R


def tipo_movimiento(a):
    if not np.isfinite(a):
        return "sin dato"
    if a < 0.9:
        return "confinado: se aleja menos de lo que daría una caminata al azar"
    if a <= 1.1:
        return "parecido a una caminata al azar"
    return "con algo de persistencia: se aleja más de lo que daría una caminata al azar"


# ------------------------------------------------------------------ figuras
def fig_conteo(seg, todas, cams, max_grupo):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for cam in cams:
        s = seg[seg["camara"] == cam].copy()
        s["t"] = pd.to_datetime(s["fecha"], format="%Y%m%d_%H%M%S")
        t0 = s["t"].min()
        s["h"] = (s["t"] - t0).dt.total_seconds() / 3600
        s = s.sort_values("h")
        suav = s["celulas"].rolling(12, center=True, min_periods=1).median()
        ax.plot(s["h"], s["celulas"], color=CAM_COLOR[cam], alpha=0.18, lw=0.8)
        ax.plot(s["h"], suav, color=CAM_COLOR[cam], lw=2)
        etiqueta_directa(ax, s["h"].iloc[-1], suav.iloc[-1], cam, CAM_COLOR[cam])
        if max_grupo:
            t = todas[cam]
            k = t[t["grupo"] <= max_grupo].groupby("frame").agg(n=("t_s", "size"), t=("t_s", "first"))
            h = k["t"].to_numpy() / 3600
            ax.plot(h, k["n"].rolling(12, center=True, min_periods=1).median(),
                    color=CAM_COLOR[cam], lw=1.6, ls=(0, (3, 2)))
    ax.set_xlabel("tiempo desde el inicio (h)")
    ax.set_ylabel("células por ciclo")
    ax.set_ylim(bottom=0)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.06)
    return svg(fig)


def fig_duracion(D, cams):
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bins = np.geomspace(25, max(D[c]["pc"]["duracion_min"].max() for c in cams) * 1.05, 30)
    for cam in cams:
        d = D[cam]["pc"]["duracion_min"]
        h, e = np.histogram(d, bins)
        ax.stairs(h, e, color=CAM_COLOR[cam], lw=2, label=f"{cam} (mediana {d.median() / 60:.1f} h)")
    ax.set_xscale("log")
    ticks = [30, 60, 120, 240, 480, 960, 1920, 3840]
    ax.set_xticks([t for t in ticks if t <= bins[-1]])
    ax.set_xticklabels([f"{t / 60:g} h" if t >= 60 else f"{t} min" for t in ticks if t <= bins[-1]])
    ax.set_xlabel("duración de la trayectoria")
    ax.set_ylabel("trayectorias")
    ax.legend(loc="upper right")
    return svg(fig)


def fig_ecdf(D, cams, cols):
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.2), sharey=True)
    axs = axs.ravel()
    for ax, (col, nombre, lim) in zip(axs, cols):
        for cam in cams:
            v = D[cam]["pc"][col].dropna().to_numpy()
            v = np.sort(v)
            if not len(v):
                continue
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=CAM_COLOR[cam], lw=2, label=cam)
            ax.axvline(np.median(v), color=CAM_COLOR[cam], lw=1, ls=(0, (2, 2)))
        ax.set_xlabel(nombre)
        if lim:
            ax.set_xlim(*lim)
    for ax in axs[::2]:
        ax.set_ylabel("fracción de células ≤ x")
    axs[0].legend(loc="lower right")
    return svg(fig)


def fig_tiempo(T, cams, col, nombre):
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for cam in cams:
        x, lo, me, hi = por_ventana(T[cam], col)
        ax.fill_between(x, lo, hi, color=CAM_COLOR[cam], alpha=0.16, lw=0)
        ax.plot(x, me, color=CAM_COLOR[cam], lw=2, marker="o", ms=4, label=f"{cam}, mediana y cuartiles")
    ax.set_xlabel(f"tiempo desde el inicio (h, ventanas de {VENTANA_H} h)")
    ax.set_ylabel(nombre)
    ax.set_xlim(0, None)
    ax.legend(loc="best")
    return svg(fig)


def fig_msd(MSD, dt_min, cams):
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for cam in cams:
        lag, m, c = MSD[cam]
        ok = np.isfinite(m) & (c >= 20)
        tau = lag[ok] * dt_min
        ax.loglog(tau, m[ok], "o-", color=CAM_COLOR[cam], ms=3.5, lw=1.4, label=cam)
    lag, m, c = MSD[cams[0]]
    ok = np.isfinite(m) & (c >= 20)
    if ok.sum() > 2:
        t0, m0 = lag[ok][0] * dt_min, m[ok][0]
        tt = np.array([t0, lag[ok][-1] * dt_min])
        ax.loglog(tt, m0 * (tt / t0), color=TINTA2, lw=1, ls=":", label="α = 1 (difusión)")
        ax.loglog(tt, m0 * (tt / t0) ** 2, color=TINTA2, lw=1, ls="--", label="α = 2 (balístico)")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, FuncFormatter
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 30, 60, 120, 180]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("intervalo τ (min)")
    ax.set_ylabel("MSD (µm²)")
    ax.legend(loc="upper left")
    return svg(fig)


def fig_deriva(T, cams):
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    for cam in cams:
        t = T[cam].dropna(subset=["dx"])
        g = t.groupby("frame")[["dx", "dy", "t_s"]].mean()
        h = g["t_s"].to_numpy() / 3600
        cx, cy = g["dx"].cumsum(), g["dy"].cumsum()
        ax.plot(h, cx, color=CAM_COLOR[cam], lw=2, label=f"{cam}, eje x")
        ax.plot(h, cy, color=CAM_COLOR[cam], lw=1.4, ls=(0, (3, 2)), label=f"{cam}, eje y")
    ax.axhline(0, color=TINTA2, lw=0.8)
    ax.set_xlabel("tiempo desde el inicio (h)")
    ax.set_ylabel("desplazamiento común\nacumulado (µm)")
    ax.set_xlim(0, None)
    ax.legend(loc="best", ncol=2)
    return svg(fig)


def fig_forma_dist(T, cams):
    cols = [("area_um2", "área (µm²)", (0, None)), ("circularidad", "circularidad 4πA/P²", (0, 1)),
            ("alargamiento", "alargamiento (eje mayor / menor)", (1, 5)), ("solidity", "solidez", (0.6, 1))]
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.0))
    axs = axs.ravel()
    for ax, (col, nombre, lim) in zip(axs, cols):
        hi = lim[1] if lim[1] is not None else np.nanpercentile(np.concatenate([T[c][col].to_numpy() for c in cams]), 99)
        bins = np.linspace(lim[0], hi, 40)
        for cam in cams:
            v = T[cam][col].dropna().to_numpy()
            h, e = np.histogram(v, bins, density=True)
            ax.stairs(h, e, color=CAM_COLOR[cam], lw=1.8, label=cam)
            ax.axvline(np.median(v), color=CAM_COLOR[cam], lw=1, ls=(0, (2, 2)))
        ax.set_xlabel(nombre)
        ax.set_yticks([])
    for ax in axs[::2]:
        ax.set_ylabel("densidad")
    axs[0].legend(loc="upper right")
    return svg(fig)


def fig_forma_vs_rapidez(D, T, cams):
    """Mediana de rapidez de las células agrupadas por quintil de su forma media."""
    cols = [("area_um2", "área media (µm²)"), ("circularidad", "circularidad media"),
            ("alargamiento", "alargamiento medio")]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.0), sharey=True)
    rho = {}
    for ax, (col, nombre) in zip(axs, cols):
        for cam in cams:
            m = T[cam].groupby("track_id")[col].mean()
            pc = D[cam]["pc"].set_index("track_id")
            x = m.reindex(pc.index)
            y = pc["rapidez_um_min"]
            ok = x.notna() & y.notna()
            x, y = x[ok], y[ok]
            if len(x) < 25:
                continue
            q = pd.qcut(x, 5, duplicates="drop")
            gx = x.groupby(q, observed=True).median()
            gy = y.groupby(q, observed=True)
            ax.plot(gx, gy.median(), "o-", color=CAM_COLOR[cam], lw=2, ms=5, label=cam)
            ax.fill_between(gx, gy.quantile(0.25), gy.quantile(0.75), color=CAM_COLOR[cam], alpha=0.14, lw=0)
            rho[(cam, col)] = x.corr(y, method="spearman")
        ax.set_xlabel(nombre)
    axs[0].set_ylabel("rapidez (µm/min)\nmediana y cuartiles")
    axs[0].legend(loc="upper right")
    return svg(fig), rho


def fig_por_grupo(T, cams):
    """Rapidez instantánea y área según el tamaño del grupo en el que está la célula."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.0))
    ancho = 0.8 / len(cams)
    for j, cam in enumerate(cams):
        t = T[cam]
        for k, (col, ax) in enumerate(((("v"), axs[0]), (("area_um2"), axs[1]))):
            med, lo, hi, ns = [], [], [], []
            for a, b, _ in GRUPO_CAT:
                v = t.loc[(t["grupo"] >= a) & (t["grupo"] <= b), col].dropna()
                ns.append(len(v))
                med.append(v.median() if len(v) >= 20 else np.nan)
                lo.append(v.quantile(0.25) if len(v) >= 20 else np.nan)
                hi.append(v.quantile(0.75) if len(v) >= 20 else np.nan)
            x = np.arange(len(GRUPO_CAT)) + (j - (len(cams) - 1) / 2) * ancho
            ax.vlines(x, lo, hi, color=CAM_COLOR[cam], lw=2)
            ax.plot(x, med, "o", color=CAM_COLOR[cam], ms=7, mec="white", mew=1.2, label=cam)
    for ax, nombre in zip(axs, ("rapidez instantánea (µm/min)", "área (µm²)")):
        ax.set_xticks(range(len(GRUPO_CAT)))
        ax.set_xticklabels([g[2] for g in GRUPO_CAT])
        ax.set_xlabel("células en el grupo")
        ax.set_ylabel(nombre)
    axs[0].legend(loc="upper right")
    return svg(fig)


# ------------------------------------------------------------------ imágenes
def fondo_plano(fotos, red):
    from scipy.ndimage import gaussian_filter
    s = EV.leer(fotos["_suma"], red)
    p = s / np.maximum(gaussian_filter(s, EV.SIGMA_FONDO_PX / red), 1e-6)
    return EV.a_8bits(p)


def ejemplos(exp, cam, entrada, masks_dir, max_grupo, n=3):
    from skimage.segmentation import find_boundaries
    import cv2
    cic = {f: fo for e, c, f, fo, su, ok in EV.ciclos(entrada) if e == exp and c == cam}
    filas = propio.serie(masks_dir, exp, cam)
    sel = [filas[i] for i in np.linspace(0, len(filas) - 1, n).astype(int)]
    t0 = filas[0][1]
    out = []
    for fecha, cuando, ruta in sel:
        z = np.load(ruta)
        m, um, red = z["masks"], float(z["um_por_px"]), int(z["reducir"])
        fo = cic.get(fecha)
        if not fo or "_suma" not in fo:
            continue
        g8 = fondo_plano(fo, red)
        rgb = np.dstack([g8] * 3).copy()
        if max_grupo:
            zona = None
            if "_dpcLR" in fo:
                zona = propio.zona_celulas(EV.leer(fo["_dpcLR"], red), EV.leer(fo["_dpcTB"], red), um)
            area_c = float(np.median(np.bincount(m.ravel())[1:][np.bincount(m.ravel())[1:] > 0])) * um ** 2
            gr = propio.grupos_ciclo(m, um, area_c, zona)
            sueltas = np.isin(m, [l for l, k in gr.items() if k <= max_grupo])
            from scipy.ndimage import binary_dilation
            b = binary_dilation(find_boundaries(m, mode="inner"))
            rgb[b & ~sueltas] = (225, 90, 80)
            rgb[b & sueltas] = (40, 200, 120)
        else:
            rgb[find_boundaries(m, mode="inner")] = (255, 196, 40)
        esc = 900 / rgb.shape[1]
        rgb = cv2.resize(rgb, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
        h = (cuando - t0).total_seconds() / 3600
        out.append((jpg(rgb), f"{cam} · {h:.0f} h · {int(m.max())} células"))
    return out


def trayectorias(exp, cam, entrada, masks_dir, tr):
    import cv2
    cic = {f: fo for e, c, f, fo, su, ok in EV.ciclos(entrada) if e == exp and c == cam}
    filas = propio.serie(masks_dir, exp, cam)
    fecha, _, ruta = filas[len(filas) // 2]
    z = np.load(ruta)
    red, um = int(z["reducir"]), float(z["um_por_px"])
    g8 = fondo_plano(cic[fecha], red)
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * g8.shape[0] / g8.shape[1]))
    ax.imshow(g8, cmap="gray", vmin=-60, vmax=330)
    from matplotlib.collections import LineCollection
    cmap = estilo.CMAP_SEQ
    tmax = tr["t_s"].max()
    segs, cols = [], []
    for _, gg in tr.groupby("track_id"):
        xy = gg[["x", "y"]].to_numpy()
        if len(xy) < 2:
            continue
        segs.append(xy)
        cols.append(cmap(0.25 + 0.75 * gg["t_s"].iloc[0] / max(tmax, 1)))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.6, alpha=0.75))
    ax.set_xlim(0, g8.shape[1])
    ax.set_ylim(g8.shape[0], 0)
    ax.axis("off")
    barra = 100 / um
    ax.plot([g8.shape[1] - 30 - barra, g8.shape[1] - 30], [g8.shape[0] - 30] * 2, color="white", lw=3)
    ax.text(g8.shape[1] - 30 - barra / 2, g8.shape[0] - 42, "100 µm", color="white", ha="center", fontsize=9)
    b = io.BytesIO()
    fig.savefig(b, format="jpg", dpi=130, bbox_inches="tight", pad_inches=0, facecolor="black")
    plt.close(fig)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


# ------------------------------------------------------------------ página
CSS = """
:root{
  /* Reporte de laboratorio: columna de lectura con figuras a todo el ancho
     de la columna; datos en monoespaciada, títulos sobrios. */
  --fondo:#f7f8fa; --papel:#ffffff; --tinta:#14181f; --tinta2:#566070; --grilla:#e3e6eb;
  --acento:#2a78d6; --aviso:#9a5b00; --aviso-fondo:#fff4e0; --borde:#d9dde4;
  --f-titulo:"Public Sans", "Segoe UI", system-ui, sans-serif;
  --f-texto:"Public Sans", "Segoe UI", system-ui, sans-serif;
  --f-datos:"JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --fondo:#0f1318; --papel:#161b22; --tinta:#e6e9ee; --tinta2:#9aa4b2; --grilla:#2a313b;
  --acento:#6da7ec; --aviso:#f0b35a; --aviso-fondo:#2b2213; --borde:#2a313b; color-scheme:dark}}
:root[data-theme="dark"]{
  --fondo:#0f1318; --papel:#161b22; --tinta:#e6e9ee; --tinta2:#9aa4b2; --grilla:#2a313b;
  --acento:#6da7ec; --aviso:#f0b35a; --aviso-fondo:#2b2213; --borde:#2a313b; color-scheme:dark}
body{background:var(--fondo);color:var(--tinta);font:16px/1.6 var(--f-texto)}
main{max-width:980px;margin:0 auto;padding-inline:20px;padding-block:40px 80px;display:flex;flex-direction:column;gap:44px}
header{display:flex;flex-direction:column;gap:10px}
.eyebrow{font:600 12px/1 var(--f-datos);letter-spacing:.08em;text-transform:uppercase;color:var(--acento)}
h1{font:700 clamp(28px,4vw,40px)/1.15 var(--f-titulo);margin:0;text-wrap:balance;letter-spacing:-.01em}
h2{font:700 24px/1.25 var(--f-titulo);margin:0;text-wrap:balance}
h3{font:600 17px/1.3 var(--f-titulo);margin:0}
p{margin:0;max-width:68ch}
section{display:flex;flex-direction:column;gap:16px}
.meta{display:flex;flex-wrap:wrap;gap:6px 22px;font:13px/1.5 var(--f-datos);color:var(--tinta2)}
.meta b{color:var(--tinta);font-weight:600}
.fig{background:var(--papel);border:1px solid var(--borde);border-radius:6px;padding:12px;overflow-x:auto}
.fig svg{display:block;width:100%;height:auto;min-width:560px}
.fig svg text{font-family:var(--f-texto)}
figure{margin:0;display:flex;flex-direction:column;gap:8px}
figcaption{font-size:14px;color:var(--tinta2);max-width:80ch}
.fotos{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.fotos figure img{width:100%;border-radius:4px;display:block}
.fotos figcaption{font:12.5px var(--f-datos)}
.tabla{overflow-x:auto;border:1px solid var(--borde);border-radius:6px;background:var(--papel)}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--grilla);white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font:600 12px var(--f-datos);color:var(--tinta2);text-transform:uppercase;letter-spacing:.04em}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--f-datos);font-size:13.5px}
.nota{background:var(--aviso-fondo);border-left:3px solid var(--aviso);padding:12px 16px;border-radius:0 6px 6px 0;max-width:80ch}
.nota p{max-width:none}
ul{margin:0;padding-left:20px;max-width:72ch;display:flex;flex-direction:column;gap:6px}
.clave{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}
.clave div{background:var(--papel);border:1px solid var(--borde);border-radius:6px;padding:14px 16px;display:flex;flex-direction:column;gap:4px}
.clave .v{font:600 24px/1.1 var(--f-datos)}
.clave .l{font-size:13px;color:var(--tinta2)}
.leyenda-color{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-1px;margin-right:4px}
footer{font-size:13px;color:var(--tinta2);border-top:1px solid var(--borde);padding-top:16px}
code{font-family:var(--f-datos);font-size:.9em}
@media (max-width:600px){.fig svg{min-width:480px}}
"""


def tabla(filas, cab):
    th = "".join(f"<th>{c}</th>" for c in cab)
    tr = "".join("<tr>" + "".join(f'<td class="{"num" if i else ""}">{v}</td>' for i, v in enumerate(r)) + "</tr>"
                 for r in filas)
    return f'<div class="tabla"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def figura(contenido, pie):
    return f"<figure>{contenido}<figcaption>{pie}</figcaption></figure>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experimento")
    ap.add_argument("--max-grupo", type=int, default=None)
    ap.add_argument("--entrada-dir", type=Path, default=ROOT / "datasets" / "microscopio_propio")
    ap.add_argument("--salida", type=Path, default=None)
    args = ap.parse_args()
    exp, mg = args.experimento, args.max_grupo
    masks_dir = cache_dir("masks", "propio")
    an = cache_dir("analisis", "propio") / exp
    cams = sorted(p.name for p in an.glob("cam*") if (p / "resumen.json").exists())
    D = cargar(exp, cams, mg)
    meta = {}
    mj = args.entrada_dir / exp / "experimento.json"
    if mj.exists():
        meta = json.loads(mj.read_text(encoding="utf-8"))
    imp = (meta.get("importado") or {}).get("de")
    if imp and (Path(imp) / "experimento.json").exists():
        meta = {**json.loads((Path(imp) / "experimento.json").read_text(encoding="utf-8")), **{"importado": meta.get("importado")}}
    seg = pd.read_csv(masks_dir / exp / "segmentacion.csv")
    seg = seg.drop_duplicates(["camara", "fecha"], keep="last")
    z0 = np.load(propio.serie(masks_dir, exp, cams[0])[0][2])
    conf = {k: (z0[k].item() if k in z0.files else None) for k in ("entrada", "reducir", "flow_threshold", "filtro_formas")}
    dt_min = D[cams[0]]["res"]["intervalo_min"]
    print("preparando datos…", flush=True)
    T = {c: preparar(D[c]["tr"], dt_min) for c in cams}
    Tall = {c: preparar(D[c]["todas"], dt_min) for c in cams} if mg else T
    # forma: sin las detecciones que tocan el borde de la imagen (salen cortadas)
    Tf = {c: T[c][~T[c]["borde"].astype(bool)] for c in cams}
    Tallf = {c: Tall[c][~Tall[c]["borde"].astype(bool)] for c in cams}
    MSD = {c: msd_frames(T[c], 36) for c in cams}
    resumen = resumir(D, T, Tf, Tall, MSD, cams, dt_min)
    # en la variante de grupos chicos, los mismos números con todas las células para comparar
    R0 = None
    if mg:
        D0 = cargar(exp, cams, None)
        R0 = resumir(D0, Tall, Tallf, Tall, {c: msd_frames(Tall[c], 36) for c in cams}, cams, dt_min)

    print("figuras…", flush=True)
    F = {}
    F["conteo"] = fig_conteo(seg, {c: Tall[c] for c in cams}, cams, mg)
    F["duracion"] = fig_duracion(D, cams)
    F["mot"] = fig_ecdf(D, cams, [("rapidez_um_min", "rapidez media (µm/min)", (0, float(np.nanpercentile(np.concatenate([D[c]["pc"]["rapidez_um_min"] for c in cams]), 98)))),
                                  ("direccionalidad", "direccionalidad (neto / recorrido)", (0, 1)),
                                  ("alfa", "α de cada célula", (0, 2.2)),
                                  ("ead1_corr", "EAD₁ corregida", (0.5, 1.05))])
    F["v_t"] = fig_tiempo(T, cams, "v", "rapidez instantánea\n(µm/min)")
    F["msd"] = fig_msd(MSD, dt_min, cams)
    F["deriva"] = fig_deriva(T, cams)
    F["forma"] = fig_forma_dist(Tf, cams)
    F["area_t"] = fig_tiempo(Tf, cams, "area_um2", "área (µm²)")
    F["circ_t"] = fig_tiempo(Tf, cams, "circularidad", "circularidad")
    F["forma_v"], rho = fig_forma_vs_rapidez(D, Tf, cams)
    F["grupo"] = fig_por_grupo(Tallf, cams)
    print("imágenes…", flush=True)
    fotos = {c: ejemplos(exp, c, args.entrada_dir, masks_dir, mg) for c in cams}
    trays = {c: trayectorias(exp, c, args.entrada_dir, masks_dir, T[c]) for c in cams}

    # ---------- texto
    nombre = meta.get("nombre", exp)
    try:
        ini = datetime.datetime.fromisoformat(meta["inicio"])
        fin = datetime.datetime.fromisoformat(meta["fin"])
        periodo = f"{fecha_larga(ini)} → {fecha_larga(fin)}"
    except (KeyError, ValueError):
        periodo = "–"
    variante = (f"Solo células sueltas o en grupos de hasta {mg}" if mg else "Todas las células")
    titulo = f"Prueba {nombre.split()[-1]} · {'células sueltas' if mg else 'todas las células'}" \
        if nombre.lower().startswith("prueba") else f"{nombre} · {'células sueltas' if mg else 'todas las células'}"
    camtxt = " y ".join(cams)
    R = resumen

    def por_cam(k, d=2, suf=""):
        return " · ".join(f"{c} {f(R[c][k], d)}{suf}" for c in cams)

    filas = [
        ["Ciclos segmentados", *[f(R[c]["ciclos"], 0) for c in cams]],
        ["Células por ciclo (todas)", *[f(R[c]["cel_ciclo"], 0) for c in cams]],
        ["Detecciones sueltas (grupo de 1)", *[f(100 * R[c]["sueltas"], 1) + " %" for c in cams]],
        ["Detecciones en grupos de hasta 3", *[f(100 * R[c]["hasta3"], 1) + " %" for c in cams]],
        ["Trayectorias analizadas (≥ 6 puntos)", *[f(R[c]["tray"], 0) for c in cams]],
        ["Duración mediana", *[f(R[c]["dur"], 1) + " h" for c in cams]],
        ["Trayectorias de 4 h o más", *[f(R[c]["dur4"], 0) for c in cams]],
        ["Trayectorias descartadas como basura (área &lt; 150 µm²)", *[f(R[c]["restos"], 0) for c in cams]],
        ["Rapidez instantánea (paso entre ciclos), mediana", *[f(R[c]["vinst"]) + " µm/min" for c in cams]],
        ["Rapidez media por célula, mediana (cuartiles)", *[f"{f(R[c]['v'])} ({f(R[c]['v25'])}–{f(R[c]['v75'])}) µm/min" for c in cams]],
        ["Direccionalidad, mediana", *[f(R[c]["dir"]) for c in cams]],
        ["Desplazamiento neto, mediana", *[f(R[c]["neto"], 1) + " µm" for c in cams]],
        ["α por célula, mediana", *[f(R[c]["alfa"]) for c in cams]],
        ["α del ensamble (primeros 30 min)", *[f(R[c]["alfa_ens"]) for c in cams]],
        ["EAD₁ corregida, mediana", *[f(R[c]["ead"]) for c in cams]],
        ["Área, mediana", *[f(R[c]["area"], 0) + " µm²" for c in cams]],
        ["Circularidad, mediana", *[f(R[c]["circ"]) for c in cams]],
        ["Alargamiento, mediana", *[f(R[c]["alarg"]) for c in cams]],
        ["Solidez, mediana", *[f(R[c]["sol"]) for c in cams]],
        ["Desplazamiento común acumulado al final", *[f(R[c]["deriva"], 0) + " µm" for c in cams]],
    ]
    if mg:
        filas = [r for r in filas if not r[0].startswith("Células por ciclo")]

    rho_filas = [[n, *[f(rho.get((c, k)), 2) for c in cams]] for k, n in
                 (("area_um2", "Área media"), ("circularidad", "Circularidad media"), ("alargamiento", "Alargamiento medio"))]

    if mg:
        intro = (f"<p>Este reporte usa solo las células que están <b>sueltas o en grupos de hasta {mg} células</b>. "
                 f"El tracking se hizo con todas las células, para no unir mal una trayectoria cuando una célula entra "
                 f"o sale de un cúmulo. Después, cada trayectoria se cortó en los tramos en los que la célula estuvo en un "
                 f"grupo chico, y solo los tramos de 6 puntos o más (30 min) entran en la estadística.</p>"
                 f"<p>Son pocas: en {camtxt}, entre el {f(100 * min(R[c]['hasta3'] for c in cams), 0)} % y el "
                 f"{f(100 * max(R[c]['hasta3'] for c in cams), 0)} % de las detecciones están en grupos de hasta {mg}. "
                 f"Casi todas las células de este experimento crecen en láminas o cúmulos.</p>")
    else:
        intro = ("<p>Este reporte usa <b>todas las células segmentadas</b>, incluidas las que están dentro de láminas y "
                 "cúmulos. Es lo que da el tracking de la interfaz tal como sale.</p>")

    v_txt = por_cam("v", 2, " µm/min")
    comparacion = ""
    if R0:
        filas_c = []
        for k, n, d, suf in (("tray", "Trayectorias analizadas", 0, ""), ("vinst", "Rapidez paso a paso, mediana", 2, " µm/min"),
                             ("v", "Rapidez media por célula, mediana", 2, " µm/min"), ("dir", "Direccionalidad, mediana", 2, ""),
                             ("neto", "Desplazamiento neto, mediana", 1, " µm"), ("alfa_ens", "α del ensamble", 2, ""),
                             ("area", "Área, mediana", 0, " µm²"), ("circ", "Circularidad, mediana", 2, ""),
                             ("alarg", "Alargamiento, mediana", 2, "")):
            filas_c.append([n, *[f"{f(R0[c][k], d)} → <b>{f(R[c][k], d)}</b>{suf}" for c in cams]])
        mas_redondas = all(R[c]["circ"] > R0[c]["circ"] + 0.05 for c in cams)
        mas_lentas = all(R[c]["vinst"] < R0[c]["vinst"] for c in cams)
        obs = []
        if mas_redondas:
            obs.append("son más redondas")
        if mas_lentas:
            obs.append("se mueven menos paso a paso")
        texto_obs = (f"<p>En las dos cámaras, las células sueltas o en grupos chicos {' y '.join(obs)} que el conjunto. "
                     "Muchas son células redondeadas que casi no avanzan (podrían estar dividiéndose o despegándose; "
                     "habría que verlo en el video); las que sí migran solas son pocas.</p>") if obs else ""
        n_tray = " y ".join(f"{R[c]['tray']} en {c}" for c in cams)
        comparacion = (f"<h3>Comparado con todas las células</h3>{tabla(filas_c, ['Medida', *[f'{c}: todas → grupos ≤ {mg}' for c in cams]])}"
                       f"{texto_obs}<p style=\"font-size:14px;color:var(--tinta2)\">Con {n_tray} trayectorias, "
                       "estas medianas son menos estables que las de todas las células.</p>")
    secciones = []
    secciones.append(f"""
<section id="resumen"><h2>Resumen</h2>{intro}
<div class="clave">
{''.join(f'<div><span class="v">{f(R[c]["tray"], 0)}</span><span class="l">trayectorias en {c}</span></div>' for c in cams)}
<div><span class="v">{f(np.median([R[c]['vinst'] for c in cams]))}</span><span class="l">µm/min, rapidez mediana paso a paso</span></div>
<div><span class="v">{f(np.median([R[c]['alfa_ens'] for c in cams]))}</span><span class="l">α del ensamble (1 = difusión)</span></div>
</div>
{tabla(filas, ["Medida", *cams])}
<ul>
<li><b>Rapidez:</b> paso a paso, la mediana es {por_cam("vinst", 2, " µm/min")}. Promediada por célula sale más alta ({v_txt}), porque algunos pasos grandes pesan mucho en el promedio; parte de esos pasos son intercambios entre células vecinas en las láminas (ver al final).</li>
<li><b>Dirección:</b> la direccionalidad mediana es {por_cam("dir")} (1 = línea recta; cerca de 0 = vuelve a donde empezó); el desplazamiento neto mediano es {por_cam("neto", 1, " µm")}.</li>
<li><b>Tipo de movimiento</b> (α del MSD del ensamble): {" · ".join(f"{c} {f(R[c]['alfa_ens'])}, {tipo_movimiento(R[c]['alfa_ens'])}" for c in cams)}.</li>
<li><b>Forma:</b> área mediana {por_cam("area", 0, " µm²")}, circularidad {por_cam("circ")}.</li>
</ul>
{comparacion}
</section>""")

    secciones.append(f"""
<section id="segmentacion"><h2>Segmentación</h2>
<p>Cellpose-SAM sobre la entrada <code>{conf['entrada']}</code> (campo claro aplanado + DPC izquierda-derecha + DPC arriba-abajo),
reducción {conf['reducir']} ({f(EV.UM_POR_PX * (conf['reducir'] or 1), 2)} µm/px), <code>flow_threshold</code> {conf['flow_threshold']},
con filtro de forma (se descartan objetos de menos de {EV.FORMA_AREA_MIN_UM2} µm², poco compactos o muy delgados).</p>
{figura(F['conteo'], 'Células segmentadas en cada ciclo (línea gruesa: mediana de 1 h; línea fina: cada ciclo).' + (f' Discontinua: células en grupos de hasta {mg}.' if mg else ''))}
<div class="fotos">{''.join(f'<figure><img src="{src}" alt="Contornos de la segmentación, {cap}"><figcaption>{cap}</figcaption></figure>' for c in cams for src, cap in fotos[c])}</div>
<p style="font-size:14px;color:var(--tinta2)">{('<span class="leyenda-color" style="background:#28c878"></span>en grupos de hasta ' + str(mg) + ' &nbsp; <span class="leyenda-color" style="background:#e15a50"></span>en cúmulos más grandes (no entran en este reporte). ') if mg else ''}Fondo: campo claro aplanado; inicio, mitad y final del experimento.</p>
<div class="nota"><p><b>Límite de la segmentación.</b> Hacia el final del experimento muchas células se aplanan, se extienden y se enciman. Los bordes entre ellas casi no se ven, y parte de esas láminas queda sin segmentar, sobre todo en cam1. Por eso el conteo baja al final: no es que se pierdan células. Para arreglarlo hace falta entrenar Cellpose con contornos dibujados a mano de estas imágenes.</p></div>
</section>""")

    secciones.append(f"""
<section id="tracking"><h2>Tracking</h2>
<p>Enlace por distancia con penalización por cambio de tamaño (laptrack, variante <code>dist_tam</code> del pipeline validado),
distancia máxima {f(max(6.0, 25.0 * dt_min / 5.0), 0)} µm entre ciclos y cierre de huecos de hasta 3 ciclos. Intervalo entre ciclos: {f(dt_min, 0)} min.</p>
<div class="fotos">{''.join(f'<figure><img src="{trays[c]}" alt="Trayectorias de {c}"><figcaption>{c} · {f(R[c]["tray"], 0)} trayectorias; color = hora de inicio (claro: al principio, oscuro: al final)</figcaption></figure>' for c in cams)}</div>
{figura(F['duracion'], 'Duración de las trayectorias analizadas. Las células entran y salen de foco, se dividen o se pierden en las láminas; pocas se siguen más de medio día.')}
</section>""")

    secciones.append(f"""
<section id="motilidad"><h2>Motilidad</h2>
{figura(F['mot'], 'Distribución acumulada por célula: rapidez media (recorrido / tiempo), direccionalidad (desplazamiento neto / recorrido), α de su MSD y EAD₁ corregida (1 = giros al azar). Línea punteada: mediana.')}
{figura(F['v_t'], f'Rapidez instantánea (paso entre ciclos) a lo largo del experimento: mediana y cuartiles en ventanas de {VENTANA_H} h.')}
{figura(F['msd'], 'Desplazamiento cuadrático medio del ensamble, con la diferencia real de tiempo entre puntos (los huecos no se juntan). Referencias: α = 1 (caminata al azar) y α = 2 (movimiento en línea recta).')}
<div class="nota"><p><b>Persistencia.</b> El ajuste de caminata persistente da un tiempo de persistencia menor que el intervalo entre fotos (5 min). Ese número no se puede medir con este muestreo y no se reporta. Para medirlo habría que tomar fotos cada 1 min o menos.</p></div>
{figura(F['deriva'], 'Desplazamiento común: promedio de los pasos de todas las células en cada ciclo, acumulado (continua: x; discontinua: y). Si fuera deriva de la platina, sería igual en las dos cámaras; si es la lámina que se mueve en bloque, cambia entre campos.')}
</section>""")

    secciones.append(f"""
<section id="morfologia"><h2>Morfología</h2>
<p>Medidas de cada detección a partir de su máscara (sin las que tocan el borde de la imagen, que salen cortadas): área, circularidad (4πA/P², 1 = círculo), alargamiento (eje mayor / eje menor de la elipse equivalente) y solidez (área / área de su envolvente convexa).</p>
{figura(F['forma'], 'Distribución de cada medida sobre todas las detecciones de las trayectorias analizadas. Línea punteada: mediana.')}
{figura(F['area_t'], f'Área a lo largo del experimento (mediana y cuartiles, ventanas de {VENTANA_H} h).')}
{figura(F['circ_t'], 'Circularidad a lo largo del experimento. Si baja, las células se extienden y se vuelven irregulares.')}
{figura(F['forma_v'], 'Forma contra movimiento: las células se ordenan por su forma media y se agrupan en quintiles; para cada quintil, mediana y cuartiles de su rapidez.')}
<h3>Correlación de Spearman entre forma media y rapidez por célula</h3>
{tabla(rho_filas, ["Forma media", *[f"ρ {c}" for c in cams]])}
<p style="font-size:14px;color:var(--tinta2)">|ρ| menor que 0.1: sin relación; 0.1–0.3: débil; mayor que 0.3: moderada. Sin valor p: las células de un mismo campo no son independientes.</p>
{figura(F['grupo'], 'Rapidez instantánea y área según cuántas células tiene el grupo en el que está la célula en ese ciclo (todas las detecciones con tracking; punto: mediana; barra: cuartiles).')}
</section>""")

    secciones.append(f"""
<section id="metodos"><h2>Cómo leer estos números</h2>
<ul>
<li>Es <b>un experimento con dos campos</b> ({camtxt}). Cada campo es una réplica técnica, no biológica. Las células de un mismo campo se tocan y se influyen, así que no son observaciones independientes y no tiene sentido sacar valores p célula por célula.</li>
<li><b>Cúmulos:</b> se cuentan juntas las células que se tocan y también la zona con células que se ve en el DPC aunque no esté segmentada. El tamaño de un grupo es el mayor entre las máscaras que tiene y su área dividida por el área típica de una célula.</li>
<li>En las láminas densas el tracking puede intercambiar células vecinas entre ciclos (el 1 % de los pasos mide unos 20 µm, el tamaño de una célula). Esto afecta poco a las medianas.</li>
<li><b>Basura:</b> se descartan las trayectorias cuya área mediana es menor que 150 µm². Son manchas compactas y casi inmóviles del medio, no células (abundan en cam1). Se usa la mediana de toda la trayectoria para no perder células que se redondean un rato al dividirse.</li>
<li>La escala ({f(EV.UM_POR_PX, 4)} µm/px a resolución completa) es la calibración del 20× con la cámara IMX219.</li>
</ul>
</section>""")

    gen = datetime.datetime.now()
    doc = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(titulo)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Public+Sans:wght@400;600;700&display=swap">
<style>{CSS}</style>
<main>
<header>
<span class="eyebrow">Tracking y morfología · {html.escape(variante)}</span>
<h1>{html.escape(titulo)}</h1>
<div class="meta"><span><b>Experimento</b> {html.escape(exp)}</span><span><b>Periodo</b> {periodo}</span>
<span><b>Intervalo</b> {f(dt_min, 0)} min</span><span><b>Cámaras</b> {camtxt}</span>
<span><b>Iluminación</b> DPC (4 mitades de la matriz LED)</span></div>
</header>
{''.join(secciones)}
<footer>Generado el {fecha_larga(gen)} con <code>scripts/35_reporte_experimento.py</code>{f' --max-grupo {mg}' if mg else ''} a partir de
<code>analisis/propio/{html.escape(exp)}/</code>. Los números salen de los archivos del análisis; si se vuelve a correr, el reporte se regenera.</footer>
</main>"""
    salida = args.salida or an / (f"reporte_hasta_{mg}.html" if mg else "reporte_todas.html")
    salida.write_text(doc, encoding="utf-8")
    print(f"listo: {salida} ({salida.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
