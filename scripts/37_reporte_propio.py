"""
Reporte completo de un experimento del microscopio propio, con el mismo
formato que el reporte v2 (23_reporte_html.py): índice, lo más importante,
cómo se analizó, cuánto se puede confiar, resultados por réplica, cambios en
el tiempo, forma, cúmulos, limitaciones, glosario y anexos.

Cada cámara es una caja de Petri distinta (misma línea, mismas condiciones):
cada una es una réplica.

Necesita, en analisis/propio/<exp>/:
  <cam>/tracks.csv ...                (lib/propio.analizar; con --max-grupo, <cam>/hasta_N/)
  estadisticas[/hasta_N]/*.csv        (36_estadisticas_propio.py)
  ajuste_segmentacion/barrido.csv     (comparación de entradas y parámetros)

Salida: analisis/propio/<exp>/reporte[_hasta_N]/index.html + figuras/ + videos/

Uso:
  venv/bin/python scripts/37_reporte_propio.py 2026-10-02_1618_prueba_7 --linea 4T1
  venv/bin/python scripts/37_reporte_propio.py 2026-10-02_1618_prueba_7 --linea 4T1 --max-grupo 3
"""
import argparse
import datetime
import html
import importlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT, cache_dir  # noqa: E402
from lib import estilo  # noqa: E402
from lib.estilo import CAT, TINTA, TINTA2, NEUTRO  # noqa: E402
from lib import en_vivo as EV  # noqa: E402
from lib import propio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PLANTILLA_CSS = Path(__file__).resolve().parent / "plantillas" / "reporte_propio.css"
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]
GRUPO_CAT = [(1, 1, "suelta"), (2, 3, "2–3"), (4, 10, "4–10"), (11, 10 ** 9, "más de 10")]
Q_JAMMING = 3.81
GRIS_L = "#d0d5da"
# Unidades de las gráficas (los textos y tablas siguen en minutos); --segundos las cambia.
U = {"t": "min", "tf": 1.0, "v": "µm/min", "vf": 1.0}


def usar_segundos():
    U.update(t="s", tf=60.0, v="µm/s", vf=1 / 60)


# ------------------------------------------------------------------ utilidades
def f(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(x):
        return "–"
    return f"{x:,.{d}f}".replace(",", " ").replace("-", "−")


def pct(x, d=0):
    return f(100 * x, d) + "%"


def fecha_larga(d, hora=True):
    s = f"{d.day} de {MESES[d.month - 1]} de {d.year}"
    return s + (f", {d:%H:%M}" if hora else "")


def tabla(filas, cab, num=None):
    num = num or [i > 0 for i in range(len(cab))]
    th = "".join(f'<th class="{"n" if n else ""}">{c}</th>' for c, n in zip(cab, num))
    tr = "".join("<tr>" + "".join(f'<td class="{"n" if n else ""}">{v}</td>' for v, n in zip(r, num)) + "</tr>"
                 for r in filas)
    return f'<div class="tabla"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


class Figuras:
    def __init__(self, carpeta):
        self.d = Path(carpeta) / "figuras"
        self.d.mkdir(parents=True, exist_ok=True)

    def png(self, fig, nombre):
        fig.savefig(self.d / f"{nombre}.png", dpi=170, facecolor="white")
        plt.close(fig)
        return f"figuras/{nombre}.png"

    def jpg(self, rgb, nombre, calidad=85):
        import cv2
        cv2.imwrite(str(self.d / f"{nombre}.jpg"), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, calidad])
        return f"figuras/{nombre}.jpg"


def figura(src, pie, alt=""):
    alt = html.escape(alt or pie.split(".")[0].replace("<b>", "").replace("</b>", ""))
    return (f'<figure><div class="placa"><img src="{src}" alt="{alt}" loading="lazy"></div>'
            f"<figcaption>{pie}</figcaption></figure>")


# ------------------------------------------------------------------ datos
def preparar_tracks(tr, dt_min):
    tr = tr.sort_values(["track_id", "frame"]).copy()
    g = tr.groupby("track_id")
    dfr = g["frame"].diff()
    tr["paso_um"] = np.hypot(g["x_um"].diff(), g["y_um"].diff())
    tr["v"] = tr["paso_um"] / (dfr * dt_min)
    tr["paso_um"] = tr["paso_um"] / dfr
    tr["t_h"] = tr["t_s"] / 3600
    tr["perimetro_um"] = tr["perimeter"] * np.sqrt(tr["area_um2"] / tr["area"].clip(lower=1))
    tr["circularidad"] = (4 * np.pi * tr["area_um2"] / tr["perimetro_um"].clip(lower=1e-6) ** 2).clip(upper=1)
    tr["alargamiento"] = tr["major_axis_length"] / tr["minor_axis_length"].replace(0, np.nan)
    tr["q"] = tr["perimetro_um"] / np.sqrt(tr["area_um2"])
    tr["borde"] = tr["borde"].astype(bool)
    return tr


# ------------------------------------------------------------------ figuras
def fig_condiciones(F, temp, foco, cams, nombre):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.1))
    t0 = temp["t"].min()
    h = (temp["t"] - t0).dt.total_seconds() / 3600
    ax[0].plot(h, temp["temperatura"], color=CAT[0], lw=1.3, label="medida")
    ax[0].plot(h, temp["setpoint"], color=NEUTRO, lw=1.2, ls="--", label="objetivo")
    ax[0].set_ylabel("temperatura (°C)")
    ax[0].set_xlabel("tiempo desde el inicio (h)")
    ax[0].set_title("Incubadora")
    ax[0].legend(loc="lower right")
    for i, cam in enumerate(cams):
        d = foco[foco["camara"] == int(cam[3:])]
        hh = (d["t"] - t0).dt.total_seconds() / 3600
        ax[1].plot(hh, d["desplazamiento_um"].cumsum(), color=CAT[i], lw=1.6, label=nombre[cam])
    ax[1].set_ylabel("corrección de foco acumulada (µm)")
    ax[1].set_xlabel("tiempo desde el inicio (h)")
    ax[1].set_title("Autofoco")
    ax[1].legend()
    return F.png(fig, "condiciones")


def fig_entradas(F, exp, entrada_dir, cam, fecha):
    """La misma imagen segmentada con las entradas que se probaron."""
    import cv2
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import find_boundaries
    cic = {fe: fo for e, c, fe, fo, su, ok in EV.ciclos(entrada_dir) if e == exp and c == cam}
    fo = cic[fecha]
    m = EV.cargar_modelo()
    casos = []
    for red, modo, ft, filtrar, nombre in ((2, "suma", 0.4, False, "Campo claro"),
                                            (2, "dpc", 0.4, False, "Magnitud del DPC"),
                                            (2, "vieja", 0.4, False, "Campo claro + DPC, sin aplanar"),
                                            (3, "combinada", 0.8, True, "Campo claro aplanado + DPC (final)")):
        imgs = {s: EV.leer(fo[s], red) for s in ("_dpcLR", "_dpcTB", "_suma")}
        if modo == "vieja":
            s = imgs["_suma"]
            z = lambda a: (a - a.mean()) / (a.std() + 1e-6)  # noqa: E731
            img, ax = np.stack([z(s), imgs["_dpcLR"] / imgs["_dpcLR"].std(), imgs["_dpcTB"] / imgs["_dpcTB"].std()], -1), 2
        else:
            img, ax = EV.preparar(imgs, ["_dpcLR", "_dpcTB", "_suma"], modo, red)
        mk, _, _ = m.eval(img, batch_size=16, channel_axis=ax, flow_threshold=ft, cellprob_threshold=0)
        if filtrar:
            mk = EV.filtrar_formas(mk, EV.UM_POR_PX * red)
        s = imgs["_suma"]
        fondo = EV.a_8bits(s / np.maximum(gaussian_filter(s, EV.SIGMA_FONDO_PX / red), 1e-6))
        rgb = np.dstack([fondo] * 3).copy()
        rgb[cv2.dilate(find_boundaries(mk, mode="inner").astype(np.uint8), np.ones((2, 2))) > 0] = (255, 190, 0)
        rgb = cv2.resize(rgb, (820, 616), interpolation=cv2.INTER_AREA)
        casos.append((rgb, f"{nombre}: {int(mk.max())} células"))
    filas = []
    for i in range(0, 4, 2):
        par = []
        for rgb, txt in casos[i:i + 2]:
            b = rgb.copy()
            cv2.rectangle(b, (0, 0), (b.shape[1], 34), (20, 24, 30), -1)
            cv2.putText(b, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 2, cv2.LINE_AA)
            par.append(b)
        filas.append(np.hstack([par[0], np.full((par[0].shape[0], 8, 3), 255, np.uint8), par[1]]))
    hueco = np.full((8, filas[0].shape[1], 3), 255, np.uint8)
    return F.jpg(np.vstack([filas[0], hueco, filas[1]]), "entradas"), [c[1] for c in casos]


def fig_conteo(F, seg, cams, nombre, todas=None, mg=None):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, cam in enumerate(cams):
        s = seg[seg["camara"] == cam].copy()
        s["t"] = pd.to_datetime(s["fecha"], format="%Y%m%d_%H%M%S")
        s = s.sort_values("t")
        h = (s["t"] - s["t"].min()).dt.total_seconds() / 3600
        ax.plot(h, s["celulas"], color=CAT[i], alpha=0.2, lw=0.7)
        ax.plot(h, s["celulas"].rolling(12, center=True, min_periods=1).median(), color=CAT[i], lw=2,
                label=f"{nombre[cam]}, todas")
        if mg and todas is not None:
            t = todas[cam]
            k = t[t["grupo"] <= mg].groupby("frame").agg(n=("t_s", "size"), t=("t_s", "first"))
            ax.plot(k["t"] / 3600, k["n"].rolling(12, center=True, min_periods=1).median(), color=CAT[i], lw=1.6,
                    ls=(0, (3, 2)), label=f"{nombre[cam]}, en grupos de hasta {mg}")
    ax.set_xlabel("tiempo desde el inicio (h)")
    ax.set_ylabel("células por ciclo")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2 if mg else 1)
    return F.png(fig, "conteo")


def fig_ejemplos(F, exp, entrada_dir, masks_dir, cams, nombre, mg):
    import cv2
    from scipy.ndimage import gaussian_filter, binary_dilation
    from skimage.segmentation import find_boundaries
    filas = []
    for cam in cams:
        cic = {fe: fo for e, c, fe, fo, su, ok in EV.ciclos(entrada_dir) if e == exp and c == cam}
        serie = propio.serie(masks_dir, exp, cam)
        t0 = serie[0][1]
        fila = []
        for i in np.linspace(0, len(serie) - 1, 3).astype(int):
            fecha, cuando, ruta = serie[i]
            z = np.load(ruta)
            mk, um, red = z["masks"], float(z["um_por_px"]), int(z["reducir"])
            fo = cic[fecha]
            s = EV.leer(fo["_suma"], red)
            rgb = np.dstack([EV.a_8bits(s / np.maximum(gaussian_filter(s, EV.SIGMA_FONDO_PX / red), 1e-6))] * 3).copy()
            b = binary_dilation(find_boundaries(mk, mode="inner"))
            if mg:
                zona = propio.zona_celulas(EV.leer(fo["_dpcLR"], red), EV.leer(fo["_dpcTB"], red), um)
                areas = np.bincount(mk.ravel())[1:]
                gr = propio.grupos_ciclo(mk, um, float(np.median(areas[areas > 0])) * um ** 2, zona)
                ok = np.isin(mk, [l for l, k in gr.items() if k <= mg])
                rgb[b & ~ok] = (225, 85, 75)
                rgb[b & ok] = (30, 200, 110)
            else:
                rgb[b] = (255, 190, 0)
            rgb = cv2.resize(rgb, (640, 481), interpolation=cv2.INTER_AREA)
            txt = f"{nombre[cam]} · {(cuando - t0).total_seconds() / 3600:.0f} h · {int(mk.max())} células"
            cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 30), (20, 24, 30), -1)
            cv2.putText(rgb, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA)
            fila.append(rgb)
            fila.append(np.full((rgb.shape[0], 6, 3), 255, np.uint8))
        filas.append(np.hstack(fila[:-1]))
        filas.append(np.full((6, filas[-1].shape[1], 3), 255, np.uint8))
    return F.jpg(np.vstack(filas[:-1]), "ejemplos")


def fig_contornos(F, exp, entrada_dir, masks_dir, cams, nombre, lado=420, esc=2):
    """Recortes ampliados (zona con más células) al inicio, a la mitad y al final
    de cada caja; cada célula con un color, para ver dónde termina una y empieza otra."""
    import colorsys
    import cv2
    from scipy.ndimage import gaussian_filter
    from skimage.segmentation import find_boundaries
    rng = np.random.default_rng(3)
    filas = []
    for cam in cams:
        cic = {fe: fo for e, c, fe, fo, su, ok in EV.ciclos(entrada_dir) if e == exp and c == cam}
        serie = propio.serie(masks_dir, exp, cam)
        t0 = serie[0][1]
        fila = []
        for i in np.linspace(0, len(serie) - 1, 3).astype(int):
            fecha, cuando, ruta = serie[i]
            z = np.load(ruta)
            mk, red, um = z["masks"], int(z["reducir"]), float(z["um_por_px"])
            s = EV.leer(cic[fecha]["_suma"], red)
            g8 = EV.a_8bits(s / np.maximum(gaussian_filter(s, EV.SIGMA_FONDO_PX / red), 1e-6))
            H, W = mk.shape
            dens = cv2.boxFilter((mk > 0).astype(np.float32), -1, (lado, lado))
            y, x = np.unravel_index(np.argmax(dens[lado // 2:H - lado // 2, lado // 2:W - lado // 2]), (H - lado, W - lado))
            m = cv2.resize(mk[y:y + lado, x:x + lado], None, fx=esc, fy=esc, interpolation=cv2.INTER_NEAREST)
            g = cv2.resize(g8[y:y + lado, x:x + lado], None, fx=esc, fy=esc, interpolation=cv2.INTER_CUBIC)
            rgb = np.dstack([g] * 3).copy()
            b = cv2.dilate(find_boundaries(m, mode="inner").astype(np.uint8), np.ones((3, 3))) > 0
            cols = np.array([[int(255 * c) for c in colorsys.hsv_to_rgb(hh, 0.9, 1.0)]
                             for hh in rng.uniform(0, 1, int(m.max()) + 1)], np.uint8)
            rgb[b] = cols[m[b]]
            h = (cuando - t0).total_seconds() / 3600
            cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 34), (20, 24, 30), -1)
            cv2.putText(rgb, f"{nombre[cam]} ({cam}) - {h:.0f} h - {len(np.unique(m)) - 1} celulas", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
            bar, y0 = int(50 / um * esc), rgb.shape[0] - 25
            cv2.line(rgb, (rgb.shape[1] - 30 - bar, y0), (rgb.shape[1] - 30, y0), (255, 255, 255), 5)
            cv2.putText(rgb, "50 um", (rgb.shape[1] - 30 - bar, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)
            fila += [rgb, np.full((rgb.shape[0], 8, 3), 255, np.uint8)]
        filas += [np.hstack(fila[:-1]), np.full((8, sum(a.shape[1] for a in fila[:-1]), 3), 255, np.uint8)]
    return F.jpg(np.vstack(filas[:-1]), "contornos", calidad=88)


def ocupacion(exp, entrada_dir, masks_dir, cams, cada=6):
    """Fracción del campo con células en cada ciclo (uno de cada `cada`):
    lo segmentado y la zona con células del DPC (que también cuenta lo que la
    segmentación pierde)."""
    filas = []
    for cam in cams:
        cic = {fe: fo for e, c, fe, fo, su, ok in EV.ciclos(entrada_dir) if e == exp and c == cam}
        serie = propio.serie(masks_dir, exp, cam)
        t0 = serie[0][1]
        for k in range(0, len(serie), cada):
            fecha, cuando, ruta = serie[k]
            z = np.load(ruta)
            m, red, um = z["masks"], int(z["reducir"]), float(z["um_por_px"])
            fo = cic[fecha]
            zona = propio.zona_celulas(EV.leer(fo["_dpcLR"], red), EV.leer(fo["_dpcTB"], red), um)
            filas.append({"cam": cam, "h": (cuando - t0).total_seconds() / 3600, "seg": float((m > 0).mean()),
                          "zona": float((zona | (m > 0)).mean()), "ancho_um": m.shape[1] * um, "alto_um": m.shape[0] * um})
    return pd.DataFrame(filas)


def fig_ocupacion(F, oc, cams, nombre):
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for i, cam in enumerate(cams):
        d = oc[oc["cam"] == cam].sort_values("h")
        ax.plot(d["h"], 100 * d["zona"].rolling(4, center=True, min_periods=1).median(), color=CAT[i], lw=2.2,
                label=f"{nombre[cam]}: zona con células (DPC)")
        ax.plot(d["h"], 100 * d["seg"].rolling(4, center=True, min_periods=1).median(), color=CAT[i], lw=1.4,
                ls=(0, (3, 2)), label=f"{nombre[cam]}: células segmentadas")
    ax.set_ylim(0, max(40, ax.get_ylim()[1]))
    ax.set_xlabel("tiempo desde el inicio (h)")
    ax.set_ylabel("% del campo ocupado")
    ax.legend(ncol=2, fontsize=8)
    return F.png(fig, "ocupacion")


def fig_escala(F, ancho, alto, diam, radios):
    """El campo a escala, con una célula y lo que se aleja en cada tiempo.
    radios: [(etiqueta, radio en µm, medido?)]"""
    from matplotlib.patches import Circle, Rectangle
    fig, ax = plt.subplots(figsize=(9, 9 * alto / ancho))
    ax.add_patch(Rectangle((0, 0), ancho, alto, fill=False, ec=TINTA2, lw=1.5))
    cx, cy = ancho / 2, alto / 2
    cols = [CAT[3], CAT[0], CAT[1], CAT[6]]
    ax.add_patch(Circle((cx, cy), diam / 2, color=CAT[2], alpha=0.85, label=f"una célula típica (~{diam:.0f} µm)"))
    for (lab, r, medido), col in zip(radios, cols):
        ax.add_patch(Circle((cx, cy), r, fill=False, ec=col, lw=2.2, ls="-" if medido else (0, (4, 3)),
                            label=f"se aleja en {lab}: {r:.0f} µm" + ("" if medido else " (extrapolado)")))
    ax.legend(loc="upper left", fontsize=10, frameon=True, facecolor="white", edgecolor=GRIS_L)
    ax.plot([20, 120], [20, 20], color=TINTA, lw=3)
    ax.text(70, 30, "100 µm", ha="center", fontsize=9)
    ax.text(ancho - 8, alto - 14, f"campo: {ancho:.0f} × {alto:.0f} µm", ha="right", fontsize=9, color=TINTA2)
    ax.set_xlim(-5, ancho + 5)
    ax.set_ylim(-5, alto + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    return F.png(fig, "escala")


def fig_pasos(F, T, cams, nombre):
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    bins = np.geomspace(0.1, 60, 50)
    for i, cam in enumerate(cams):
        p = T[cam]["paso_um"].dropna()
        h, e = np.histogram(p, bins)
        ax.stairs(h / h.sum(), e, color=CAT[i], lw=2, label=f"{nombre[cam]} (mediana {p.median():.1f} µm)")
    ax.axvline(20, color=NEUTRO, ls="--", lw=1)
    ax.text(21, ax.get_ylim()[1] * 0.9, "≈ una célula", color=TINTA2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("desplazamiento entre ciclos consecutivos (µm)")
    ax.set_ylabel("fracción de pasos")
    ax.legend(loc="upper left")
    return F.png(fig, "pasos")


def fig_msd(F, msd, peli, cams, nombre):
    from lib import motilidad as M
    fig, axs = plt.subplots(1, len(cams), figsize=(5.0 * len(cams), 3.7), sharey=True)
    for i, (ax, cam) in enumerate(zip(np.atleast_1d(axs), cams)):
        mv = i + 1
        d = msd[(msd["movie"] == mv) & (msd["pares"] >= 50)]
        tau = d["tau_s"].to_numpy() / 60
        m = d["msd_um2"].to_numpy()
        p = peli[peli["movie"] == mv].iloc[0]
        ax.loglog(tau * U["tf"], m, "o", ms=3.5, color=CAT[i], label="medido")
        if np.isfinite(p["prw_D_um2_min"]):
            fit = M.prw_msd(tau * 60, p["prw_D_um2_min"] / 60, p["prw_P_min"] * 60, p["prw_sigma_um"] ** 2)
            ax.loglog(tau * U["tf"], fit, color=CAT[1] if i == 0 else CAT[2], lw=1.6, label="caminata persistente + error")
        for (a, b), col, nom in (((5, 30), TINTA, "corto"), ((60, 240), CAT[2], "largo")):
            sel = (tau >= a) & (tau <= b)
            if sel.sum() >= 2:
                k, c0 = np.polyfit(np.log(tau[sel]), np.log(m[sel]), 1)
                tt = np.array([a, b])
                ax.loglog(tt * U["tf"], np.exp(c0) * tt ** k * 1.35, ls=":", color=col, lw=1.6, label=f"α {nom} = {k:.2f}")
        t0, m0 = tau[0], m[0]
        tt = np.array([tau[0], tau[-1]])
        ax.loglog(tt * U["tf"], m0 * tt / t0, color=NEUTRO, lw=0.8, ls="--")
        ax.set_title(f"{nombre[cam]}")
        ax.set_xlabel(f"intervalo τ ({U['t']})")
        ax.legend(loc="upper left", fontsize=7.5)
    np.atleast_1d(axs)[0].set_ylabel("MSD (µm²)")
    return F.png(fig, "msd")


def fig_alfa(F, msd, peli, cel, cams, nombre):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4))
    for i, cam in enumerate(cams):
        mv = i + 1
        p = peli[peli["movie"] == mv].iloc[0]
        axs[0].plot([0, 1], [p["alfa_corto"], p["alfa_largo"]], "o-", color=CAT[i], ms=8, lw=1.5, label=nombre[cam])
        a = cel[(cel["movie"] == mv)]["alfa_celula"].dropna()
        axs[1].boxplot(a.clip(-0.5, 3), positions=[i], widths=0.5, showfliers=False,
                       medianprops=dict(color=TINTA, lw=2), boxprops=dict(color=CAT[i]),
                       whiskerprops=dict(color=CAT[i]), capprops=dict(color=CAT[i]))
        d = msd[(msd["movie"] == mv) & (msd["pares"] >= 50)]
        tau = d["tau_s"].to_numpy() / 60
        lm = np.log(d["msd_um2"].to_numpy())
        al = np.gradient(lm, np.log(tau))
        sm = pd.Series(al).rolling(5, center=True, min_periods=1).mean()
        axs[2].semilogx(tau * U["tf"], sm, color=CAT[i], lw=2, label=nombre[cam])
    for ax in axs:
        ax.axhline(1, color=NEUTRO, ls="--", lw=1)
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(["corto\n5–30 min", "largo\n1–4 h"] if U["t"] == "min" else ["corto\n300–1800 s", "largo\n3600–14400 s"])
    axs[0].set_xlim(-0.4, 1.4)
    axs[0].set_ylabel("α del ensamble")
    axs[0].legend()
    axs[1].set_xticks(range(len(cams)))
    axs[1].set_xticklabels([nombre[c] for c in cams])
    axs[1].set_ylabel("α de cada célula")
    axs[2].set_xlabel(f"τ ({U['t']})")
    axs[2].set_ylabel("α local")
    axs[2].set_ylim(0.4, 1.6)
    return F.png(fig, "alfa")


def fig_vacf_ead(F, vacf, ead, sens, cams, nombre):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3))
    for i, cam in enumerate(cams):
        mv = i + 1
        v = vacf[vacf["movie"] == mv]
        axs[0].plot(v["tau_min"] * U["tf"], v["vacf"], "o-", ms=3, color=CAT[i], label=nombre[cam])
        e = ead[ead["movie"] == mv]
        axs[1].plot(e["tau_min"] * U["tf"], e["ead_ens_corr"], color=CAT[i], lw=2, label=f"{nombre[cam]}, conjunto")
        axs[1].plot(e["tau_min"] * U["tf"], e["ead_cel_media_corr"], color=CAT[i], lw=1.3, ls="--", label=f"{nombre[cam]}, por célula")
        s = sens[sens["movie"] == mv]
        axs[2].plot(s["paso_min"] * U["tf"], s["ead1_ens_corr"], "o-", color=CAT[i], label=nombre[cam])
    axs[0].axhline(0, color=NEUTRO, lw=1)
    axs[0].set_xlim(0, 60 * U["tf"])
    axs[0].set_xlabel(f"τ ({U['t']})")
    axs[0].set_ylabel("autocorrelación de la velocidad")
    axs[0].legend()
    axs[1].axhline(1, color=NEUTRO, ls="--", lw=1)
    axs[1].set_xlabel(f"τ ({U['t']})")
    axs[1].set_ylabel("EAD corregida")
    axs[1].legend(fontsize=7)
    axs[2].axhline(1, color=NEUTRO, ls="--", lw=1)
    axs[2].set_xlabel(f"paso de análisis ({U['t']})")
    axs[2].set_ylabel("EAD₁ corregida del conjunto")
    return F.png(fig, "vacf_ead")


def fig_distrib(F, cel, cams, nombre):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
    for ax, (col, nom, lim) in zip(axs, (("rapidez_um_min", f"rapidez media de la célula ({U['v']})", None),
                                         ("direccionalidad_1h", "direccionalidad en 1 h", (0, 1)))):
        for i, cam in enumerate(cams):
            v = np.sort(cel[cel["movie"] == i + 1][col].dropna().to_numpy()) * (U["vf"] if col == "rapidez_um_min" else 1)
            if len(v):
                ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=CAT[i], lw=2, label=nombre[cam])
                ax.axvline(np.median(v), color=CAT[i], ls=":", lw=1)
        ax.set_xlabel(nom)
        if lim:
            ax.set_xlim(*lim)
        else:
            ax.set_xlim(0, np.nanpercentile(cel[col], 98) * U["vf"])
    axs[0].set_ylabel("fracción de células ≤ x")
    axs[0].legend(loc="lower right")
    return F.png(fig, "distribuciones")


def fig_ventanas(F, ven, cams, nombre):
    cols = [("rapidez_um_min", f"rapidez ({U['v']})"), ("alfa_corto", "α corto"),
            ("ead1_ens_corr", "EAD₁ corregida"), ("densidad_cel_mm2", "células por mm²"),
            ("area_um2", "área (µm²)"), ("circularidad", "circularidad")]
    fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
    for ax, (col, nom) in zip(axs.ravel(), cols):
        for i, cam in enumerate(cams):
            d = ven[ven["camara"] == cam].sort_values("t0_h")
            x = (d["t0_h"] + d["t1_h"]) / 2
            ax.plot(x, d[col] * (U["vf"] if col == "rapidez_um_min" else 1), "o-", color=CAT[i], lw=2, ms=5, label=nombre[cam])
        ax.set_ylabel(nom)
        ax.set_xlabel("tiempo (h, centro de la ventana)")
    axs[0, 0].legend()
    return F.png(fig, "ventanas")


def fig_corr(F, corr, cams, nombre):
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    for i, cam in enumerate(cams):
        d = corr[corr["movie"] == i + 1]
        ax.plot(d["r_um"], d["corr_direccion"], "o-", color=CAT[i], lw=2, ms=4, label=nombre[cam])
    ax.axhline(0, color=NEUTRO, lw=1)
    ax.set_xlabel("distancia entre las dos células (µm)")
    ax.set_ylabel("coseno medio entre\nsus direcciones")
    ax.legend()
    return F.png(fig, "correlacion")


def fig_forma(F, T, cams, nombre):
    cols = [("area_um2", "área (µm²)", (0, None)), ("circularidad", "circularidad 4πA/P²", (0.2, 1)),
            ("alargamiento", "alargamiento (largo / ancho)", (1, 4)), ("q", "índice de forma q = P/√A", (3.4, 6))]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.0))
    for ax, (col, nom, lim) in zip(axs, cols):
        hi = lim[1] if lim[1] else np.nanpercentile(np.concatenate([T[c][col] for c in cams]), 99)
        bins = np.linspace(lim[0], hi, 40)
        for i, cam in enumerate(cams):
            v = T[cam][col].dropna()
            hh, e = np.histogram(v, bins, density=True)
            ax.stairs(hh, e, color=CAT[i], lw=1.8, label=nombre[cam])
            ax.axvline(v.median(), color=CAT[i], ls=":", lw=1)
        if col == "q":
            ax.axvline(Q_JAMMING, color=TINTA, lw=1, ls="--")
            ax.text(Q_JAMMING + 0.05, ax.get_ylim()[1] * 0.9, "3.81", fontsize=8, color=TINTA2)
        ax.set_xlabel(nom)
        ax.set_yticks([])
    axs[0].legend()
    return F.png(fig, "forma")


def fig_forma_rapidez(F, T, cams, nombre):
    """Rapidez instantánea según la forma de la célula en ese mismo ciclo."""
    cols = [("area_um2", "área (µm²)"), ("alargamiento", "alargamiento"), ("q", "índice de forma q")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    rho, extremos = {}, {}
    for ax, (col, nom) in zip(axs, cols):
        for i, cam in enumerate(cams):
            t = T[cam].dropna(subset=["v", col])
            q = pd.qcut(t[col], 10, duplicates="drop")
            gx = t.groupby(q, observed=True)[col].median()
            gy = t.groupby(q, observed=True)["v"]
            extremos[(cam, col)] = (gx.iloc[0], gy.median().iloc[0], gx.iloc[-1], gy.median().iloc[-1])
            ax.plot(gx, gy.median() * U["vf"], "o-", color=CAT[i], lw=2, ms=4, label=nombre[cam])
            ax.fill_between(gx, gy.quantile(.25) * U["vf"], gy.quantile(.75) * U["vf"], color=CAT[i], alpha=.12, lw=0)
            rho[(cam, col)] = t[col].corr(t["v"], method="spearman")
        if col == "q":
            ax.axvline(Q_JAMMING, color=TINTA, lw=1, ls="--")
        ax.set_xlabel(nom + " (deciles)")
    axs[0].set_ylabel(f"rapidez instantánea ({U['v']})\nmediana y cuartiles")
    axs[0].legend()
    return F.png(fig, "forma_rapidez"), rho, extremos


def fig_grupos(F, T, cams, nombre):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.2))
    ancho = 0.8 / len(cams)
    for j, cam in enumerate(cams):
        t = T[cam][~T[cam]["borde"]]
        for ax, col in zip(axs, ("v", "area_um2", "circularidad")):
            med, lo, hi = [], [], []
            for a, b, _ in GRUPO_CAT:
                v = t.loc[(t["grupo"] >= a) & (t["grupo"] <= b), col].dropna() * (U["vf"] if col == "v" else 1)
                ok = len(v) >= 20
                med.append(v.median() if ok else np.nan)
                lo.append(v.quantile(.25) if ok else np.nan)
                hi.append(v.quantile(.75) if ok else np.nan)
            x = np.arange(len(GRUPO_CAT)) + (j - (len(cams) - 1) / 2) * ancho
            ax.vlines(x, lo, hi, color=CAT[j], lw=2)
            ax.plot(x, med, "o", color=CAT[j], ms=7, mec="white", mew=1.2, label=nombre[cam])
    for ax, nom in zip(axs, (f"rapidez instantánea ({U['v']})", "área (µm²)", "circularidad")):
        ax.set_xticks(range(len(GRUPO_CAT)))
        ax.set_xticklabels([g[2] for g in GRUPO_CAT])
        ax.set_xlabel("células en el grupo")
        ax.set_ylabel(nom)
    axs[0].legend()
    return F.png(fig, "grupos")


def video_caja(carpeta, exp, entrada_dir, masks_dir, cam, tr, nombre, cada=3, fps=12, ancho=900):
    """Campo claro aplanado con contornos y la cola de cada trayectoria (100 min)."""
    import cv2
    from scipy.ndimage import gaussian_filter
    A = importlib.import_module("22_animaciones")
    ruta = Path(carpeta) / "videos" / f"{cam}.mp4"
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        return f"videos/{cam}.mp4"
    cic = {fe: fo for e, c, fe, fo, su, ok in EV.ciclos(entrada_dir) if e == exp and c == cam}
    serie = propio.serie(masks_dir, exp, cam)
    t0 = serie[0][1]
    frames = []
    for k in range(0, len(serie), cada):
        fecha, cuando, r = serie[k]
        z = np.load(r)
        mk, red = z["masks"], int(z["reducir"])
        s = EV.leer(cic[fecha]["_suma"], red)
        g8 = EV.a_8bits(s / np.maximum(gaussian_filter(s, EV.SIGMA_FONDO_PX / red), 1e-6))
        bgr = np.dstack([g8] * 3).copy()
        A.dibujar_contornos(bgr, mk, (0, 190, 255))
        A.dibujar_tray(bgr, tr, k, cola=20, grosor=2)
        esc = ancho / bgr.shape[1]
        bgr = cv2.resize(bgr, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
        h = (cuando - t0).total_seconds() / 3600
        frames.append(np.vstack([A.titulo(f"{nombre} · {h:5.1f} h · {int(mk.max())} células", bgr.shape[1]), bgr]))
    A.escribir_mp4(frames, ruta, fps)
    if ruta.stat().st_size > 14e6:          # límite de 15 MB por archivo al publicar
        import subprocess
        tmp = ruta.with_name(ruta.stem + "_tmp.mp4")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(ruta), "-c:v", "libx264", "-preset", "slow",
                        "-crf", "31", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)], check=True)
        tmp.replace(ruta)
    return f"videos/{cam}.mp4"


# ------------------------------------------------------------------ página
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experimento")
    ap.add_argument("--linea", default="")
    ap.add_argument("--max-grupo", type=int, default=None)
    ap.add_argument("--entrada-dir", type=Path, default=ROOT / "datasets" / "microscopio_propio")
    ap.add_argument("--salida", type=Path, default=None)
    ap.add_argument("--segundos", action="store_true", help="gráficas con el tiempo en segundos y la rapidez en µm/s")
    args = ap.parse_args()
    if args.segundos:
        usar_segundos()
    exp, mg, linea = args.experimento, args.max_grupo, args.linea.strip()
    an = cache_dir("analisis", "propio") / exp
    masks_dir = cache_dir("masks", "propio")
    est_dir = an / "estadisticas" / (f"hasta_{mg}" if mg else "")
    est0_dir = an / "estadisticas"
    cams = sorted(p.name for p in an.glob("cam*") if (p / "resumen.json").exists())
    nombre = {c: f"Caja {i}" for i, c in enumerate(cams, start=1)}
    nombre_l = {c: f"Caja {i} ({c})" for i, c in enumerate(cams, start=1)}
    out = args.salida or an / (f"reporte_hasta_{mg}" if mg else "reporte")
    out.mkdir(parents=True, exist_ok=True)
    F = Figuras(out)

    # ---------- metadatos del experimento (la carpeta original que trajo la Pi)
    meta = json.loads((args.entrada_dir / exp / "experimento.json").read_text(encoding="utf-8"))
    orig = Path((meta.get("importado") or {}).get("de") or args.entrada_dir / exp)
    if (orig / "experimento.json").exists():
        meta = {**meta, **json.loads((orig / "experimento.json").read_text(encoding="utf-8"))}
    ini = datetime.datetime.fromisoformat(meta["inicio"])
    fin = datetime.datetime.fromisoformat(meta["fin"])
    temp = pd.read_csv(orig / "temperatura.csv") if (orig / "temperatura.csv").exists() else None
    foco = pd.read_csv(orig / "autofoco.csv") if (orig / "autofoco.csv").exists() else None
    for d in (temp, foco):
        if d is not None:
            d["t"] = pd.to_datetime(d["timestamp"].astype(str), format="%Y%m%d_%H%M%S")

    # ---------- datos de análisis
    peli = pd.read_csv(est_dir / "por_pelicula.csv")
    ven = pd.read_csv(est_dir / "por_ventana.csv")
    cel = pd.read_csv(est_dir / "por_celula.csv")
    msd = pd.read_csv(est_dir / "msd.csv")
    vacf = pd.read_csv(est_dir / "vacf.csv")
    ead = pd.read_csv(est_dir / "ead_tau.csv")
    sens = pd.read_csv(est_dir / "sensibilidad_paso.csv")
    corr = pd.read_csv(est_dir / "corr_espacial.csv")
    peli0 = pd.read_csv(est0_dir / "por_pelicula.csv")
    res = {c: json.loads((an / c / (f"hasta_{mg}" if mg else "") / "resumen.json").read_text()) for c in cams}
    res0 = {c: json.loads((an / c / "resumen.json").read_text()) for c in cams}
    dt_min = res0[cams[0]]["intervalo_min"]
    um = res0[cams[0]]["um_por_px"]
    T0 = {c: preparar_tracks(pd.read_csv(an / c / "tracks.csv"), dt_min) for c in cams}
    T = {c: preparar_tracks(pd.read_csv(an / c / f"hasta_{mg}" / "tracks.csv"), dt_min) for c in cams} if mg else T0
    Tf = {c: T[c][~T[c]["borde"]] for c in cams}
    seg = pd.read_csv(masks_dir / exp / "segmentacion.csv").drop_duplicates(["camara", "fecha"], keep="last")
    seg = seg[seg["entrada"] == "combinada"] if "entrada" in seg else seg
    barrido = pd.read_csv(an / "ajuste_segmentacion" / "barrido.csv")
    P = {c: peli[peli["camara"] == c].iloc[0] for c in cams}
    P0 = {c: peli0[peli0["camara"] == c].iloc[0] for c in cams}
    z0 = np.load(propio.serie(masks_dir, exp, cams[0])[0][2])
    red = int(z0["reducir"])
    ft = float(z0["flow_threshold"]) if "flow_threshold" in z0.files else 0.4

    # ---------- figuras
    print("figuras…", flush=True)
    src = {}
    if temp is not None and foco is not None:
        src["condiciones"] = fig_condiciones(F, temp, foco, cams, nombre)
    src["entradas"], txt_entradas = fig_entradas(F, exp, args.entrada_dir, cams[0], "20261002_172330") \
        if not mg else (None, None)
    src["conteo"] = fig_conteo(F, seg, cams, nombre, T0, mg)
    src["ejemplos"] = fig_ejemplos(F, exp, args.entrada_dir, masks_dir, cams, nombre, mg)
    src["contornos"] = fig_contornos(F, exp, args.entrada_dir, masks_dir, cams, nombre)
    basura = an / "ajuste_segmentacion" / "sueltas_cam1.jpg"
    if basura.exists():
        shutil.copy(basura, F.d / "basura.jpg")
        src["basura"] = "figuras/basura.jpg"
    src["pasos"] = fig_pasos(F, T, cams, nombre)
    src["msd"] = fig_msd(F, msd, peli, cams, nombre)
    src["alfa"] = fig_alfa(F, msd, peli, cel, cams, nombre)
    src["vacf_ead"] = fig_vacf_ead(F, vacf, ead, sens, cams, nombre)
    src["distrib"] = fig_distrib(F, cel, cams, nombre)
    ven_ok = ven.groupby("camara")["n_segmentos"].min().min() >= 20
    if ven_ok:
        src["ventanas"] = fig_ventanas(F, ven, cams, nombre)
    src["corr"] = fig_corr(F, corr, cams, nombre)
    src["forma"] = fig_forma(F, Tf, cams, nombre)
    src["forma_rapidez"], rho, extremos = fig_forma_rapidez(F, Tf, cams, nombre)
    qx = {c: extremos[(c, "q")] for c in cams}
    q_txt = (f"las células del decil más irregular (q ≈ {' y '.join(f(qx[c][2], 1) for c in cams)}) se mueven paso a paso a "
             f"{' y '.join(f(qx[c][3]) for c in cams)} µm/min, contra {' y '.join(f(qx[c][1]) for c in cams)} µm/min en el decil "
             f"más compacto (q ≈ {' y '.join(f(qx[c][0], 1) for c in cams)}); ρ de Spearman {' y '.join(f(rho[(c, 'q')]) for c in cams)}")
    src["grupos"] = fig_grupos(F, T0, cams, nombre)
    print("ocupación…", flush=True)
    oc = ocupacion(exp, args.entrada_dir, masks_dir, cams)
    oc.to_csv(est_dir / "ocupacion.csv", index=False)
    src["ocupacion"] = fig_ocupacion(F, oc, cams, nombre)
    campo_w, campo_h = float(oc["ancho_um"].iloc[0]), float(oc["alto_um"].iloc[0])
    # MSD promedio de las dos cajas a 1 h y 5 h; más allá, extrapolado con α largo
    def msd_en(tau_s):
        return float(np.mean([msd[(msd["movie"] == i + 1) & (msd["tau_s"].round() == tau_s)]["msd_um2"].mean()
                              for i in range(len(cams))]))
    m1, m5 = msd_en(3600), msd_en(18000)
    a_l = float(np.mean([P[c]["alfa_largo"] for c in cams]))
    t_tot = float(oc["h"].max())
    rad1, rad5 = np.sqrt(m1), np.sqrt(m5)
    rad24 = np.sqrt(m5 * (24 / 5) ** a_l)
    radtot = np.sqrt(m5 * (t_tot / 5) ** a_l)
    diam = float(2 * np.sqrt(np.mean([P[c]["area_um2"] for c in cams]) / np.pi))
    src["escala"] = fig_escala(F, campo_w, campo_h, diam,
                               [("1 h", rad1, True), ("5 h", rad5, True), ("24 h", rad24, False), (f"{t_tot:.0f} h", radtot, False)])
    # horas para alejarse la mitad del ancho del campo, si siguiera igual
    t_mitad = 5 * ((campo_w / 2) ** 2 / m5) ** (1 / a_l)
    recorrido_5h = {c: float(T0[c]["v"].median()) * 300 for c in cams}
    zona_ini = {c: 100 * oc[(oc["cam"] == c) & (oc["h"] < 12)]["zona"].mean() for c in cams}
    zona_fin = {c: 100 * oc[(oc["cam"] == c) & (oc["h"] >= oc["h"].max() - 12)]["zona"].mean() for c in cams}
    seg_ini = {c: 100 * oc[(oc["cam"] == c) & (oc["h"] < 12)]["seg"].mean() for c in cams}
    seg_fin = {c: 100 * oc[(oc["cam"] == c) & (oc["h"] >= oc["h"].max() - 12)]["seg"].mean() for c in cams}
    print("videos…", flush=True)
    videos = {c: video_caja(an / "reporte", exp, args.entrada_dir, masks_dir, c, T0[c], nombre_l[c]) for c in cams}
    if mg:  # el reporte de grupos chicos reusa los videos del reporte completo
        for c in cams:
            (out / "videos").mkdir(exist_ok=True)
            for ext in (".mp4", ".jpg"):
                shutil.copy(an / "reporte" / "videos" / f"{c}{ext}", out / "videos" / f"{c}{ext}")

    # ---------- números que usa el texto
    def ambos(col, d=2, fuente=None, suf=""):
        fu = fuente or P
        return " y ".join(f"{f(fu[c][col], d)}{suf}" for c in cams)

    def prom(col, fuente=None):
        fu = fuente or P
        return float(np.mean([fu[c][col] for c in cams]))

    def dif(col, fuente=None):
        fu = fuente or P
        a, b = (fu[c][col] for c in cams[:2])
        return abs(a - b) / max(abs((a + b) / 2), 1e-9)

    n_det = {c: res0[c]["detecciones"] for c in cams}
    frac3 = {c: float((T0[c]["grupo"] <= 3).mean()) for c in cams}
    frac1 = {c: float((T0[c]["grupo"] == 1).mean()) for c in cams}
    vinst = {c: float(T[c]["v"].median()) for c in cams}
    paso_med = {c: float(T[c]["paso_um"].median()) for c in cams}
    paso_20 = {c: float((T[c]["paso_um"] > 20).mean()) for c in cams}
    msd5h = {c: float(msd[(msd["movie"] == i + 1) & (msd["tau_s"].round() == 18000)]["msd_um2"].mean())
             for i, c in enumerate(cams)}
    deriva = {}
    for c in cams:
        t = T0[c].sort_values(["track_id", "frame"])
        g = t.groupby("track_id")
        dd = pd.DataFrame({"frame": t["frame"], "dx": g["x_um"].diff() / g["frame"].diff(),
                           "dy": g["y_um"].diff() / g["frame"].diff()}).dropna().groupby("frame")[["dx", "dy"]].mean()
        deriva[c] = float(np.hypot(dd["dx"].sum(), dd["dy"].sum()))
    texto_ventanas = "<p>Repetimos todo en ventanas de 12 horas.</p>"
    for c in cams:
        v = ven[ven["camara"] == c].sort_values("t0_h")
        if len(v) < 2:
            continue
        r0, r1 = v.iloc[0], v.iloc[-1]
        cae = r1["densidad_cel_mm2"] < 0.7 * v["densidad_cel_mm2"].max()
        texto_ventanas += (f"<p><b>{nombre_l[c]}.</b> Rapidez entre {f(v['rapidez_um_min'].min())} y {f(v['rapidez_um_min'].max())} µm/min "
                           f"(primera ventana {f(r0['rapidez_um_min'])}, última {f(r1['rapidez_um_min'])}). α a escala corta {f(r0['alfa_corto'])} en las "
                           f"primeras 12 h y entre {f(v['alfa_corto'].iloc[1:].min())} y {f(v['alfa_corto'].iloc[1:].max())} después. Área típica de "
                           f"{f(r0['area_um2'], 0)} µm² al principio y {f(r1['area_um2'], 0)} µm² al final, con {f(r0['densidad_cel_mm2'], 0)} y "
                           f"{f(r1['densidad_cel_mm2'], 0)} células/mm² segmentadas."
                           + (" En las últimas ventanas la densidad segmentada cae mucho: parte de las células ya no se segmenta (las láminas aplanadas), "
                              "así que esas ventanas describen solo a las células que sí se detectan y no son comparables con las primeras." if cae else "")
                           + "</p>")
    temp_txt = ""
    if temp is not None:
        tt = temp[temp["t"] > temp["t"].min() + pd.Timedelta(hours=2)]["temperatura"]
        temp_txt = (f"Fuera de las primeras dos horas (calentamiento), la incubadora se mantuvo en {f(tt.median(), 1)} °C "
                    f"(entre {f(tt.quantile(.01), 1)} y {f(tt.quantile(.99), 1)} °C en el 98% de los ciclos), con objetivo de "
                    f"{f(temp['setpoint'].median(), 1)} °C.")

    tit_var = f"Células sueltas o en grupos de hasta {mg}" if mg else "Todas las células"
    linea_txt = linea or "la línea celular"
    linea_desc = (f"<b>{linea}</b>, una línea de carcinoma mamario de ratón muy metastásica que en cultivo "
                  "crece en islas de células pegadas entre sí") if linea.upper() == "4T1" else f"<b>{linea_txt}</b>"

    # ---------- secciones
    S = []
    indice = [("resumen", "Lo más importante"), ("contexto", "Qué medimos y por qué"), ("datos", "El experimento"),
              ("proceso", "Cómo se analizó"), ("validacion", "¿Son confiables las mediciones?"),
              ("persistencia", "Cómo se mide la persistencia"), ("resultados", "Resultados por caja"),
              ("tiempo", "Cambios a lo largo de las 72 h"), ("vecinas", "Movimiento en bloque"),
              ("forma", "Forma de las células"), ("cumulos", "Células sueltas y cúmulos"),
              ("comparacion", "Comparación con el reporte anterior"), ("limitaciones", "Limitaciones"),
              ("siguiente", "Próximos pasos"), ("glosario", "Glosario"), ("replicas", "Anexo: cifras por caja y por ventana"),
              ("tecnico", "Anexo técnico")]
    if not ven_ok:
        indice = [x for x in indice if x[0] != "tiempo"]
    num = {k: i for i, (k, _) in enumerate(indice[1:], start=1)}

    def h2(k, t):
        return f'<h2><span class="num">{num[k]}</span>{t}</h2>'

    # --- lo más importante
    if not mg:
        hall = [
            f"<b>La segmentación funciona mejor combinando el campo claro aplanado con el DPC.</b> En la misma foto, el modelo encontraba {txt_entradas[0].split(': ')[1]} con solo campo claro y {txt_entradas[3].split(': ')[1]} con la configuración final, que divide las láminas célula por célula.",
            f"<b>Las dos cajas dan resultados casi iguales</b>, aunque la caja 1 tiene el doble de células por mm² ({ambos('densidad_cel_mm2', 0)}). Rapidez media {ambos('rapidez_um_min')} µm/min, α a escala corta {ambos('alfa_corto')}, EAD₁ del conjunto {ambos('ead1_ens_corr', 3)}.",
            f"<b>Las células se mueven despacio y confinadas.</b> α está por debajo de 1 tanto a escala corta (5–30 min) como larga (1–4 h). En 5 horas una célula se aleja en promedio unos {f(np.sqrt(np.mean(list(msd5h.values()))), 0)} µm de donde estaba, poco más que su propio tamaño. Entre el {pct(min(P[c]['frac_celulas_subdifusivas'] for c in cams))} y el {pct(max(P[c]['frac_celulas_subdifusivas'] for c in cams))} de las células son subdifusivas y solo el {pct(prom('frac_celulas_superdifusivas'))} superdifusivas.",
            f"<b>No hay persistencia direccional medible con fotos cada 5 minutos.</b> La autocorrelación de la velocidad a un paso es levemente negativa ({ambos('vacf_lag1')}), el coseno medio del giro es casi 0 ({ambos('cos_giro_medio')}) y la EAD₁ corregida queda en {ambos('ead1_ens_corr', 3)}; con pasos de 10–20 min se acerca aún más a 1. El modelo de caminata persistente no puede medir un tiempo de persistencia: sale menor que el intervalo entre fotos.",
            f"<b>Las láminas se mueven en bloque.</b> La dirección de una célula se parece a la de sus vecinas ({ambos('corr_dir_0_50um')} a menos de 50 µm) y esa coordinación casi no decae con la distancia, lo que indica movimiento colectivo de las islas o una deriva común del campo.",
            f"<b>Forma:</b> área típica {ambos('area_um2', 0, suf=' µm²')}, alargamiento {ambos('aspect_ratio')} y circularidad {ambos('circularidad')}. <b>Las células más irregulares se mueven más:</b> {q_txt}. Es la misma dirección que predice la transición sólido–fluido (q por encima de 3.81), aunque aquí casi todas las células están por encima de ese valor.",
            f"<b>Casi todas las células viven en cúmulos.</b> Solo el {pct(min(frac3.values()), 0)}–{pct(max(frac3.values()), 0)} de las detecciones está sola o en un grupo de hasta 3 células. Las sueltas son más redondas y se mueven menos paso a paso (ver el reporte de células sueltas).",
            "<b>Límite principal:</b> hacia el día 3 muchas células se aplanan, se extienden y se enciman, y la segmentación pierde parte de esas láminas, sobre todo en la caja 2. El paso siguiente es entrenar Cellpose con contornos dibujados a mano de estas imágenes.",
        ]
    else:
        hall = [
            f"<b>Este reporte usa solo las células sueltas o en grupos de hasta {mg}.</b> Son pocas: el {pct(min(frac3.values()), 0)}–{pct(max(frac3.values()), 0)} de las detecciones. Quedan {ambos('n_segmentos', 0)} tramos de al menos una hora para el análisis, contra {ambos('n_segmentos', 0, P0)} con todas las células.",
            f"<b>Se mueven menos paso a paso que el conjunto</b> ({' y '.join(f(vinst[c]) for c in cams)} µm/min de mediana instantánea, contra {' y '.join(f(T0[c]['v'].median()) for c in cams)} con todas las células) y son más redondas (circularidad {ambos('circularidad')} contra {ambos('circularidad', 2, P0)}).",
            (f"<b>Las dos cajas no concuerdan en la persistencia.</b> α a escala corta {ambos('alfa_corto')}, coseno medio del giro {ambos('cos_giro_medio')} y EAD₁ corregida {ambos('ead1_ens_corr', 3)}. "
             + ("Una de las cajas muestra algo de persistencia (α mayor que 1 y coseno positivo) y la otra no. " if max(P[c]['alfa_corto'] for c in cams) > 1.1 and min(P[c]['alfa_corto'] for c in cams) < 0.95 else "")
             + "Con tan pocas células no se puede saber si es real: haría falta repetirlo con más células sueltas."),
            "<b>Interpretación:</b> en este experimento las células que quedan solas suelen ser células redondeadas que casi no avanzan (podrían estar dividiéndose o despegándose; habría que confirmarlo en el video). Las que migran solas son muy pocas. Para estudiar migración individual conviene sembrar a menor densidad.",
        ]
    S.append(f'<section class="hallazgos" id="resumen"><h2>Lo más importante</h2><ol>{"".join(f"<li>{h}</li>" for h in hall)}</ol></section>')

    # --- contexto
    S.append(f"""<section id="contexto" class="texto">{h2('contexto', 'Qué medimos y por qué')}
<p>Filmamos células de {linea_desc} durante {f((fin - ini).total_seconds() / 3600, 0)} horas en el microscopio propio, dentro de la incubadora, con una foto cada {f(dt_min, 0)} minutos. Como en el reporte anterior, medimos <b>cuánto</b> se mueven (rapidez) y <b>cómo</b>: si avanzan en una dirección sostenida (persistencia) o cambian de rumbo todo el tiempo, y si se mueven solas o en grupo. Además medimos la <b>forma</b> de cada célula en cada foto.</p>
<p>La diferencia con el reporte anterior es la imagen. Aquí no hay video público de referencia ni núcleos teñidos: todo sale de nuestras propias fotos con iluminación <b>DPC</b> (contraste de fase diferencial), que la matriz de LEDs del microscopio produce encendiendo por turnos la mitad izquierda, derecha, superior e inferior. La resta de cada par resalta el relieve de las células, que en campo claro casi no se ve.</p>
</section>""")

    # --- datos
    cards = ""
    for c in cams:
        cards += (f'<article class="dato"><img src="{videos[c][:-4]}.jpg" alt="{nombre_l[c]}" loading="lazy"><div><h4>{nombre_l[c]}</h4>'
                  f'Misma línea y mismas condiciones; caja de Petri independiente.'
                  f'<p class="meta">{res0[c]["ciclos"]} ciclos · {f(res0[c]["celulas_por_ciclo"], 0)} células por ciclo · {f(P0[c]["densidad_cel_mm2"], 0)} células/mm² · {res0[c]["trayectorias_analizadas"]} trayectorias</p></div></article>')
    S.append(f"""<section id="datos">{h2('datos', 'El experimento')}
<div class="texto"><p><b>{html.escape(meta.get('nombre', exp))}</b>: del {fecha_larga(ini)} al {fecha_larga(fin)} ({html.escape(meta.get('estado', ''))}). Una foto cada {f(dt_min, 0)} min en cada caja, {meta.get('ciclos', '')} ciclos. Objetivo 20×, cámara IMX219, {f(EV.UM_POR_PX, 4)} µm por píxel a resolución completa (el análisis trabaja a {f(um, 2)} µm/px). Cada cámara mira una <b>caja de Petri distinta</b> con la misma línea y las mismas condiciones, así que cada caja es una réplica.</p>
<p>La Pi calcula el DPC al terminar cada ciclo y borra las cuatro fotos crudas: quedan el relieve izquierda–derecha, el relieve arriba–abajo y una foto de campo claro (la suma de las cuatro).</p></div>
<div class="rejilla-datos">{cards}</div>
{figura(src['condiciones'], f'<b>Condiciones durante el experimento.</b> Izquierda: temperatura de la incubadora en cada ciclo. Derecha: cuánto movió el autofoco en total en cada caja. {temp_txt}') if 'condiciones' in src else ''}
<div class="texto"><h3>Cuánto espacio ocupan</h3><p>Según la zona con células que se ve en el DPC, las células cubren {' y '.join(f(zona_ini[c], 0) + '%' for c in cams)} del campo en las primeras 12 h y {' y '.join(f(zona_fin[c], 0) + '%' for c in cams)} en las últimas: más de tres cuartos del campo queda libre todo el experimento y las islas no lo invaden. Lo segmentado baja más ({' y '.join(f(seg_ini[c], 0) + '% → ' + f(seg_fin[c], 0) + '%' for c in cams)}) porque al final la segmentación pierde las láminas aplanadas.</p></div>
{figura(src['ocupacion'], '<b>Porcentaje del campo ocupado por células</b> (mediana de 2 h). Continua: zona con células según el relieve DPC. Discontinua: área de las células segmentadas.')}
<div class="rejilla-videos">{''.join(f'<figure><div class="placa"><video src="{videos[c]}" poster="{videos[c][:-4]}.jpg" controls muted loop playsinline preload="metadata" aria-label="Video de {nombre_l[c]}"></video></div><figcaption><b>{nombre_l[c]}, las 72 h completas</b> (una de cada 3 fotos). En celeste, el contorno de cada célula; en color, el camino de cada una en los últimos 100 minutos.</figcaption></figure>' for c in cams)}</div>
</section>""")

    # --- proceso
    sweep = barrido[barrido["barrido"] == "barrido 2"].copy()
    sweep["cobertura"] = sweep["cobertura"].astype(float)
    sweep["falsos"] = sweep["falsos"].astype(float)
    sw = sweep.groupby(["entrada", "reducir"])[["cobertura", "falsos", "n"]].mean().reset_index()
    nom_e = {"combinada": "Campo claro + DPC, sin aplanar", "comb_plana": "Campo claro aplanado + DPC",
             "comb_plana_tn": "Campo claro aplanado + DPC, normalización local"}
    sw_filas = [[nom_e.get(r.entrada, r.entrada), f"{int(r.reducir)} ({f(EV.UM_POR_PX * r.reducir, 2)} µm/px)",
                 f(r.n, 0), pct(r.cobertura), pct(r.falsos)] for r in sw.sort_values("cobertura", ascending=False).itertuples()]
    flow = barrido[barrido["barrido"] == "flow_threshold"].copy()
    fl = flow.groupby("ft")[["n", "cobertura", "falsos"]].mean().reset_index()
    fl_filas = [[{0.4: "0.4 (por defecto)", 0.8: "0.8 (elegido)", 0.0: "sin control"}.get(float(r.ft), r.ft),
                 f(r.n, 0), pct(r.cobertura), pct(r.falsos)] for r in fl.sort_values("ft", ascending=False).itertuples()]
    S.append(f"""<section id="proceso">{h2('proceso', 'Cómo se analizó')}
<ol class="pasos texto">
<li><span class="n">1</span><div><b>Preparar la imagen.</b> La foto de campo claro se divide por su propio fondo (suavizado de 150 píxeles) para quitar el viñeteado de la matriz de LEDs, que deja el centro mucho más brillante que los bordes. Se apila con los dos relieves DPC: la red recibe tres canales.</div></li>
<li><span class="n">2</span><div><b>Detectar las células</b> con Cellpose-SAM, a {f(EV.UM_POR_PX * red, 2)} µm por píxel (reducción {red}), con <span class="mono">flow_threshold = {ft:g}</span>. Después se descartan objetos que no parecen células: de menos de {EV.FORMA_AREA_MIN_UM2} µm², poco compactos o muy delgados.</div></li>
<li><span class="n">3</span><div><b>Seguir cada célula</b> de una foto a la siguiente (laptrack, la misma variante <span class="mono">dist_tam</span> validada en el reporte anterior: distancia con penalización por cambio de tamaño, hasta 25 µm entre fotos y huecos de hasta 3 fotos).</div></li>
<li><span class="n">4</span><div><b>Descartar la basura.</b> Las trayectorias cuya área mediana es menor que {f(propio.AREA_MIN_CELULA_UM2, 0)} µm² son manchas del medio, no células (ver la sección 5). Se descartaron {ambos('trayectorias_descartadas_restos', 0, {c: res0[c] for c in cams})} trayectorias.</div></li>
<li><span class="n">5</span><div><b>Medir los cúmulos.</b> En cada foto se cuenta cuántas células tiene el grupo en el que está cada una: las que se tocan y también la zona con células que se ve en el DPC aunque no esté segmentada.</div></li>
<li><span class="n">6</span><div><b>Calcular las medidas</b> con el mismo programa del reporte anterior, por caja (réplica) y por ventanas de 12 horas.</div></li>
</ol>
{figura(src['entradas'], '<b>La misma foto (caja 1, hora 1) segmentada con cada entrada que se probó.</b> En amarillo, los contornos. Con solo campo claro, la red no ve las células planas del centro; con el relieve DPC sí, pero sin aplanar el campo claro todavía pierde las láminas; la versión final las divide célula por célula.') if src.get('entradas') else ''}
<div class="texto"><h3>Cómo se eligió la configuración</h3>
<p>Sin contornos dibujados a mano, comparamos cada configuración contra la <b>zona con células</b> que se ve en el DPC (donde el relieve tiene textura): cobertura es qué parte de esa zona queda dentro de alguna célula detectada, y fuera de zona es qué parte de lo detectado cae sobre fondo vacío. Es una referencia aproximada (subestima los bordes planos de las láminas), útil para comparar configuraciones entre sí. Promedio de 12 fotos repartidas en las 72 h de las dos cajas:</p></div>
{tabla(sw_filas, ['Entrada', 'Reducción', 'Células', 'Cobertura', 'Fuera de zona'], [False, False, True, True, True])}
<div class="texto"><p>En las fotos más difíciles (día 2 y 3), Cellpose encontraba muchas células de las láminas pero las descartaba por su control de forma (<span class="mono">flow_threshold</span>). Aflojarlo a 0.8 las recupera; quitarlo del todo agrega rebanadas que no son células. Promedio de 4 fotos difíciles:</p></div>
{tabla(fl_filas, ['flow_threshold', 'Células', 'Cobertura', 'Fuera de zona'])}
</section>""")

    # --- validación
    S.append(f"""<section id="validacion">{h2('validacion', '¿Son confiables las mediciones?')}
<div class="aviso"><strong>Sin referencia independiente</strong><p>En el reporte anterior la segmentación se comprobó contra núcleos teñidos y dibujos hechos a mano. Aquí no hay ninguna de las dos cosas, así que las comprobaciones son indirectas. Las cifras de esta sección dicen cuánto se puede confiar, no cuánto acierta exactamente.</p></div>
{figura(src['conteo'], '<b>Células segmentadas en cada foto</b> (línea gruesa: mediana de 1 h). Entre fotos consecutivas el conteo cambia poco, señal de que la segmentación es estable. La caída del final no es que se pierdan células: son las láminas aplanadas que la segmentación ya no separa.' + (f' Discontinua: células en grupos de hasta {mg}.' if mg else ''))}
{figura(src['ejemplos'], '<b>Inicio, mitad y final de cada caja.</b> ' + (f'En verde, células sueltas o en grupos de hasta {mg} (las que entran en este reporte); en rojo, células en cúmulos más grandes.' if mg else 'En amarillo, los contornos detectados. Hacia el final las células se aplanan y se enciman, y parte de las láminas queda sin segmentar, sobre todo en la caja 2.'))}
{figura(src['contornos'], '<b>Contornos detectados, de cerca.</b> Recortes de unos 270 µm (ampliados 2×) en la zona con más células de cada caja, al inicio, a la mitad y al final; cada color es una célula. Las láminas quedan divididas una por una, con bordes que siguen las uniones entre células. Se escapan sobre todo las células grandes y aplanadas que están solas, algunas alargadas sobre el fondo y a veces una de dos células redondas pegadas (en división). Los bordes escalonados vienen de segmentar a un tercio de la resolución.')}
{figura(src['basura'], '<b>Objetos «sueltos» de la caja 2, por tamaño</b> (relieve DPC, 52 µm de lado). Fila de arriba: menos de 150 µm², casi todos manchas compactas y brillantes del medio, no células. Fila del medio: 150–250 µm², mezcla de manchas y células redondeadas. Fila de abajo: más de 250 µm², células. Por eso se descartan las trayectorias con área mediana menor que 150 µm².') if 'basura' in src else ''}
<div class="texto"><h3>Seguimiento</h3>
<p>Sin núcleos teñidos no podemos medir el porcentaje de uniones correctas como en el reporte anterior. Lo que sí se puede ver es el tamaño de cada paso: la mediana es {' y '.join(f(paso_med[c], 1) for c in cams)} µm entre fotos, y {' y '.join(pct(paso_20[c], 1) for c in cams)} de los pasos miden más de 20 µm, el tamaño de una célula. Esos saltos grandes son, en parte, intercambios entre vecinas en las láminas densas: afectan poco a las medianas pero inflan los promedios.</p></div>
{figura(src['pasos'], '<b>Cuánto se mueve una célula entre fotos consecutivas</b> (escala logarítmica). La línea punteada marca 20 µm, más o menos el diámetro de una célula.')}
</section>""")

    # --- persistencia (breve, remite al reporte anterior)
    S.append(f"""<section id="persistencia" class="texto">{h2('persistencia', 'Cómo se mide la persistencia')}
<p>Las medidas son las mismas del reporte anterior, donde están explicadas en detalle y con un simulador. En resumen:</p>
<ul>
<li><b>MSD y α.</b> Cuánto se aleja una célula de donde estaba según el tiempo. α = 1 es una caminata al azar; entre 1 y 2, movimiento persistente; por debajo de 1, confinado o frenado.</li>
<li><b>Caminata persistente (PRW).</b> Modelo con rapidez, tiempo de persistencia P y error de posición σ (cuánto «tiembla» el centro de la célula entre fotos aunque no se mueva).</li>
<li><b>Autocorrelación de la velocidad (VACF).</b> Cuánto se parece la dirección y rapidez de un paso a la del siguiente.</li>
<li><b>EAD y SE</b> (entropías de Shannon de Liu et al.). Valen 1 si el movimiento no tiene dirección preferida y menos de 1 si hay persistencia; corregidas por la cantidad de datos de cada célula.</li>
</ul>
<div class="aviso"><strong>Qué no se puede medir aquí</strong><p>Con una foto cada 5 minutos y un error de posición de ~2 µm, el reporte anterior mostró con simulaciones que la persistencia de unos pocos minutos queda tapada. Aquí el ajuste de caminata persistente da un tiempo de persistencia menor que el intervalo entre fotos en las dos cajas: no se reporta como medido.</p></div>
</section>""")

    # --- resultados
    filas = []
    def fila(nom, col, d=2, porc=False, fuente=P):
        vals = [fuente[c][col] for c in cams]
        fmt = (lambda x: pct(x, 0)) if porc else (lambda x: f(x, d))
        return [nom, *[fmt(v) for v in vals], fmt(np.mean(vals)), pct(dif(col, fuente), 0)]
    for args_f in (("Trayectorias de al menos 1 h", "n_segmentos", 0), ("Densidad (células/mm²)", "densidad_cel_mm2", 0),
                   ("Rapidez media de la célula (µm/min)", "rapidez_um_min", 3), ("Direccionalidad en 1 h", "direccionalidad_1h", 3),
                   ("α a escala corta (5–30 min)", "alfa_corto", 3), ("α a escala larga (1–4 h)", "alfa_largo", 3),
                   ("α de cada célula (mediana)", "alfa_celula_mediana", 3), ("Células superdifusivas (α > 1.2)", "frac_celulas_superdifusivas", 0, True),
                   ("Células subdifusivas (α < 0.8)", "frac_celulas_subdifusivas", 0, True),
                   ("Error de posición σ (µm)", "prw_sigma_um", 2), ("Autocorrelación de la velocidad a 5 min", "vacf_lag1", 3),
                   ("Coseno medio del giro", "cos_giro_medio", 3), ("EAD₁ del conjunto (corregida)", "ead1_ens_corr", 3),
                   ("EAD₁ por célula (corregida)", "ead1_cel_corr", 3), ("SE corregida", "se_fps_completa_corr", 3),
                   ("SE en el tiempo (corregida)", "se_wavelet_corr", 3), ("Alineamiento con vecinas (0–50 µm)", "corr_dir_0_50um", 3),
                   ("Alineamiento a 100–200 µm", "corr_dir_100_200um", 3), ("Área (µm²)", "area_um2", 0),
                   ("Alargamiento (largo/ancho)", "aspect_ratio", 2), ("Circularidad", "circularidad", 2), ("Solidez", "solidez", 2)):
        filas.append(fila(*args_f))
    S.append(f"""<section id="resultados">{h2('resultados', f'Resultados por caja: {html.escape(tit_var.lower())}')}
<div class="texto"><p>Cada número se calcula dentro de cada caja con el mismo programa del reporte anterior. Con dos réplicas no hay intervalos de confianza útiles: la última columna dice cuánto difieren las dos cajas, en porcentaje de su promedio.</p></div>
{tabla(filas, ['Medida', *[nombre[c] for c in cams], 'Promedio', 'Diferencia entre cajas'])}
<h3>Cómo se alejan las células de su punto de partida</h3>
<div class="texto"><p>En las dos cajas el MSD crece más despacio que en una caminata al azar a todas las escalas: α {ambos('alfa_corto')} entre 5 y 30 min, y {ambos('alfa_largo')} entre 1 y 4 h. {'En 5 horas una célula se aleja en promedio unos ' + f(np.sqrt(np.mean(list(msd5h.values()))), 0) + ' µm.' if np.isfinite(np.mean(list(msd5h.values()))) else ''} Célula por célula, la mediana de α es {ambos('alfa_celula_mediana')}; {ambos('frac_celulas_subdifusivas', 0, {c: {k: 100 * v if k == 'frac_celulas_subdifusivas' else v for k, v in P[c].items()} for c in cams}, '%')} de las células son subdifusivas.</p></div>
{figura(src['msd'], '<b>MSD de cada caja</b> (escala log-log). Puntos: datos; línea de color: modelo de caminata persistente con error de posición; punteadas: ajuste de α a escala corta y larga; gris: pendiente 1 (caminata al azar).')}
<h3>¿Cuánto es eso en el campo?</h3>
<div class="texto"><p>El campo mide {f(campo_w, 0)} × {f(campo_h, 0)} µm. En 1 hora una célula se aleja en promedio {f(rad1, 0)} µm de donde estaba y en 5 horas {f(rad5, 0)} µm: el {f(100 * rad5 / campo_w, 1)}% del ancho del campo, poco más que el diámetro de una célula (~{f(diam, 0)} µm). Paso a paso recorre más (unos {f(np.mean(list(recorrido_5h.values())), 0)} µm en 5 h), pero va y vuelve. Si siguiera igual (α {f(a_l)}), en 24 h se alejaría ~{f(rad24, 0)} µm y en todo el experimento ~{f(radtot, 0)} µm; para alejarse medio campo tardaría unos {f(t_mitad / 24, 0)} días.</p></div>
{figura(src['escala'], '<b>El campo a escala.</b> En verde, una célula de tamaño típico; los círculos muestran cuánto se aleja en promedio en 1 h y 5 h (medido, línea continua) y en 24 h y todo el experimento (extrapolado con α del ensamble, línea discontinua).')}
{figura(src['alfa'], '<b>α por caja, por célula y a cada escala.</b> Izquierda: α corto y largo de cada caja. Centro: α de cada célula (caja: cuartiles). Derecha: α local según τ; la línea punteada es α = 1.')}
<h3>Persistencia</h3>
<div class="texto"><p>La autocorrelación de la velocidad a un paso es levemente negativa ({ambos('vacf_lag1')}): un paso tiende a ir un poco en contra del anterior, lo típico de células que «tiemblan» en su lugar o de error de posición. El coseno medio del giro es casi 0 y la EAD₁ corregida queda en {ambos('ead1_ens_corr', 3)} con pasos de 5 min; con pasos de 10–20 min sube a 0.99, es decir, sin dirección preferida a ninguna escala que permitan estas fotos.</p></div>
{figura(src['vacf_ead'], '<b>Izquierda:</b> autocorrelación de la velocidad. <b>Centro:</b> EAD corregida según el intervalo, del conjunto de células y promedio por célula. <b>Derecha:</b> EAD₁ del conjunto según el paso de análisis.')}
{figura(src['distrib'], '<b>Distribución por célula</b> de la rapidez media y de la direccionalidad en 1 h (desplazamiento neto en una hora dividido por lo recorrido). Línea punteada: mediana.')}
</section>""")

    # --- tiempo
    if ven_ok:
        filas_v = []
        for c in cams:
            for r in ven[ven["camara"] == c].sort_values("t0_h").itertuples():
                filas_v.append([f"{nombre[c]} · {f(r.t0_h, 0)}–{f(r.t1_h, 0)} h", f(r.n_segmentos, 0), f(r.densidad_cel_mm2, 0),
                                f(r.rapidez_um_min, 2), f(r.alfa_corto, 2), f(r.alfa_largo, 2), f(r.ead1_ens_corr, 3),
                                f(r.area_um2, 0), f(r.circularidad, 2)])
        tabla_ventanas = tabla(filas_v, ["Caja · ventana", "Trayect.", "Células/mm²", "Rapidez", "α corto", "α largo",
                                         "EAD₁ ens.", "Área µm²", "Circ."])
        S.append(f"""<section id="tiempo">{h2('tiempo', 'Cambios a lo largo de las 72 h')}
<div class="texto">{texto_ventanas}</div>
{figura(src['ventanas'], '<b>Cada medida en ventanas de 12 h</b>, por caja.')}
</section>""")
    else:
        tabla_ventanas = ""

    # --- vecinas
    S.append(f"""<section id="vecinas">{h2('vecinas', 'Movimiento en bloque')}
<div class="texto"><p>Para cada par de células en la misma foto medimos qué tan parecidas son sus direcciones (1 = misma dirección, 0 = sin relación). Las vecinas cercanas se mueven parecido ({ambos('corr_dir_0_50um')} a menos de 50 µm), pero la coordinación casi no decae con la distancia ({ambos('corr_dir_100_200um')} a 100–200 µm). Eso tiene dos explicaciones posibles, que estos datos no separan: islas enteras de células que se desplazan juntas, o una pequeña deriva común de todo el campo (de la platina o del medio). Sumando el paso promedio de todas las células en cada foto, el campo entero se desplazó {' y '.join(f(deriva[c], 0) for c in cams)} µm en todo el experimento: poco frente a lo que recorre cada célula, pero suficiente para sumar algo de coordinación aparente.</p></div>
{figura(src['corr'], '<b>Alineamiento de direcciones según la distancia</b> entre células, por caja.')}
</section>""")

    # --- forma
    rho_filas = [[n, *[f(rho.get((c, k)), 2) for c in cams]] for k, n in
                 (("area_um2", "Área"), ("alargamiento", "Alargamiento"), ("q", "Índice de forma q"))]
    S.append(f"""<section id="forma">{h2('forma', 'Forma de las células')}
<div class="texto"><p>Medimos la forma de cada célula en cada foto a partir de su contorno, sin las que tocan el borde de la imagen (salen cortadas). Además del área, la circularidad y el alargamiento, calculamos el <b>índice de forma q</b> (perímetro / √área), el mismo del reporte anterior: 3.54 para un círculo; en tejidos epiteliales, por encima de 3.81 las células suelen pasar de un estado compacto y quieto a uno fluido que migra.</p></div>
{figura(src['forma'], '<b>Distribución de cada medida de forma</b>, por caja. Línea punteada de color: mediana; línea negra: q = 3.81.')}
<div class="texto"><p>En el reporte anterior las MDA-MB-231 más alargadas se movían más rápido. Aquí el área casi no se relaciona con la rapidez y el alargamiento solo un poco, pero el índice de forma sí: {q_txt}. Una célula con contorno irregular y extendido es una célula que está emitiendo prolongaciones, y eso coincide con moverse. Hay que tomarlo con una precaución: un contorno mal segmentado también da un q alto y puede «moverse» por error de posición.</p></div>
{figura(src['forma_rapidez'], '<b>Rapidez según la forma de la célula en esa misma foto</b>: las detecciones se agrupan en deciles de cada medida; mediana y cuartiles de la rapidez en cada decil.')}
<h4>Correlación de Spearman entre forma y rapidez (cada detección)</h4>
{tabla(rho_filas, ['Forma', *[f'ρ {nombre[c]}' for c in cams]])}
<p class="texto" style="font-size:15px;color:var(--ink-2)">|ρ| menor que 0.1: sin relación; 0.1–0.3: débil. Sin valor p: las detecciones de una misma caja no son independientes.</p>
</section>""")

    # --- cúmulos
    if mg:
        comp = []
        for nom_c, col, d in (("Trayectorias de al menos 1 h", "n_segmentos", 0), ("Rapidez media de la célula (µm/min)", "rapidez_um_min", 2),
                              ("Direccionalidad en 1 h", "direccionalidad_1h", 2), ("α corto", "alfa_corto", 2), ("α largo", "alfa_largo", 2),
                              ("EAD₁ del conjunto (corregida)", "ead1_ens_corr", 3), ("Área (µm²)", "area_um2", 0),
                              ("Alargamiento", "aspect_ratio", 2), ("Circularidad", "circularidad", 2)):
            comp.append([nom_c, *[f"{f(P0[c][col], d)} → <b>{f(P[c][col], d)}</b>" for c in cams]])
        comp_html = (f"<h3>Comparado con todas las células</h3>{tabla(comp, ['Medida', *[f'{nombre[c]}: todas → grupos ≤ {mg}' for c in cams]])}"
                     f"<p class=\"texto\" style=\"font-size:15px;color:var(--ink-2)\">Con tan pocas trayectorias, las cifras de la derecha son mucho menos estables que las de todas las células.</p>")
    else:
        comp_html = ""
    S.append(f"""<section id="cumulos">{h2('cumulos', 'Células sueltas y cúmulos')}
<div class="texto"><p>En cada foto contamos cuántas células tiene el grupo en el que está cada una. Casi todas viven en cúmulos: solo el {' y '.join(pct(frac1[c], 1) for c in cams)} de las detecciones está completamente sola, y el {' y '.join(pct(frac3[c], 1) for c in cams)} en grupos de hasta 3. Las células sueltas son más chicas y redondas y se mueven menos paso a paso que las que están en láminas.</p></div>
{figura(src['grupos'], '<b>Rapidez instantánea, área y circularidad según el tamaño del grupo</b> en el que está la célula en esa foto (punto: mediana; barra: cuartiles).')}
{comp_html}
</section>""")

    # --- comparación con el reporte anterior
    S.append(f"""<section id="comparacion">{h2('comparacion', 'Comparación con el reporte anterior')}
<div class="texto"><p>Como referencia, las MDA-MB-231 del reporte anterior (videos públicos en plástico, campo claro, una foto cada 5 minutos, 16 videos):</p></div>
{tabla([["Rapidez media (µm/min)", "0.77", f(prom('rapidez_um_min'))], ["Persistencia P (min)", "15", "no medible (&lt; 5)"],
        ["α a escala corta", "1.22 (superdifusivo)", f(prom('alfa_corto')) + (' (subdifusivo)' if prom('alfa_corto') < 0.95 else '')],
        ["α a escala larga", "1.15", f(prom('alfa_largo'))], ["EAD₁ del conjunto (corregida)", "0.980", f(prom('ead1_ens_corr'), 3)],
        ["Error de posición σ (µm)", "1.97", f(prom('prw_sigma_um'))], ["Área (µm²)", "436", f(prom('area_um2'), 0)],
        ["Alargamiento", "2.02", f(prom('aspect_ratio'))]],
       ["Medida", "MDA-MB-231 (reporte anterior)", f"{html.escape(linea_txt)} (promedio de las dos cajas)"], [False, True, True])}
<div class="texto"><p>Son células, microscopios y densidades distintas, así que la comparación es solo orientativa. Aun así la diferencia es clara: las MDA-MB-231 sueltas recorren distancias mayores y mantienen la dirección unos 15 minutos, mientras que aquí las células, casi todas en cúmulos, son menos alargadas y se mueven de forma confinada.</p></div>
</section>""")

    # --- limitaciones, próximos pasos
    S.append(f"""<section id="limitaciones" class="texto">{h2('limitaciones', 'Limitaciones')}
<ul>
<li><b>Dos réplicas.</b> Dos cajas alcanzan para ver si un resultado se repite, no para estimar su variabilidad ni para hacer pruebas estadísticas. Para comparar condiciones hacen falta al menos 3–5 cajas por condición.</li>
<li><b>Sin referencia para la segmentación.</b> No hay núcleos teñidos ni contornos dibujados a mano; la configuración se eligió con una referencia aproximada y a ojo.</li>
<li><b>Día 3.</b> Las láminas aplanadas se segmentan mal, sobre todo en la caja 2: el conteo y la densidad de las últimas horas están subestimados.</li>
<li><b>Intercambios en el seguimiento.</b> En las láminas densas el tracking puede intercambiar células vecinas; sin núcleos no se puede medir cuánto.</li>
<li><b>Muestreo de 5 minutos y error de posición de ~2 µm.</b> No permiten medir una persistencia de pocos minutos.</li>
<li><b>Deriva común.</b> Parte del movimiento coordinado puede ser deriva de la platina o del medio, no de las células; no se corrigió.</li>
</ul></section>""")
    S.append(f"""<section id="siguiente" class="texto">{h2('siguiente', 'Próximos pasos')}
<ul>
<li>Dibujar a mano los contornos de 5–10 fotos de los días 2 y 3 y entrenar Cellpose con ellas: es lo que más mejoraría la segmentación de las láminas aplanadas.</li>
<li>Para medir persistencia, filmar cada 1–2 minutos al menos en algunas cajas.</li>
<li>Agregar un marcador nuclear (o fotos de fluorescencia ocasionales) para validar segmentación y seguimiento como en el reporte anterior.</li>
<li>Para estudiar migración individual, sembrar a menor densidad: aquí casi no hay células sueltas.</li>
<li>Corregir la deriva común del campo (con marcas fijas o restando el movimiento medio) antes de interpretar el movimiento en bloque.</li>
<li>Tres a cinco cajas por condición para poder comparar tratamientos con estadística.</li>
</ul></section>""")

    # --- glosario
    glos = [("DPC", "Contraste de fase diferencial: se ilumina por turnos con mitades opuestas de la matriz de LEDs y se restan las fotos; resalta el relieve de las células."),
            ("Campo claro aplanado", "La foto de campo claro dividida por su propio fondo suavizado, para quitar el viñeteado de la iluminación."),
            ("Cellpose-SAM", "Modelo de aprendizaje profundo que dibuja el contorno de cada célula."),
            ("flow_threshold", "Control de calidad de Cellpose: descarta contornos cuya forma no concuerda con lo que predijo la red. Más alto = menos estricto."),
            ("Tracking", "Seguir a cada célula de una foto a la siguiente."),
            ("Cúmulo / grupo", "Conjunto de células que se tocan en una foto, incluida la zona con células que se ve en el DPC aunque no esté segmentada."),
            ("Réplica", "Unidad independiente para la estadística: aquí, cada caja de Petri."),
            ("MSD", "Desplazamiento cuadrático medio: cuánto se aleja, en promedio, una célula de donde estaba después de un tiempo τ."),
            ("α", "Cómo crece el MSD con el tiempo: 1 = caminata al azar, 2 = línea recta, menos de 1 = confinado."),
            ("Subdifusivo / superdifusivo", "Una célula con α menor que 0.8 / mayor que 1.2."),
            ("PRW, P y σ", "Caminata aleatoria persistente: P es cuánto tiempo mantiene su dirección; σ cuánto tiembla la posición medida."),
            ("VACF", "Autocorrelación de la velocidad: cuánto se parece un paso al anterior. Negativa: tiende a revertir."),
            ("EAD, SE", "Entropías de Shannon de los ángulos de giro y del espectro de la velocidad (Liu et al. 2021, 2024). Corregidas valen 1 para movimiento al azar."),
            ("Direccionalidad", "Desplazamiento neto dividido por lo recorrido en una hora: 1 = línea recta."),
            ("Circularidad", "4π × área / perímetro²: 1 para un círculo."),
            ("Índice de forma q", "Perímetro dividido por la raíz del área: 3.54 para un círculo; crece cuanto más alargada o irregular es la célula."),
            ("Viñeteado", "Iluminación más fuerte en el centro de la imagen que en los bordes.")]
    S.append(f'<section id="glosario">{h2("glosario", "Glosario")}<dl class="glosario">'
             + "".join(f"<dt>{a}</dt><dd>{b}</dd>" for a, b in glos) + "</dl></section>")

    # --- anexos
    cols_anexo = [("n_segmentos", "Trayect.", 0), ("densidad_cel_mm2", "Densidad /mm²", 0), ("rapidez_um_min", "Rapidez", 3),
                  ("direccionalidad_1h", "Direcc. 1 h", 3), ("alfa_corto", "α corto", 3), ("alfa_largo", "α largo", 3),
                  ("alfa_celula_mediana", "α célula", 3), ("prw_P_min", "P min", 2), ("prw_S_um_min", "S µm/min", 2),
                  ("prw_sigma_um", "σ µm", 2), ("prw_r2", "R² PRW", 3), ("vacf_lag1", "VACF(1)", 3), ("cos_giro_medio", "⟨cos giro⟩", 3),
                  ("rho_rapidez_cosgiro", "ρ rap.–giro", 3), ("ead1_ens_corr", "EAD₁ ens.", 3), ("ead1_cel_corr", "EAD₁ cél.", 3),
                  ("tl1_ens_min", "TL₁ min", 1), ("se_fps_completa_corr", "SE", 3), ("se_wavelet_corr", "SE(t)", 3),
                  ("corr_dir_0_50um", "Alin. 0–50", 3), ("area_um2", "Área", 0), ("aspect_ratio", "Alarg.", 2), ("circularidad", "Circ.", 2)]
    filas_a = [[nombre_l[c], *[f(P[c][k], d) for k, _, d in cols_anexo]] for c in cams]
    S.append(f"""<section id="replicas">{h2('replicas', 'Anexo: cifras por caja y por ventana')}
<div class="texto"><p>Todas las medidas de cada caja. P, S y σ son la persistencia, la rapidez y el error de posición del modelo de caminata persistente (P no es interpretable: queda por debajo del intervalo entre fotos); ρ rap.–giro es el acoplamiento entre rapidez y persistencia; «Alin.» es el alineamiento de direcciones entre vecinas.</p></div>
<details open><summary><b>Por caja</b></summary>{tabla(filas_a, ['Caja', *[n for _, n, _ in cols_anexo]])}</details>
{f'<details><summary><b>Por ventana de 12 h</b></summary>{tabla_ventanas}</details>' if tabla_ventanas else ''}
</section>""")
    S.append(f"""<section id="tecnico">{h2('tecnico', 'Anexo técnico')}
<div class="texto"><p>El análisis se regenera con estos pasos (rama <span class="mono">experimental/interfaz-cuda</span> del pipeline):</p>
<ol>
<li>Segmentación: la interfaz (<span class="mono">33_interfaz.py</span>) o <span class="mono">32_segmentar_en_vivo.py</span>, con entrada <span class="mono">combinada</span>, reducción {red}, <span class="mono">flow_threshold</span> {ft:g} y filtro de forma (valores por defecto desde el 5 de octubre de 2026).</li>
<li>Tracking, cúmulos y basura: <span class="mono">lib/propio.analizar</span> (botón «Analizar trayectorias» de la interfaz; con el selector de células para grupos chicos).</li>
<li>Estadística por caja y por ventana: <span class="mono">36_estadisticas_propio.py {html.escape(exp)}{f' --max-grupo {mg}' if mg else ''}</span>.</li>
<li>Este reporte: <span class="mono">37_reporte_propio.py {html.escape(exp)} --linea {html.escape(linea)}{f' --max-grupo {mg}' if mg else ''}</span>.</li>
</ol>
<h3>Archivos</h3><ul>
<li><span class="mono">~/microscopio_cache/masks/propio/{html.escape(exp)}/</span>: máscaras de cada foto (con los parámetros usados).</li>
<li><span class="mono">~/microscopio_cache/analisis/propio/{html.escape(exp)}/&lt;cam&gt;/tracks.csv</span>: cada detección con su trayectoria y el tamaño de su grupo; <span class="mono">por_celula.csv</span> y <span class="mono">resumen.json</span>.</li>
<li><span class="mono">…/estadisticas/</span>: <span class="mono">por_pelicula.csv</span> (una fila por caja), <span class="mono">por_ventana.csv</span>, <span class="mono">por_celula.csv</span>, MSD, VACF, EAD, alineamiento.</li>
<li><span class="mono">…/ajuste_segmentacion/barrido.csv</span>: la comparación de entradas y parámetros.</li>
</ul></div></section>""")

    gen = datetime.datetime.now()
    nav = "".join(f'<li><a href="#{k}">{t}</a></li>' for k, t in indice)
    titulo = f"{linea + ' en ' if linea else ''}{html.escape(meta.get('nombre', exp)).capitalize()}" + (" · sueltas" if mg else "")
    h1 = (f"Cómo se mueven las células {linea} en el microscopio propio" if not mg else
          f"Células {linea} sueltas o en grupos chicos") if linea else ("Migración y forma en la " + html.escape(meta.get("nombre", exp)))
    lede = (f"Seguimos miles de células durante {f((fin - ini).total_seconds() / 3600, 0)} horas en dos cajas de Petri, con iluminación DPC, y medimos su rapidez, cómo se alejan de su punto de partida, si mantienen la dirección, si se mueven en grupo y cómo es su forma. Este reporte explica cómo se hizo, cuánto se puede confiar en cada número y qué encontramos."
            if not mg else f"La misma {html.escape(meta.get('nombre', exp))}, pero contando solo los momentos en que cada célula está sola o en un grupo de hasta {mg} células, para separar el movimiento individual del de las láminas.")
    css = PLANTILLA_CSS.read_text(encoding="utf-8")
    doc = f"""<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{titulo}</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600;7..72,700&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
{css}
[hidden] {{ display: none !important; }}
</style>
<div class="envoltorio">
<nav class="indice" aria-label="Índice"><p>Contenido</p><ol>{nav}</ol></nav>
<main>
<header class="portada">
<p class="ceja">Reporte · {html.escape(meta.get('nombre', exp))} · {html.escape(tit_var)} · {fecha_larga(gen, hora=False)}</p>
<h1>{h1}</h1>
<p class="lede">{lede}</p>
<div class="explica" style="max-width:none"><strong>Objetivo de este reporte</strong><p>(1) Contar cómo se analizaron las fotos DPC del microscopio propio y cuánto se puede confiar en el resultado; (2) mostrar, caja por caja, cómo se mueven y qué forma tienen las células; y (3) señalar qué falta para usar estos datos en comparaciones entre condiciones.</p></div>
</header>
{''.join(S)}
<footer>Generado automáticamente a partir de <span class="mono">~/microscopio_cache/analisis/propio/{html.escape(exp)}/</span> por <span class="mono">scripts/37_reporte_propio.py</span> · {fecha_larga(gen, hora=False)}. Referencias: Liu Y. et al., Biophys J 120:2552 (2021) · Liu Y. et al., Biophys J 123:730 (2024) · Pachitariu M., Rariden M. y Stringer C., Cellpose-SAM, bioRxiv (2025) · Tian L. y Waller L., Opt Express 23:11394 (2015) · Bi D. et al., Nat Phys 11:1074 (2015).</footer>
</main></div>
<div id="visor" hidden><img alt=""></div>
<script>
(() => {{
  const visor = document.getElementById("visor");
  const vimg = visor.querySelector("img");
  document.querySelectorAll("figure img").forEach(img => {{
    img.addEventListener("click", () => {{ vimg.src = img.src; vimg.alt = img.alt; visor.hidden = false; }});
  }});
  visor.addEventListener("click", () => {{ visor.hidden = true; }});
  document.addEventListener("keydown", e => {{ if (e.key === "Escape") visor.hidden = true; }});
}})();
</script>"""
    (out / "index.html").write_text(doc, encoding="utf-8")
    tam = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"listo: {out}/index.html ({tam:.1f} MB en total)")


if __name__ == "__main__":
    main()
