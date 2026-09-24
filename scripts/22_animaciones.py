"""
Animaciones MP4 livianas para el reporte HTML (H.264, ~1-4 MB cada una).

  pipeline_bf.mp4        recorte de una película BF en 4 paneles: imagen
                         original | contornos de Cellpose | núcleos SiR-DNA |
                         trayectorias. Muestra el pipeline paso a paso.
  bf_campo_denso.mp4     campo completo de la película más densa (15) con
                         las trayectorias.
  camad_expNN.mp4        un experimento por sustrato de CAMAD con contornos
                         y trayectorias (1 de cada 4 frames = 2 min por frame).

Se dibuja con OpenCV (rápido) y los títulos con Pillow (soporta acentos).
Cada video se genera en un proceso aparte.

Uso: ../venv/bin/python 22_animaciones.py
"""
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import CAMAD_CONDICION, DATASETS, RES, res_dir  # noqa: E402
from lib.fuentes import listar_frames, leer_frame, cargar_mascara  # noqa: E402

OUT = res_dir("reporte", "videos")
# colores BGR (OpenCV): paleta del reporte
TRAY = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232)]
CONTORNO = (80, 220, 255)
NUCLEO = (255, 170, 60)
FUENTE = "/usr/share/fonts/TTF/DejaVuSans.ttf"


def variante(ds):
    p = RES / f"variante_{ds}.txt"
    return p.read_text().strip() if p.exists() else "dist_tam"


def a_gris8(img, lo=None, hi=None):
    img = img.astype(np.float32)
    if lo is None:
        lo, hi = np.percentile(img, (1, 99.7))
    return np.clip((img - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def titulo(texto, ancho, alto=34, tam=18):
    im = Image.new("RGB", (ancho, alto), (24, 26, 30))
    d = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype(FUENTE, tam)
    except OSError:
        f = ImageFont.load_default()
    d.text((10, (alto - tam) // 2 - 1), texto, fill=(235, 235, 235), font=f)
    return np.array(im)[:, :, ::-1]


def dibujar_contornos(bgr, masks, color, grosor=1):
    b = np.zeros(masks.shape, np.uint8)
    # borde = píxel cuyo vecino tiene otra etiqueta
    m = masks.astype(np.int32)
    borde = np.zeros_like(b, bool)
    borde[:-1, :] |= m[:-1, :] != m[1:, :]
    borde[:, :-1] |= m[:, :-1] != m[:, 1:]
    borde &= m > 0
    if grosor > 1:
        borde = cv2.dilate(borde.astype(np.uint8), np.ones((grosor, grosor), np.uint8)) > 0
    bgr[borde] = color
    return bgr


def dibujar_tray(bgr, tr, frame, escala=1.0, offset=(0, 0), cola=15, grosor=2):
    ox, oy = offset
    g = tr[(tr.frame <= frame) & (tr.frame > frame - cola)]
    for tid, t in g.groupby("track_id"):
        t = t.sort_values("frame")
        pts = np.column_stack([(t.x - ox) * escala, (t.y - oy) * escala]).astype(np.int32)
        col = TRAY[int(tid) % len(TRAY)]
        if len(pts) > 1:
            cv2.polylines(bgr, [pts], False, col, grosor, cv2.LINE_AA)
        if t.frame.iloc[-1] == frame:
            cv2.circle(bgr, tuple(pts[-1]), grosor + 1, col, -1, cv2.LINE_AA)
    return bgr


def escribir_mp4(frames, ruta, fps):
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w2}x{h2}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "28",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ruta)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f[:h2, :w2]).tobytes())
    p.stdin.close()
    p.wait()
    # póster (primer frame con trayectorias ya visibles) para mostrar sin reproducir
    cv2.imwrite(str(ruta.with_suffix(".jpg")), frames[min(len(frames) - 1, len(frames) // 2)],
                [cv2.IMWRITE_JPEG_QUALITY, 85])


def video_pipeline_bf(movie=1):
    tr = pd.read_csv(res_dir("bf", "tracking", variante("bf")) / "tracks.csv.gz",
                     usecols=["movie", "track_id", "frame", "x", "y"])
    tr = tr[tr.movie == movie]
    items_bf = [i for i in listar_frames("bf") if i["movie"] == movie]
    items_nu = [i for i in listar_frames("sirdna") if i["movie"] == movie]
    # recorte de 400x400 px centrado en la zona con más trayectorias largas
    y0, x0, L = 300, 300, 400
    esc = 0.9
    S = int(L * esc)
    cab = np.hstack([titulo(t, S) for t in ("1. Imagen original (brightfield)", "2. Células detectadas (Cellpose)",
                                            "3. Núcleos (fluorescencia SiR-DNA)", "4. Trayectorias (tracking)")])
    lo_b, hi_b = np.percentile(leer_frame(items_bf[0]), (1, 99.7))
    lo_n, hi_n = np.percentile(leer_frame(items_nu[0]), (1, 99.9))
    frames = []
    for ib, inu in zip(items_bf, items_nu):
        f = ib["frame"]
        b = a_gris8(leer_frame(ib), lo_b, hi_b)[y0:y0 + L, x0:x0 + L]
        n = a_gris8(leer_frame(inu), lo_n, hi_n)[y0:y0 + L, x0:x0 + L]
        mb = cargar_mascara("bf", movie, f)[y0:y0 + L, x0:x0 + L]
        mn = cargar_mascara("sirdna", movie, f)[y0:y0 + L, x0:x0 + L]
        p1 = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        p2 = dibujar_contornos(p1.copy(), mb, CONTORNO, 2)
        p3 = dibujar_contornos(cv2.applyColorMap(n, cv2.COLORMAP_BONE), mn, NUCLEO, 2)
        p4 = (p1 * 0.75).astype(np.uint8)
        paneles = [cv2.resize(p, (S, S), interpolation=cv2.INTER_AREA) for p in (p1, p2, p3, p4)]
        paneles[3] = dibujar_tray(paneles[3], tr, f, escala=esc, offset=(x0, y0), cola=20)
        fila = np.hstack(paneles)
        reloj = titulo(f"Película {movie} · t = {f * DATASETS['bf']['dt_s'] / 3600:.1f} h  "
                       f"(1 imagen cada 5 min; recorte de {L * DATASETS['bf']['um_per_px']:.0f} µm)", fila.shape[1], 28, 15)
        frames.append(np.vstack([cab, fila, reloj]))
    escribir_mp4(frames, OUT / "pipeline_bf.mp4", 10)
    return "pipeline_bf.mp4"


def video_campo(ds, movie, nombre, cada=1, fps=10, ancho=760, cola=12, texto=""):
    tr = pd.read_csv(res_dir(ds, "tracking", variante(ds)) / "tracks.csv.gz",
                     usecols=["movie", "track_id", "frame", "x", "y"])
    tr = tr[tr.movie == movie]
    items = [i for i in listar_frames(ds) if i["movie"] == movie][::cada]
    lo, hi = np.percentile(leer_frame(items[0]), (1, 99.7))
    frames = []
    dt = DATASETS[ds]["dt_s"]
    for it in items:
        f = it["frame"]
        g = cv2.cvtColor(a_gris8(leer_frame(it), lo, hi), cv2.COLOR_GRAY2BGR)
        m = cargar_mascara(ds, movie, f)
        g = dibujar_contornos(g, m, CONTORNO)
        esc = ancho / g.shape[1]
        g = cv2.resize(g, (ancho, int(g.shape[0] * esc)), interpolation=cv2.INTER_AREA)
        g = dibujar_tray(g, tr, f, escala=esc, cola=cola * cada, grosor=2)
        cab = titulo(f"{texto} · t = {f * dt / 3600:.2f} h", ancho, 30, 15)
        frames.append(np.vstack([cab, g]))
    escribir_mp4(frames, OUT / nombre, fps)
    return nombre


def tarea(args):
    tipo = args[0]
    try:
        if tipo == "pipeline":
            return video_pipeline_bf(1)
        if tipo == "campo":
            return video_campo(*args[1:])
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"FALLA {args}: {e!r}"


def main():
    partes = sys.argv[1:] or ["bf", "camad"]
    tareas = []
    if "bf" in partes:
        tareas += [("pipeline",),
                   ("campo", "bf", 15, "bf_campo_denso.mp4", 1, 10, 760, 12,
                    "Película 15 (la más densa), campo completo")]
    if "camad" in partes:
        elegidos = {"Vidrio": 14, "Matrigel": 1, "Colágeno I": 12, "Matriz RAW dispersa": 4,
                    "Matriz RAW confluente": 7, "Matriz 231 (exp8-9)": 9}
        for cond, e in elegidos.items():
            tareas.append(("campo", "camad", e, f"camad_exp{e:02d}.mp4", 4, 15, 760, 10,
                           f"CAMAD exp{e} · {cond}"))
    with ProcessPoolExecutor(8) as ex:
        for r in ex.map(tarea, tareas):
            print(r, flush=True)


if __name__ == "__main__":
    main()
