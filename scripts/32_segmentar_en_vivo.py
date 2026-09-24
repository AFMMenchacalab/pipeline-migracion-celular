"""
Segmentación en vivo de las imágenes del microscopio propio (corre en la PC).

Vigila la carpeta donde 31_receptor_microscopio.py deja lo que manda la
Raspberry Pi y segmenta cada ciclo en cuanto está completo. El modelo se
carga una sola vez en la GPU; después cada imagen tarda unos segundos, muy
por debajo del intervalo entre fotos (2–5 min).

Qué es "un ciclo completo": en modo DPC la Pi guarda cuatro fotos por
tiempo (img_<fecha>_L/_R/_T/_B.tif). Se espera a tener las cuatro; el modo
y sus sufijos se leen de experimento.json, que la Pi manda al empezar.

Qué imagen recibe Cellpose (--entrada). Todavía NO está decidido cuál
segmenta mejor las imágenes DPC; se decide en el piloto comparando contra
contornos dibujados a mano. Opciones:
  suma       (L+R+T+B)/4: equivale a campo claro. Por defecto, porque es la
             modalidad en la que el pipeline está validado.
  dpc        magnitud del DPC de los dos ejes, sqrt(DPC_LR² + DPC_TB²):
             resalta bordes en todas las direcciones.
  combinada  tres canales [suma, DPC_LR, DPC_TB]; Cellpose-SAM acepta
             varios canales.
(La fase cuantitativa se agregará como opción cuando esté implementada.)

Salidas, reanudables (si se corta, al relanzarlo sigue donde quedó):
  ~/microscopio_cache/masks/propio/<experimento>/cam<N>/<fecha>.npz
      masks (uint16), um_por_px, entrada, reducir
  ~/microscopio_cache/masks/propio/<experimento>/cam<N>/ultima.png
      superposición de la última imagen, para revisar a ojo mientras corre
  ~/microscopio_cache/masks/propio/<experimento>/segmentacion.csv
      una fila por ciclo: cámara, fecha, número de células, segundos

Uso:
  ../venv/bin/python scripts/32_segmentar_en_vivo.py
  ../venv/bin/python scripts/32_segmentar_en_vivo.py --entrada combinada --una-vez
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT, cache_dir  # noqa: E402

ENTRADA_DEF = ROOT / "datasets" / "microscopio_propio"
UM_POR_PX = 0.2159          # calibración del 20x con la cámara IMX219 (2026-09-10)
RE_IMG = re.compile(r"^img_(\d{8}_\d{6})(_[LRTB])?\.tif$")


def cargar_modelo(fp16=True):
    import torch
    from cellpose import models
    m = models.CellposeModel(gpu=True, use_bfloat16=False)
    if fp16:
        m.net.half()
        m.net.dtype = torch.float16
    return m


def leer(ruta, reducir):
    import cv2
    import tifffile
    img = tifffile.imread(ruta).astype(np.float32)
    if img.ndim == 3:                       # por si llega en color: a gris
        img = img.mean(axis=-1)
    if reducir > 1:
        h, w = img.shape
        img = cv2.resize(img, (w // reducir, h // reducir), interpolation=cv2.INTER_AREA)
    return img


def preparar(fotos, sufijos, modo):
    """fotos: {sufijo: imagen}. Devuelve la imagen para Cellpose."""
    suma = np.mean([fotos[s] for s in sufijos], axis=0)
    if modo == "suma" or not {"_L", "_R", "_T", "_B"} <= set(fotos):
        return suma, None
    eps = 1e-6
    lr = (fotos["_L"] - fotos["_R"]) / (fotos["_L"] + fotos["_R"] + eps)
    tb = (fotos["_T"] - fotos["_B"]) / (fotos["_T"] + fotos["_B"] + eps)
    if modo == "dpc":
        return np.sqrt(lr ** 2 + tb ** 2), None
    # combinada: tres canales en la misma escala aproximada
    s = (suma - suma.mean()) / (suma.std() + eps)
    return np.stack([s, lr / (lr.std() + eps), tb / (tb.std() + eps)], axis=-1), 2


def superposicion(img, masks, ruta):
    """PNG chico con los contornos, para mirar mientras corre."""
    import cv2
    from skimage.segmentation import find_boundaries
    g = img if img.ndim == 2 else img[..., 0]
    lo, hi = np.percentile(g, (1, 99))
    g8 = (np.clip((g - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)
    rgb = np.dstack([g8] * 3)
    rgb[find_boundaries(masks, mode="inner")] = (255, 200, 40)
    esc = 1024 / max(rgb.shape[:2])
    if esc < 1:
        rgb = cv2.resize(rgb, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(ruta), rgb[..., ::-1])


def ciclos_completos(entrada):
    """Recorre entrada/<experimento>/cam*/ y devuelve los ciclos con todas
    sus fotos: [(experimento, cam, fecha, {sufijo: ruta}, sufijos)]."""
    salida = []
    for exp in sorted(p for p in entrada.iterdir() if p.is_dir()):
        sufijos = [""]
        meta = exp / "experimento.json"
        if meta.exists():
            try:
                sufijos = json.loads(meta.read_text()).get("sufijos") or [""]
            except (json.JSONDecodeError, OSError):
                pass
        for cam in sorted(exp.glob("cam*")):
            grupos = {}
            for f in cam.iterdir():
                m = RE_IMG.match(f.name)
                if m:
                    grupos.setdefault(m.group(1), {})[m.group(2) or ""] = f
            for fecha, fotos in sorted(grupos.items()):
                if all(s in fotos for s in sufijos):
                    salida.append((exp.name, cam.name, fecha, fotos, sufijos))
    return salida


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada-dir", type=Path, default=ENTRADA_DEF,
                    help="Carpeta que llena 31_receptor_microscopio.py")
    ap.add_argument("--salida-dir", type=Path, default=None,
                    help="Dónde guardar las máscaras (por defecto ~/microscopio_cache/masks/propio)")
    ap.add_argument("--entrada", choices=["suma", "dpc", "combinada"], default="suma",
                    help="Qué imagen recibe Cellpose (ver encabezado)")
    ap.add_argument("--reducir", type=int, default=2,
                    help="Reducción de resolución (2: 0.43 µm/px, sigue por debajo de la "
                         "resolución óptica del 20x y es ~4× más rápido)")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--guardar-flows", action="store_true",
                    help="Guardar flows para reajustar umbrales sin volver a correr la red")
    ap.add_argument("--una-vez", action="store_true",
                    help="Procesar lo que haya y salir (sin quedarse esperando)")
    ap.add_argument("--espera", type=float, default=2.0, help="Segundos entre revisiones")
    args = ap.parse_args()

    args.entrada_dir.mkdir(parents=True, exist_ok=True)
    salida = args.salida_dir or cache_dir("masks", "propio")
    salida.mkdir(parents=True, exist_ok=True)
    print(f"[en vivo] vigilando {args.entrada_dir}", flush=True)
    print(f"[en vivo] entrada={args.entrada}  reducir={args.reducir}  "
          f"({UM_POR_PX * args.reducir:.3f} µm/px)  ->  {salida}", flush=True)
    print("[en vivo] cargando Cellpose-SAM en la GPU…", flush=True)
    model = cargar_modelo()
    print("[en vivo] listo; esperando imágenes (Ctrl+C para detener)", flush=True)

    n_hechos, t_total = 0, 0.0
    try:
        while True:
            pendientes = []
            for exp, cam, fecha, fotos, sufijos in ciclos_completos(args.entrada_dir):
                destino = salida / exp / cam / f"{fecha}.npz"
                if not destino.exists():
                    pendientes.append((exp, cam, fecha, fotos, sufijos, destino))
            for exp, cam, fecha, fotos, sufijos, destino in pendientes:
                t0 = time.time()
                try:
                    imgs = {s: leer(fotos[s], args.reducir) for s in sufijos}
                except Exception as e:      # archivo ilegible: avisar y seguir
                    print(f"[en vivo] {exp}/{cam}/{fecha}: no se pudo leer ({e})", flush=True)
                    continue
                img, eje_canal = preparar(imgs, sufijos, args.entrada)
                masks, flows, _ = model.eval(img, batch_size=16, channel_axis=eje_canal,
                                             flow_threshold=args.flow_threshold,
                                             cellprob_threshold=args.cellprob_threshold)
                destino.parent.mkdir(parents=True, exist_ok=True)
                extra = {}
                if args.guardar_flows:
                    extra = {"dP": flows[1].astype(np.float16), "cellprob": flows[2].astype(np.float16)}
                np.savez_compressed(destino, masks=masks.astype(np.uint16),
                                    um_por_px=UM_POR_PX * args.reducir,
                                    entrada=args.entrada, reducir=args.reducir, **extra)
                superposicion(img, masks, destino.parent / "ultima.png")
                dt = time.time() - t0
                n_hechos += 1
                t_total += dt
                n_cel = int(masks.max())
                registro = salida / exp / "segmentacion.csv"
                nuevo = not registro.exists()
                with open(registro, "a", newline="") as f:
                    w = csv.writer(f)
                    if nuevo:
                        w.writerow(["camara", "fecha", "celulas", "segundos", "entrada", "reducir"])
                    w.writerow([cam, fecha, n_cel, round(dt, 2), args.entrada, args.reducir])
                print(f"[en vivo] {time.strftime('%H:%M:%S')}  {exp}/{cam}/{fecha}: "
                      f"{n_cel} células en {dt:.1f} s  (promedio {t_total / n_hechos:.1f} s)",
                      flush=True)
            if args.una_vez:
                break
            time.sleep(args.espera)
    except KeyboardInterrupt:
        pass
    print(f"\n[en vivo] detenido: {n_hechos} ciclos segmentados", flush=True)


if __name__ == "__main__":
    main()
