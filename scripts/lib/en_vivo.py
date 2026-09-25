"""
Segmentación en vivo de las imágenes del microscopio propio: funciones que
usan tanto 32_segmentar_en_vivo.py (consola) como 33_interfaz.py (interfaz
gráfica).

Dispositivo (GPU): Cellpose usa PyTorch, y PyTorch expone la GPU con la
misma API (`torch.cuda`) tanto en NVIDIA (CUDA) como en AMD (ROCm). Por eso
el mismo código corre en la PC con la RX 6800 XT y en la del laboratorio
con la RTX 3070; lo único que cambia es qué paquete de PyTorch se instala
(ver instalar.sh / instalar.ps1). Sin GPU cae a CPU (unas 40× más lento).

fp16: en NVIDIA desde Turing (RTX 20xx) y en AMD RDNA2 la media precisión
es nativa y da las mismas máscaras que fp32 (ver 11_segmentar_gpu.py).
"""
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

UM_POR_PX = 0.2159          # calibración del 20x con la cámara IMX219 (2026-09-10)
RE_IMG = re.compile(r"^img_(\d{8}_\d{6})(_[LRTB])?\.tif$")
ENTRADAS = ("suma", "dpc", "combinada", "fase")
_FASE = None                # ReconstructorEnVivo, se crea al primer uso


def dispositivo():
    """Qué va a usar Cellpose: {'backend': 'CUDA'|'ROCm'|'CPU', 'nombre': ...}."""
    try:
        import torch
    except ImportError:
        return {"backend": "sin PyTorch", "nombre": "", "gpu": False}
    if torch.cuda.is_available():
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        return {"backend": backend, "nombre": torch.cuda.get_device_name(0), "gpu": True,
                "torch": torch.__version__}
    return {"backend": "CPU", "nombre": "sin GPU compatible", "gpu": False,
            "torch": torch.__version__}


def cargar_modelo(fp16=True):
    import torch
    from cellpose import models
    gpu = torch.cuda.is_available()
    m = models.CellposeModel(gpu=gpu, use_bfloat16=False)
    if fp16 and gpu:
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


def preparar(fotos, sufijos, modo, reducir=1):
    """fotos: {sufijo: imagen}. Devuelve (imagen para Cellpose, eje de canal)."""
    global _FASE
    if modo == "fase":
        if not {"_L", "_R", "_T", "_B"} <= set(fotos):
            raise ValueError("la entrada 'fase' necesita las 4 fotos DPC (L, R, T, B)")
        if _FASE is None:
            from .fase_dpc import ReconstructorEnVivo
            _FASE = ReconstructorEnVivo()
        return _FASE(fotos, reducir), None
    suma = np.mean([fotos[s] for s in sufijos], axis=0)
    if modo == "suma" or not {"_L", "_R", "_T", "_B"} <= set(fotos):
        return suma, None
    eps = 1e-6
    # Cada mitad de la matriz ilumina el campo distinto (viñeteado, ángulo):
    # sin normalizar, (L-R)/(L+R) arrastra un gradiente de fondo del mismo
    # orden que el relieve de los bordes (~1% con la matriz lejos).
    from .fase_dpc import normalizar
    n = {s: normalizar(fotos[s]) for s in ("_L", "_R", "_T", "_B")}
    lr = (n["_L"] - n["_R"]) / (n["_L"] + n["_R"] + eps)
    tb = (n["_T"] - n["_B"]) / (n["_T"] + n["_B"] + eps)
    if modo == "dpc":
        return np.sqrt(lr ** 2 + tb ** 2), None
    s = (suma - suma.mean()) / (suma.std() + eps)
    return np.stack([s, lr / (lr.std() + eps), tb / (tb.std() + eps)], axis=-1), 2


def a_8bits(img):
    g = img if img.ndim == 2 else img[..., 0]
    lo, hi = np.percentile(g, (1, 99))
    return (np.clip((g - lo) / (hi - lo + 1e-6), 0, 1) * 255).astype(np.uint8)


def superposicion(img, masks, ruta, lado_max=1024):
    """PNG con los contornos, para mirar mientras corre."""
    import cv2
    from skimage.segmentation import find_boundaries
    g8 = a_8bits(img)
    rgb = np.dstack([g8] * 3)
    rgb[find_boundaries(masks, mode="inner")] = (255, 200, 40)
    esc = lado_max / max(rgb.shape[:2])
    if esc < 1:
        rgb = cv2.resize(rgb, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    tmp = Path(ruta).with_suffix(".tmp.png")
    cv2.imwrite(str(tmp), rgb[..., ::-1])
    tmp.replace(ruta)                       # que la interfaz nunca lea un PNG a medias


def sufijos_de(exp_dir):
    meta = Path(exp_dir) / "experimento.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("sufijos") or [""]
        except (json.JSONDecodeError, OSError):
            pass
    return [""]


def ciclos(entrada):
    """Todos los ciclos de todas las cámaras de todos los experimentos:
    [(experimento, cam, fecha, {sufijo: ruta}, sufijos, completo)]."""
    salida = []
    entrada = Path(entrada)
    if not entrada.is_dir():
        return salida
    for exp in sorted(p for p in entrada.iterdir() if p.is_dir()):
        sufijos = sufijos_de(exp)
        for cam in sorted(exp.glob("cam*")):
            grupos = {}
            for f in cam.iterdir():
                m = RE_IMG.match(f.name)
                if m:
                    grupos.setdefault(m.group(1), {})[m.group(2) or ""] = f
            for fecha, fotos in sorted(grupos.items()):
                salida.append((exp.name, cam.name, fecha, fotos, sufijos,
                               all(s in fotos for s in sufijos)))
    return salida


def pendientes(entrada, salida):
    return [(e, c, f, fotos, suf, Path(salida) / e / c / f"{f}.npz")
            for e, c, f, fotos, suf, completo in ciclos(entrada)
            if completo and not (Path(salida) / e / c / f"{f}.npz").exists()]


def segmentar_ciclo(model, exp, cam, fecha, fotos, sufijos, destino, entrada="suma",
                    reducir=2, flow_threshold=0.4, cellprob_threshold=0.0,
                    guardar_flows=False):
    """Segmenta un ciclo y guarda máscaras, superposición y registro.
    Devuelve (número de células, segundos)."""
    t0 = time.time()
    imgs = {s: leer(fotos[s], reducir) for s in sufijos}
    img, eje_canal = preparar(imgs, sufijos, entrada, reducir)
    masks, flows, _ = model.eval(img, batch_size=16, channel_axis=eje_canal,
                                 flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if guardar_flows:
        extra = {"dP": flows[1].astype(np.float16), "cellprob": flows[2].astype(np.float16)}
    np.savez_compressed(destino, masks=masks.astype(np.uint16), um_por_px=UM_POR_PX * reducir,
                        entrada=entrada, reducir=reducir, **extra)
    superposicion(img, masks, destino.parent / "ultima.png")
    dt = time.time() - t0
    n_cel = int(masks.max())
    registro = destino.parent.parent / "segmentacion.csv"
    nuevo = not registro.exists()
    with open(registro, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["camara", "fecha", "celulas", "segundos", "entrada", "reducir"])
        w.writerow([cam, fecha, n_cel, round(dt, 2), entrada, reducir])
    return n_cel, dt
