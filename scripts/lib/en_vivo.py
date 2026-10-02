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
# Dos formas de nombre: img_20260831_121729_L.tif (hasta septiembre) y
# 0001_2026-10-02_10-30-00_L.tif (MicroscopeOS desde el 2026-10-02: número
# de ciclo, fecha y hora). Además de las crudas _L/_R/_T/_B pueden llegar
# los archivos que deja core/dpc.py de la Pi cuando borra las crudas.
SUFIJOS_PI = ("_dpcLR", "_dpcTB", "_suma", "_fase")
RE_IMG = re.compile(r"^(?:img_(\d{8}_\d{6})|(\d{4}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}))"
                    r"(_[LRTB]|_dpcLR|_dpcTB|_suma|_fase)?\.tif$")
ETIQUETA_PI = 65000         # TIFF tag con el JSON de metadatos de MicroscopeOS
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
    """Imagen en float32, reducida `reducir` veces respecto de la resolución
    completa de la cámara.

    Los TIFF que calcula la Pi (core/dpc.py) vienen en uint16 con el cero en
    32768: se devuelven en su valor físico (DPC de -1 a 1, fase en rad)
    según el "cero" y la "escala" de sus metadatos. La suma llega ya
    reducida (por defecto a la mitad): se lleva al mismo tamaño que el DPC.
    """
    import cv2
    import tifffile
    with tifffile.TiffFile(ruta) as t:
        img = t.pages[0].asarray().astype(np.float32)
        tag = t.pages[0].tags.get(ETIQUETA_PI)
        texto = tag.value if tag is not None else None
    try:
        meta = json.loads(texto) if texto else {}
    except (TypeError, ValueError):
        meta = {}
    if img.ndim == 3:                       # por si llega en color: a gris
        img = img.mean(axis=-1)
    for clave, escala_vieja in (("dpc", 1 / 32767), ("fase", 1e-4)):
        if isinstance(meta.get(clave), dict):
            v = meta[clave]                 # sin "escala": primer formato, 1/32767
            img = (img - v.get("cero", 32768)) * v.get("escala", escala_vieja)
    guardada = float((meta.get("suma") or {}).get("reduccion", 1) or 1)
    h, w = img.shape
    alto, ancho = round(h * guardada) // reducir, round(w * guardada) // reducir
    if (alto, ancho) != (h, w):
        interp = cv2.INTER_AREA if ancho < w else cv2.INTER_LINEAR
        img = cv2.resize(img, (ancho, alto), interpolation=interp)
    return img


def preparar(fotos, sufijos, modo, reducir=1):
    """fotos: {sufijo: imagen}. Devuelve (imagen para Cellpose, eje de canal).

    Sirve igual con las 4 crudas que con lo que deja la Pi cuando las borra
    (_dpcLR, _dpcTB, _suma): mismas entradas, mismo resultado."""
    global _FASE
    if {"_dpcLR", "_dpcTB"} <= set(fotos):
        return _preparar_pi(fotos, modo, reducir)
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
    return _combinar(suma, lr, tb), 2


def _combinar(suma, lr, tb, eps=1e-6):
    s = (suma - suma.mean()) / (suma.std() + eps)
    return np.stack([s, lr / (lr.std() + eps), tb / (tb.std() + eps)], axis=-1)


def _preparar_pi(fotos, modo, reducir):
    """Lo mismo que preparar() con los DPC que calculó la Pi. La Pi ya
    normalizó cada foto por su fondo antes de restar (core/dpc.py, sigma
    150 px): lr y tb llegan listos."""
    global _FASE
    lr, tb = fotos["_dpcLR"], fotos["_dpcTB"]
    if modo == "fase":
        if _FASE is None:
            from .fase_dpc import ReconstructorEnVivo
            _FASE = ReconstructorEnVivo()
        return _FASE(fotos, reducir), None
    if modo == "dpc":
        return np.sqrt(lr ** 2 + tb ** 2), None
    if "_suma" not in fotos:
        raise ValueError(f"la entrada '{modo}' necesita el campo claro (_suma.tif) y este "
                         "experimento no lo guardó: usar 'dpc' o 'fase'")
    if modo == "suma":
        return fotos["_suma"], None
    return _combinar(fotos["_suma"], lr, tb), 2


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


def _experimento(exp_dir):
    try:
        return json.loads((Path(exp_dir) / "experimento.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def sufijos_de(exp_dir):
    """Lo que la Pi captura en cada ciclo ("" = una sola foto)."""
    return _experimento(exp_dir).get("sufijos") or [""]


def sufijos_guardados(exp_dir):
    """Si la Pi calculó el DPC y borró las crudas, lo que queda de cada
    ciclo (p. ej. ["_dpcLR", "_dpcTB", "_suma"]); si no, None."""
    return (_experimento(exp_dir).get("dpc_procesado") or {}).get("sufijos")


def grupos(carpeta):
    """{fecha: {sufijo: ruta}} de las imágenes de una carpeta camN. La
    "fecha" es lo que comparten las fotos de un ciclo: 20260831_121729 o
    0001_2026-10-02_10-30-00."""
    salida = {}
    for f in Path(carpeta).iterdir():
        m = RE_IMG.match(f.name)
        if m:
            salida.setdefault(m.group(1) or m.group(2), {})[m.group(3) or ""] = f
    return salida


def ciclos(entrada):
    """Todos los ciclos de todas las cámaras de todos los experimentos:
    [(experimento, cam, fecha, {sufijo: ruta}, sufijos, completo)]."""
    salida = []
    entrada = Path(entrada)
    if not entrada.is_dir():
        return salida
    for exp in sorted(p for p in entrada.iterdir() if p.is_dir()):
        capturados, guardados = sufijos_de(exp), sufijos_guardados(exp)
        for cam in sorted(exp.glob("cam*")):
            for fecha, fotos in sorted(grupos(cam).items()):
                sufijos = capturados
                # Con crudas borradas se espera lo que calcula la Pi; si a
                # un ciclo le falló el cálculo, la Pi conserva y manda las
                # crudas, y entonces se usan esas.
                if guardados and not (all(s in fotos for s in capturados)
                                      and not all(s in fotos for s in guardados)):
                    sufijos = guardados
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
