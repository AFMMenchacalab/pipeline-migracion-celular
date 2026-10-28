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
ENTRADAS = ("suma", "dpc", "combinada", "fase", "combinada+dpc")
# combinada+dpc (prueba 11, MG63 con etopósido, 2026-10-10): la combinada no
# ve las células oscuras y granulosas, la magnitud del DPC sí, pero pierde
# las planas y tenues. Se corre Cellpose con las dos y a lo de la combinada
# se le agrega lo del DPC que no se encima (< UNION_ENCIME_MAX de su área).
# Con células sueltas el filtro de forma quitaba células alargadas o muy
# extendidas: en este modo solo se quitan los objetos diminutos.
# La magnitud del DPC se suaviza antes de Cellpose: las células planas son
# una textura salpicada y suavizada queda como una mancha completa (cam1
# día 3: 25 -> 35 células). Con eso aparecen también algunos objetos en
# zonas vacías; de lo que agrega el DPC se quedan solo los que tienen un
# DPC medio de al menos UNION_CONTRASTE_MIN veces el del fondo (falsos:
# 1.2-1.7, células: 1.8 o más). Lo de la combinada no se filtra así: hay
# células planas que encuentra con contraste 1.3.
UNION_ENCIME_MAX = 0.2
DPC_SUAVIZADO_UM = 2.0
UNION_CONTRASTE_MIN = 1.7
# Fondo del campo claro en la entrada combinada: el mismo suavizado que usa
# la Pi para el DPC (core/dpc.py, sigma 150 px a resolución completa).
SIGMA_FONDO_PX = 150
# Filtro de forma después de Cellpose (ajustado con prueba 7, 2026-10-05):
# con flow_threshold 0.8 aparecen las células de las láminas que con 0.4 se
# descartaban, y también algunas rebanadas y astillas que no son células.
FORMA_AREA_MIN_UM2 = 80
FORMA_SOLIDEZ_MIN = 0.75
FORMA_EJE_MENOR_MIN_UM = 6
FORMA_ASPECTO_MAX = 4
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
    return _combinar(suma, lr, tb, reducir), 2


def _combinar(suma, lr, tb, reducir, eps=1e-6):
    # El campo claro se divide por su fondo: con el viñeteado de la matriz
    # el centro queda tan brillante que, sin aplanar, Cellpose no ve las
    # células planas de esa zona (prueba 7: cobertura 0.47 -> 0.60).
    from scipy.ndimage import gaussian_filter
    plano = suma / np.maximum(gaussian_filter(suma, SIGMA_FONDO_PX / reducir), eps)
    s = (plano - plano.mean()) / (plano.std() + eps)
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
    return _combinar(fotos["_suma"], lr, tb, reducir), 2


def filtrar_formas(masks, um_por_px):
    """Quita los objetos que no parecen células: diminutos, delgados
    (rebanadas, astillas en los bordes) o poco compactos. Renumera."""
    from skimage.measure import regionprops
    from skimage.segmentation import relabel_sequential
    fuera = []
    for p in regionprops(masks):
        menor = max(p.axis_minor_length, 1e-6)
        if (p.area * um_por_px ** 2 < FORMA_AREA_MIN_UM2 or p.solidity < FORMA_SOLIDEZ_MIN
                or menor * um_por_px < FORMA_EJE_MENOR_MIN_UM
                or p.axis_major_length / menor > FORMA_ASPECTO_MAX):
            fuera.append(p.label)
    if fuera:
        masks = np.where(np.isin(masks, fuera), 0, masks)
    return relabel_sequential(masks)[0]


def quitar_diminutos(masks, um_por_px):
    """Solo la parte del área de filtrar_formas. Renumera."""
    from skimage.segmentation import relabel_sequential
    areas = np.bincount(masks.ravel())
    fuera = np.flatnonzero(areas * um_por_px ** 2 < FORMA_AREA_MIN_UM2)
    masks = np.where(np.isin(masks, fuera[fuera > 0]), 0, masks)
    return relabel_sequential(masks)[0]


def unir(base, extra, mag=None):
    """base + los objetos de extra que se enciman menos de UNION_ENCIME_MAX
    de su área con base (solo en los píxeles libres). Con mag (magnitud del
    DPC), solo los de extra con contraste >= UNION_CONTRASTE_MIN."""
    salida = base.astype(np.int32)
    n = int(salida.max())
    areas = np.bincount(extra.ravel())
    encime = np.bincount(extra[base > 0].ravel(), minlength=len(areas))
    if mag is not None:
        fondo = np.median(mag[(base == 0) & (extra == 0)])
        media = np.bincount(extra.ravel(), weights=mag.ravel()) / np.maximum(areas, 1)
    for etq in np.flatnonzero(areas[1:]) + 1:
        if encime[etq] >= UNION_ENCIME_MAX * areas[etq]:
            continue
        if mag is not None and media[etq] < UNION_CONTRASTE_MIN * fondo:
            continue
        n += 1
        salida[(extra == etq) & (salida == 0)] = n
    return salida


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
                    guardar_flows=False, filtrar=True):
    """Segmenta un ciclo y guarda máscaras, superposición y registro.
    Devuelve (número de células, segundos)."""
    t0 = time.time()
    imgs = {s: leer(fotos[s], reducir) for s in sufijos}
    union = entrada == "combinada+dpc"
    img, eje_canal = preparar(imgs, sufijos, "combinada" if union else entrada, reducir)
    masks, flows, _ = model.eval(img, batch_size=16, channel_axis=eje_canal,
                                 flow_threshold=flow_threshold,
                                 cellprob_threshold=cellprob_threshold)
    if union:
        from scipy.ndimage import gaussian_filter
        mag, _ = preparar(imgs, sufijos, "dpc", reducir)
        img_dpc = gaussian_filter(mag, DPC_SUAVIZADO_UM / (UM_POR_PX * reducir))
        masks_dpc = model.eval(img_dpc, batch_size=16, channel_axis=None,
                               flow_threshold=flow_threshold,
                               cellprob_threshold=cellprob_threshold)[0]
        if filtrar:
            masks = quitar_diminutos(masks, UM_POR_PX * reducir)
            masks_dpc = quitar_diminutos(masks_dpc, UM_POR_PX * reducir)
        masks = unir(masks, masks_dpc, mag)
    elif filtrar:
        masks = filtrar_formas(masks, UM_POR_PX * reducir)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    extra = {}
    if guardar_flows:
        extra = {"dP": flows[1].astype(np.float16), "cellprob": flows[2].astype(np.float16)}
    np.savez_compressed(destino, masks=masks.astype(np.uint16), um_por_px=UM_POR_PX * reducir,
                        entrada=entrada, reducir=reducir, flow_threshold=flow_threshold,
                        cellprob_threshold=cellprob_threshold, filtro_formas=filtrar, **extra)
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
