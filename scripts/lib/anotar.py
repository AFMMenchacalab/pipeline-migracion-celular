"""
Contornos dibujados a mano en la interfaz (página /anotar), opcional.

Para qué: medir qué tan bien segmenta Cellpose (con contornos a mano como
referencia, no a ojo) y, más adelante, entrenarlo con esas fotos.

Se marcan unas cuantas fotos elegidas al azar, repartidas a lo largo del
experimento y entre las cámaras (las del día 3 se segmentan distinto que
las del día 1). Se dibuja sobre la misma imagen y a la misma resolución que
las máscaras de Cellpose (reducir 3), así se comparan píxel a píxel.

Se guarda en ~/microscopio_cache/anotaciones/propio/<exp>/:
  muestra.json         las fotos elegidas (no cambian al volver a entrar)
  <cam>/<fecha>.npz    masks (una etiqueta por célula), um_por_px, reducir,
                       poligonos (JSON) y si se partió de lo de Cellpose

Las primeras y las últimas fotos de la muestra (PRUEBA) se marcan desde
cero, sin ver lo de Cellpose: son las que miden la segmentación. Las demás
se pueden corregir a partir de Cellpose (más rápido; sirven para entrenar).

Para marcar en otra computadora (38_paquete_anotar.py): un paquete con solo
las fotos de la muestra y sus máscaras de Cellpose en _anotar/. Se agrega en
la interfaz con «Agregar fotos de una carpeta» (instalar_paquete) y las
marcas vuelven en un .zip (zip_marcas).
"""
import io
import json
import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np

from . import en_vivo as EV
from .config import cache_dir

REDUCIR = 3
VISTAS = ("campo", "dpc", "color")
# Una célula de Cellpose cuenta como acierto si se encima con una dibujada
# a mano con IoU >= 0.5 (el criterio usual para segmentación de instancias).
IOU_ACIERTO = 0.5
# Muestra: N_MUESTRA fotos; las PRUEBA primeras y PRUEBA últimas (inicio y
# final del experimento, alternando cámaras) se marcan desde cero.
N_MUESTRA = 14
PRUEBA = 2


def carpeta(exp):
    return cache_dir("anotaciones", "propio", exp)


def _ciclos_completos(entrada_dir, exp):
    return [(cam, fecha) for e, cam, fecha, fotos, suf, completo in EV.ciclos(entrada_dir)
            if e == exp and completo and all(fotos[s].stat().st_size > 0 for s in suf)]


def muestra(entrada_dir, exp, n=N_MUESTRA, otra=False):
    """Fotos para marcar: n repartidas a lo largo del tiempo, alternando
    cámaras, al azar dentro de cada tramo. Se guarda la lista: la misma
    muestra cada vez que se entra. otra=True agrega una foto al azar."""
    ruta = carpeta(exp) / "muestra.json"
    lista = json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else []
    if not lista or otra:
        todos = _ciclos_completos(entrada_dir, exp)
        if not todos:
            return []
        rnd = random.Random()
        usados = {(x["cam"], x["fecha"]) for x in lista}
        if not lista:
            cams = sorted({c for c, _ in todos})
            for i in range(n):
                cam = cams[i % len(cams)]
                serie = sorted(f for c, f in todos if c == cam)
                k = len(serie)
                tramo = serie[i * k // n:(i + 1) * k // n] or serie
                lista.append({"cam": cam, "fecha": rnd.choice(tramo),
                              "prueba": i < PRUEBA or i >= n - PRUEBA})
        else:
            libres = [x for x in todos if x not in usados]
            if libres:
                cam, fecha = rnd.choice(libres)
                lista.append({"cam": cam, "fecha": fecha, "prueba": False})
        ruta.write_text(json.dumps(lista, indent=1), encoding="utf-8")
    for x in lista:
        a = carpeta(exp) / x["cam"] / f"{x['fecha']}.npz"
        x["marcada"] = a.exists()
        x["celulas"] = int(np.load(a)["masks"].max()) if a.exists() else 0
    return lista


def _fotos(entrada_dir, exp, cam, fecha):
    g = EV.grupos(Path(entrada_dir) / exp / cam).get(fecha)
    if not g:
        raise FileNotFoundError(f"no está el ciclo {cam}/{fecha}")
    suf = [s for s in ("_dpcLR", "_dpcTB", "_suma") if s in g]
    return {s: EV.leer(g[s], REDUCIR) for s in suf}


def imagen(entrada_dir, exp, cam, fecha, vista):
    """PNG para dibujar encima. campo: campo claro aplanado (el canal 1 de
    la combinada); dpc: magnitud del relieve; color: DPC izquierda-derecha
    en rojo y arriba-abajo en verde, como el _dpc.jpg de la Pi."""
    import cv2
    f = _fotos(entrada_dir, exp, cam, fecha)
    lr, tb = f["_dpcLR"], f["_dpcTB"]
    if vista == "campo" and "_suma" in f:
        g8 = EV.a_8bits(EV._combinar(f["_suma"], lr, tb, REDUCIR)[..., 0])
    elif vista == "color":
        def canal(x):
            s = np.percentile(np.abs(x), 99.5) + 1e-6
            return (np.clip(x / s, -1, 1) * 127 + 128).astype(np.uint8)
        g8 = np.dstack([np.full_like(canal(lr), 128), canal(tb), canal(lr)])   # BGR
    else:
        g8 = EV.a_8bits(np.sqrt(lr ** 2 + tb ** 2))
    return cv2.imencode(".png", g8)[1].tobytes()


def _poligonos(masks):
    """Contorno de cada etiqueta, simplificado a ~1 px."""
    import cv2
    salida = []
    for etq in np.unique(masks)[1:]:
        cs, _ = cv2.findContours((masks == etq).astype(np.uint8), cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0, :]
            if len(c) >= 3:
                salida.append(c.tolist())
    return salida


def de_cellpose(masks_dir, exp, cam, fecha):
    """Contornos de Cellpose para esa foto (de las máscaras ya hechas)."""
    ruta = Path(masks_dir) / exp / cam / f"{fecha}.npz"
    if not ruta.exists():
        return None
    return _poligonos(np.load(ruta)["masks"])


def cargar(exp, cam, fecha):
    ruta = carpeta(exp) / cam / f"{fecha}.npz"
    if not ruta.exists():
        return None
    z = np.load(ruta)
    return {"poligonos": json.loads(str(z["poligonos"])),
            "partio_de_cellpose": bool(z["partio_de_cellpose"])}


def guardar(entrada_dir, masks_dir, exp, cam, fecha, poligonos, partio_de_cellpose):
    """Rasteriza los polígonos (lo dibujado después queda encima) y guarda.
    Devuelve la comparación con Cellpose si ya está segmentada."""
    import cv2
    alto, ancho = _fotos(entrada_dir, exp, cam, fecha)["_dpcLR"].shape
    masks = np.zeros((alto, ancho), np.uint16)
    limpios = []
    for p in poligonos:
        p = np.round(np.asarray(p, float)).astype(np.int32)
        if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2:
            continue
        limpios.append(p.tolist())
        cv2.fillPoly(masks, [p], len(limpios))
    destino = carpeta(exp) / cam
    destino.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destino / f"{fecha}.npz", masks=masks, um_por_px=EV.UM_POR_PX * REDUCIR,
                        reducir=REDUCIR, poligonos=json.dumps(limpios),
                        partio_de_cellpose=bool(partio_de_cellpose), guardada=time.time())
    auto = Path(masks_dir) / exp / cam / f"{fecha}.npz"
    return comparar(masks, np.load(auto)["masks"]) if auto.exists() else None


def comparar(mano, auto, iou_min=IOU_ACIERTO):
    """Aciertos de Cellpose contra lo dibujado a mano: cada célula a mano se
    empareja con la de Cellpose que más se le encima (IoU >= iou_min).
    sensibilidad = encontradas / a mano; precision = correctas / de Cellpose."""
    n_m, n_a = int(mano.max()), int(auto.max())
    if n_m == 0 or n_a == 0:
        return {"a_mano": n_m, "cellpose": n_a, "aciertos": 0, "sensibilidad": 0.0 if n_m else None,
                "precision": 0.0 if n_a else None, "f1": 0.0, "iou_medio": None}
    # tabla de encime entre etiquetas, sin recorrer célula por célula
    par = mano.astype(np.int64) * (n_a + 1) + auto.astype(np.int64)
    inter = np.bincount(par.ravel(), minlength=(n_m + 1) * (n_a + 1)).reshape(n_m + 1, n_a + 1)
    area_m, area_a = inter.sum(1), inter.sum(0)
    iou = inter / np.maximum(area_m[:, None] + area_a[None, :] - inter, 1)
    iou = iou[1:, 1:]
    libre, ious = np.ones(n_a, bool), []
    for i in np.argsort(-iou.max(1)):
        fila = np.where(libre, iou[i], -1)
        j = int(np.argmax(fila))
        if fila[j] >= iou_min:
            libre[j] = False
            ious.append(float(fila[j]))
    aciertos = len(ious)
    sens, prec = aciertos / n_m, aciertos / n_a
    return {"a_mano": n_m, "cellpose": n_a, "aciertos": aciertos,
            "sensibilidad": round(sens, 3), "precision": round(prec, 3),
            "f1": round(2 * sens * prec / (sens + prec), 3) if aciertos else 0.0,
            "iou_medio": round(float(np.mean(ious)), 3) if ious else None}


def instalar_paquete(paquete, exp, masks_dir):
    """Después de importar un paquete de 38_paquete_anotar.py: las máscaras
    de Cellpose y la lista de fotos de _anotar/ van a donde las busca la
    interfaz (sin pisar lo que ya haya). Devuelve cuántas máscaras copió."""
    origen = Path(paquete) / "_anotar"
    if not origen.is_dir():
        return 0
    n = 0
    for npz in origen.glob("cellpose/cam*/*.npz"):
        destino = Path(masks_dir) / exp / npz.parent.name / npz.name
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(npz, destino)
            n += 1
    m = origen / "muestra.json"
    if m.exists() and not (carpeta(exp) / "muestra.json").exists():
        shutil.copy2(m, carpeta(exp) / "muestra.json")
    return n


def zip_marcas(exp):
    """Las marcas de un experimento en un .zip, para llevarlas a otra PC."""
    buf = io.BytesIO()
    base = carpeta(exp)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(base.rglob("*")):
            if f.is_file():
                z.write(f, Path(exp) / f.relative_to(base))
    return buf.getvalue()


def resumen(masks_dir, exp):
    """Comparación de las fotos marcadas, por cámara y en total (sumando
    aciertos, no promediando fotos). El total cuenta solo las marcadas
    desde cero: las corregidas a partir de Cellpose se le parecen por
    construcción y lo harían ver mejor de lo que es."""
    filas = []
    for a in sorted(carpeta(exp).glob("cam*/*.npz")):
        auto = Path(masks_dir) / exp / a.parent.name / a.name
        if auto.exists():
            z = np.load(a)
            c = comparar(z["masks"], np.load(auto)["masks"])
            filas.append(dict(c, cam=a.parent.name, fecha=a.stem,
                              partio_de_cellpose=bool(z["partio_de_cellpose"])))
    grupos = {}
    for f in filas:
        if f["partio_de_cellpose"]:
            continue
        for clave in (f["cam"], "total"):
            g = grupos.setdefault(clave, {"fotos": 0, "a_mano": 0, "cellpose": 0, "aciertos": 0})
            for k in ("a_mano", "cellpose", "aciertos"):
                g[k] += f[k]
            g["fotos"] += 1
    for g in grupos.values():
        s = g["aciertos"] / g["a_mano"] if g["a_mano"] else 0
        p = g["aciertos"] / g["cellpose"] if g["cellpose"] else 0
        g.update(sensibilidad=round(s, 3), precision=round(p, 3),
                 f1=round(2 * s * p / (s + p), 3) if s + p else 0.0)
    return {"fotos": filas, "grupos": grupos}
