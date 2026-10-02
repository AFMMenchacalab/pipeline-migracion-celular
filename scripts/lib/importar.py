"""
Agregar a la carpeta de entrada fotos que no llegaron por la recepción en
vivo: tomadas a mano en la Pi, bajadas de su página, copiadas de una
memoria USB, etc. Las usa la interfaz (33_interfaz.py, «Agregar fotos de
una carpeta»).

Qué acepta: una carpeta con TIFF sueltos o con subcarpetas camN, con
cualquiera de los nombres que usó la Pi o uno propio:

    cam0_20261001_162709_L.tif        foto manual (hasta el 2026-10-01)
    cam0/2026-10-02_10-30-00_L.tif    foto manual (MicroscopeOS nuevo)
    0001_2026-10-02_10-30-00_L.tif    timelapse
    campo3_L.tif                      nombre propio (la hora sale del archivo)

Las fotos de un mismo momento (mismo nombre salvo _L/_R/_T/_B, o los
_dpcLR/_dpcTB/_suma que deja la Pi) son un ciclo; una foto sin sufijo es un
ciclo de una sola imagen (campo claro).

No se copian: se enlazan dentro de <entrada>/<experimento>/camN/ con el
nombre img_<AAAAMMDD_HHMMSS><sufijo>.tif que entiende todo el pipeline, y
se escribe experimento.json como lo haría la Pi. Volver a importar la misma
carpeta agrega solo las fotos nuevas, con el mismo nombre de ciclo que ya
tenían (así no se vuelven a segmentar).
"""
import json
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

SUFIJOS_CRUDAS = ("_L", "_R", "_T", "_B")
RE_FOTO = re.compile(r"^(?P<base>.+?)(?P<suf>_[LRTB]|_dpcLR|_dpcTB|_suma|_fase)?\.tiff?$", re.I)
RE_CAM = re.compile(r"(?:^|[_\-])cam(\d+)(?=[_\-]|$)", re.I)
RE_FECHAS = (
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})"), None),
    (re.compile(r"(\d{8}_\d{6})"), "%Y%m%d_%H%M%S"),
)
# Lo que escriben los scripts del pipeline o la Pi junto a las fotos y no es
# una foto del microscopio.
IGNORAR = re.compile(r"^(fase_|preview_)|_conteo", re.I)


def fecha_de(texto):
    """datetime de un nombre de ciclo (cualquiera de los formatos de la Pi), o None."""
    for rx, formato in RE_FECHAS:
        m = rx.search(texto)
        if not m:
            continue
        try:
            if formato:
                return datetime.strptime(m.group(1), formato)
            return datetime(*map(int, m.groups()))
        except ValueError:
            pass
    return None


def _sufijo(s):
    """Sufijo con las mayúsculas del pipeline (_l.TIF -> _L)."""
    if not s:
        return ""
    for k in SUFIJOS_CRUDAS + ("_dpcLR", "_dpcTB", "_suma", "_fase"):
        if s.lower() == k.lower():
            return k
    return s


def buscar(carpeta):
    """{(cam, base): {sufijo: ruta}} de los TIFF de la carpeta y de sus
    subcarpetas (un nivel, o dos si es un experimento con camN/)."""
    carpeta = Path(carpeta)
    archivos = []
    for nivel in (carpeta.glob("*"), carpeta.glob("*/*"), carpeta.glob("*/*/*")):
        archivos += [p for p in nivel if p.is_file()]
    grupos = {}
    for p in sorted(archivos):
        m = RE_FOTO.match(p.name)
        if not m or IGNORAR.search(p.name):
            continue
        base, suf = m.group("base"), _sufijo(m.group("suf"))
        mc = RE_CAM.search(base) or next(
            (RE_CAM.search(q.name) for q in p.relative_to(carpeta).parents if RE_CAM.search(q.name)), None)
        cam = f"cam{mc.group(1)}" if mc else "cam0"
        # La subcarpeta forma parte del ciclo: dos carpetas con fotos del
        # mismo nombre son ciclos distintos.
        sub = p.parent.relative_to(carpeta).as_posix()
        clave = (cam, base if sub == "." else f"{sub}/{base}")
        grupos.setdefault(clave, {})[suf] = p
    return grupos


def _sufijos(grupos):
    """(capturados, guardados) como en experimento.json de la Pi."""
    hay = [set(f) for f in grupos.values()]
    procesados = [h for h in hay if {"_dpcLR", "_dpcTB"} <= h]
    guardados = None
    if procesados:
        guardados = ["_dpcLR", "_dpcTB"] + (["_suma"] if all("_suma" in h for h in procesados) else [])
    if procesados or any(h & set(SUFIJOS_CRUDAS) for h in hay):
        return list(SUFIJOS_CRUDAS), guardados
    return [""], None


def _enlazar(origen, destino):
    """Enlace simbólico; si el sistema no deja (Windows sin modo de
    desarrollador), enlace duro; si tampoco (otro disco), copia."""
    for hacer in (lambda: os.symlink(origen, destino), lambda: os.link(origen, destino)):
        try:
            hacer()
            return
        except (OSError, NotImplementedError):
            pass
    shutil.copy2(origen, destino)


def _nombre_libre(entrada, deseado):
    limpio = re.sub(r"[^A-Za-z0-9_\-]+", "_", deseado).strip("_")[:60] or "fotos"
    nombre, k = limpio, 2
    while (Path(entrada) / nombre).exists():
        nombre, k = f"{limpio}_{k}", k + 1
    return nombre


def importar(carpeta, entrada, nombre=None):
    """Agrega las fotos de `carpeta` como un experimento de `entrada`.
    Devuelve {"experimento", "nuevos", "ciclos", "sufijos", "modo"}."""
    carpeta = Path(carpeta).expanduser().resolve()
    entrada = Path(entrada)
    if not carpeta.is_dir():
        raise ValueError(f"No existe la carpeta {carpeta}")
    if entrada.resolve() in (carpeta, *carpeta.parents):
        raise ValueError("Esa carpeta ya está dentro de la carpeta de entrada")
    grupos = buscar(carpeta)
    if not grupos:
        raise ValueError(f"No encontré fotos .tif en {carpeta}")

    # ¿Ya se importó? Reusar el experimento y su tabla de ciclos.
    exp_dir, meta = None, {}
    for j in sorted(entrada.glob("*/experimento.json")):
        try:
            m = json.loads(j.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (m.get("importado") or {}).get("de") == str(carpeta):
            exp_dir, meta = j.parent, m
            break
    if exp_dir is None:
        exp_dir = entrada / _nombre_libre(entrada, nombre or carpeta.name)
    tabla = dict((meta.get("importado") or {}).get("ciclos") or {})   # "cam/base" -> fecha

    usadas = {}
    for k, v in tabla.items():
        usadas.setdefault(k.split("/", 1)[0], set()).add(v)
    nuevos = 0
    for (cam, base), fotos in sorted(grupos.items()):
        clave = f"{cam}/{base}"
        if clave not in tabla:
            cuando = fecha_de(base) or datetime.fromtimestamp(
                min(p.stat().st_mtime for p in fotos.values())).replace(microsecond=0)
            while cuando.strftime("%Y%m%d_%H%M%S") in usadas.setdefault(cam, set()):
                cuando += timedelta(seconds=1)      # dos fotos del mismo segundo
            tabla[clave] = cuando.strftime("%Y%m%d_%H%M%S")
            usadas[cam].add(tabla[clave])
        destino = exp_dir / cam
        destino.mkdir(parents=True, exist_ok=True)
        for suf, origen in fotos.items():
            d = destino / f"img_{tabla[clave]}{suf}.tif"
            if not d.exists() and not d.is_symlink():
                _enlazar(origen, d)
                nuevos += 1

    capturados, guardados = _sufijos(grupos)
    meta.update({
        "modo": "dpc" if capturados != [""] else "blanco",
        "sufijos": capturados,
        "nombre": meta.get("nombre") or nombre or carpeta.name,
        "camaras": sorted({int(c[3:]) for c, _ in grupos}),
        "importado": {"de": str(carpeta), "cuando": datetime.now().isoformat(timespec="seconds"),
                      "ciclos": tabla},
    })
    if guardados:
        meta["dpc_procesado"] = dict(meta.get("dpc_procesado") or {}, sufijos=guardados)
    tmp = exp_dir / "experimento.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(exp_dir / "experimento.json")
    return {"experimento": exp_dir.name, "nuevos": nuevos, "ciclos": len(grupos),
            "sufijos": guardados or capturados, "modo": meta["modo"]}
