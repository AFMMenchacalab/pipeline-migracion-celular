"""
Iteradores de frames comunes a los tres datasets: cada frame se identifica
por (película, frame local 0..N-1) y se lee siempre de la misma forma, para
que la segmentación, el tracking y la validación no tengan que saber de qué
dataset vienen.
"""
from pathlib import Path

import numpy as np
import tifffile

from .config import DATASETS, cache_dir


def listar_frames(dataset):
    """Lista de dicts {movie, frame, key, loader} en orden temporal.

    bf / sirdna: 1600 TIFF en orden alfabético = 16 películas x 100 frames
    (límites de película verificados en los metadatos, ver 02_tracking.py v1).
    camad: un stack .npy (T, H, W) por experimento, generado por
    10_preparar_camad.py; los frames negros (relleno del video) ya vienen
    marcados en camad_frames/expNN_validos.npy y se saltean acá.
    """
    cfg = DATASETS[dataset]
    out = []
    if dataset in ("bf", "sirdna"):
        paths = sorted(Path(cfg["src"]).glob(cfg["pattern"]), key=lambda p: p.name)
        fpm = cfg["frames_per_movie"]
        for i, p in enumerate(paths):
            out.append({"movie": i // fpm + 1, "frame": i % fpm, "key": p.stem,
                        "path": str(p)})
    elif dataset == "camad":
        src = Path(cfg["src"])
        for stack_path in sorted(src.glob("exp*.npy")):
            if stack_path.stem.endswith("_validos"):
                continue
            exp = int(stack_path.stem[3:])
            validos = np.load(src / f"{stack_path.stem}_validos.npy")
            for t in np.nonzero(validos)[0]:
                out.append({"movie": exp, "frame": int(t), "key": f"exp{exp:02d}_{t:03d}",
                            "path": str(stack_path)})
    else:
        raise ValueError(dataset)
    return out


_stack_cache = {}


def leer_frame(item):
    p = item["path"]
    if p.endswith(".npy"):
        if p not in _stack_cache:
            _stack_cache.clear()
            _stack_cache[p] = np.load(p, mmap_mode="r")
        return np.asarray(_stack_cache[p][item["frame"]])
    return tifffile.imread(p)


def ruta_mascara(dataset, movie, frame):
    return cache_dir("masks", dataset, f"m{movie:02d}") / f"f{frame:03d}.npz"


def cargar_mascara(dataset, movie, frame):
    with np.load(ruta_mascara(dataset, movie, frame)) as z:
        return z["masks"]
