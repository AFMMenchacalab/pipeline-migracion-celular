"""
Preparación de trayectorias para el análisis: huecos, remuestreo temporal y
velocidades.

Decisiones (y por qué):
- Huecos cortos (<= MAX_HUECO frames, los que dejó el gap-closing del
  tracking) se rellenan por interpolación lineal de la posición. Los
  análisis espectrales (FPS, wavelet) y los de ángulo a lag fijo (EAD)
  necesitan muestreo uniforme; tirar la trayectoria entera por un frame
  perdido sesgaría la muestra hacia células fáciles de seguir. Los huecos
  más largos cortan la trayectoria en segmentos independientes.
- Paso de análisis ("paso" en frames). Para CAMAD (30 s/frame) el
  desplazamiento en un frame (~0.2-0.5 um) es del orden del error de
  localización del centroide, así que las velocidades y sobre todo los
  ÁNGULOS entre pasos consecutivos quedarían dominados por ruido (la EAD
  daría "aleatorio" aunque la célula fuera persistente). Se analiza con
  paso de 5 min (10 frames), el mismo intervalo que el dataset BF: además
  así las entropías de ambos datasets son comparables entre sí (la EAD
  depende del intervalo de muestreo). Se incluye un análisis de
  sensibilidad al paso.
"""
import numpy as np
import pandas as pd

MAX_HUECO = 3


def segmentos_uniformes(tr, max_hueco=MAX_HUECO):
    """tr: filas de UN track (frame, x_um, y_um, ...). Devuelve lista de
    arrays (frames, xy) sin huecos, interpolando huecos <= max_hueco."""
    f = tr["frame"].to_numpy()
    xy = tr[["x_um", "y_um"]].to_numpy(float)
    orden = np.argsort(f)
    f, xy = f[orden], xy[orden]
    cortes = np.nonzero(np.diff(f) > max_hueco + 1)[0] + 1
    segs = []
    for fs, ps in zip(np.split(f, cortes), np.split(xy, cortes)):
        if len(fs) < 2:
            continue
        full = np.arange(fs[0], fs[-1] + 1)
        x = np.interp(full, fs, ps[:, 0])
        y = np.interp(full, fs, ps[:, 1])
        segs.append((full, np.column_stack([x, y])))
    return segs


def remuestrear(frames, xy, paso):
    """Submuestreo cada `paso` frames desde el inicio del segmento."""
    return frames[::paso], xy[::paso]


def tabla_segmentos(tracks, paso=1, min_puntos=4, max_hueco=MAX_HUECO):
    """Devuelve lista de dicts {movie, track_id, seg, frames, xy} ya
    remuestreados al paso de análisis."""
    out = []
    for (mv, tid), g in tracks.groupby(["movie", "track_id"], sort=False):
        for k, (fr, xy) in enumerate(segmentos_uniformes(g, max_hueco)):
            fr, xy = remuestrear(fr, xy, paso)
            if len(fr) >= min_puntos:
                out.append({"movie": mv, "track_id": tid, "seg": k, "frames": fr, "xy": xy})
    return out


def velocidades(xy, dt):
    return np.diff(xy, axis=0) / dt
