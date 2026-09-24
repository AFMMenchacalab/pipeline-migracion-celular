"""
Detecciones (regionprops) y tracking (laptrack) genéricos para cualquier
dataset del pipeline v2.

Cambios respecto de 02_tracking.py (v1) y por qué (la elección final de
parámetros se justifica con la validación contra núcleos, 14_validacion_tracking.py):

- Umbrales en micrómetros, no en píxeles: el mismo criterio físico sirve
  para BF (0.65 um/px, 5 min) y para CAMAD (0.47 um/px, 30 s).
- Costo con tamaño opcional: además de la distancia entre centroides se
  puede penalizar el cambio de tamaño (w * (sqrt(A1) - sqrt(A2))^2). En
  zonas densas dos células vecinas pueden estar a la misma distancia del
  centroide, pero rara vez tienen el mismo tamaño; esto desempata sin
  recurrir a un costo en Python por par (OverLapTrack de laptrack calcula
  el IoU par por par con una función Python: con 600 células/frame son
  ~36 M llamadas por película, inviable acá).
- Mitosis / cambios de identidad: se probó el "splitting" de laptrack
  (asignar hijas a la madre por distancia) y NO es confiable acá: en la
  película 1 da 0, 7 o 74 divisiones según el umbral sea 5, 8 o 12 um (lo
  esperable para ~200 células en 8 h con un tiempo de duplicación de
  ~35 h es ~50). Sin un marcador de mitosis no hay forma de distinguir
  una hija de una vecina que pasa cerca. En cambio se cortan las
  trayectorias en SALTOS ANÓMALOS (cortar_saltos): un paso mucho más largo
  que lo típico de la película Y con un cambio brusco de área. Eso es lo
  que produce una mitosis mal asignada (una hija hereda la trayectoria de
  la madre con un salto de media célula) o un cambio de identidad entre
  vecinas; cortar ahí evita que ese salto espurio infle la velocidad y
  rompa la persistencia. Si ayuda o no se decide con la validación contra
  núcleos.
- Bandera de borde: una célula que toca el borde de la imagen tiene la
  máscara recortada, así que su centroide y su morfología están sesgados.
  Se marca para poder excluirla de la morfología.
"""
import numpy as np
import pandas as pd
from laptrack import LapTrack
from skimage.measure import regionprops_table

from .config import DATASETS
from .fuentes import listar_frames, cargar_mascara

PROPS = ("label", "centroid", "area", "perimeter", "eccentricity", "solidity",
         "major_axis_length", "minor_axis_length", "orientation", "bbox")


def detecciones_pelicula(dataset, movie):
    items = [it for it in listar_frames(dataset) if it["movie"] == movie]
    filas = []
    for it in items:
        m = cargar_mascara(dataset, movie, it["frame"])
        if m.max() == 0:
            continue
        p = regionprops_table(m, properties=PROPS)
        d = pd.DataFrame(p).rename(columns={"centroid-0": "y", "centroid-1": "x"})
        H, W = m.shape
        d["borde"] = (d["bbox-0"] == 0) | (d["bbox-1"] == 0) | (d["bbox-2"] == H) | (d["bbox-3"] == W)
        d = d.drop(columns=[c for c in d.columns if c.startswith("bbox")])
        d["frame"] = it["frame"]
        filas.append(d)
    df = pd.concat(filas, ignore_index=True)
    df["movie"] = movie
    amin = DATASETS[dataset].get("area_min_um2", 0) / DATASETS[dataset]["um_per_px"] ** 2
    return df[df["area"] >= amin].reset_index(drop=True)


def trackear(det, dataset, max_link_um=30.0, gap_frames=2, gap_link_um=None,
             size_weight=0.0, split_um=None, min_len=3):
    """det: detecciones de UNA película. Devuelve (tracks, divisiones)."""
    um = DATASETS[dataset]["um_per_px"]
    cutoff = (max_link_um / um) ** 2
    gap_cutoff = ((gap_link_um or max_link_um * 1.5) / um) ** 2
    det = det.copy()
    coords = ["y", "x"]
    if size_weight > 0:
        det["s"] = size_weight * np.sqrt(det["area"])
        coords = ["y", "x", "s"]
    kw = dict(metric="sqeuclidean", cutoff=cutoff,
              gap_closing_metric="sqeuclidean",
              gap_closing_cutoff=gap_cutoff if gap_frames > 0 else False,
              gap_closing_max_frame_count=max(gap_frames, 1))
    if split_um:
        kw.update(splitting_metric="sqeuclidean", splitting_cutoff=(split_um / um) ** 2)
    lt = LapTrack(**kw)
    tr, split_df, _ = lt.predict_dataframe(det, coordinate_cols=coords, frame_col="frame",
                                           only_coordinate_cols=False)
    tr = tr.reset_index(drop=True)
    largos = tr.groupby("track_id")["frame"].transform("size")
    tr = tr[largos >= min_len].copy()
    movie = int(det["movie"].iloc[0])
    off = movie * 1_000_000
    tr["track_id"] = tr["track_id"] + off
    tr["tree_id"] = tr["tree_id"] + off
    if len(split_df):
        split_df = split_df.copy()
        split_df["parent_track_id"] += off
        split_df["child_track_id"] += off
        split_df["movie"] = movie
    tr = tr.drop(columns=[c for c in ("s",) if c in tr.columns])
    return tr.sort_values(["track_id", "frame"]), split_df


def cortar_saltos(tr, dataset, q=0.99, factor_area=1.5, min_len=3):
    """Corta tracks donde el paso supera el percentil q de los pasos de la
    película Y el área cambia más de factor_area (en cualquier sentido).
    Cada pedazo recibe un track_id nuevo (mismo tree_id)."""
    tr = tr.sort_values(["track_id", "frame"]).copy()
    dx = tr.groupby("track_id")["x"].diff()
    dy = tr.groupby("track_id")["y"].diff()
    df = tr.groupby("track_id")["frame"].diff()
    paso = np.hypot(dx, dy) / df        # px por frame (normaliza huecos)
    ra = tr["area"] / tr.groupby("track_id")["area"].shift()
    umbral = np.nanquantile(paso, q)
    corte = (paso > umbral) & ((ra > factor_area) | (ra < 1 / factor_area))
    pieza = corte.groupby(tr["track_id"]).cumsum().astype(int)
    tr["track_id"] = tr["track_id"] * 100 + pieza
    largos = tr.groupby("track_id")["frame"].transform("size")
    return tr[largos >= min_len].copy(), int(corte.sum())


def a_micrometros(tr, dataset):
    """Agrega columnas físicas (x_um, y_um, t_s, area_um2, ...)."""
    cfg = DATASETS[dataset]
    um, dt = cfg["um_per_px"], cfg["dt_s"]
    tr = tr.copy()
    tr["x_um"] = tr["x"] * um
    tr["y_um"] = tr["y"] * um
    tr["t_s"] = tr["frame"] * dt
    tr["area_um2"] = tr["area"] * um ** 2
    tr["perimetro_um"] = tr["perimeter"] * um
    tr["aspect_ratio"] = tr["major_axis_length"] / tr["minor_axis_length"].replace(0, np.nan)
    tr["circularidad"] = 4 * np.pi * tr["area"] / tr["perimeter"].replace(0, np.nan) ** 2
    return tr
