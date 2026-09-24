"""
PASO 2 del pipeline: tracking celular (seguimiento de cada célula entre frames).

01_segmentacion.py ya dice, para cada frame por separado, "acá hay una célula
en tal posición". Este script conecta esos puntos entre frames consecutivos
para armar trayectorias: decide que "la célula #7 del frame 3 es la misma
célula #12 del frame 4" según qué tan cerca está su centroide (centro de
masa). El resultado es una tabla donde cada fila es una detección y cada
trayectoria completa comparte un mismo `track_id`.

Las 1600 imágenes NO son una única secuencia continua: son 16 adquisiciones
("películas") independientes de 100 frames cada una, confirmado leyendo el
tag ImageDescription de ImageJ en los TIFF (finterval/min cambian de valor
exactamente cada 100 imágenes: 0100->0101, 0200->0201, ...). Trackear todo
junto como una sola secuencia conectaría por error la última célula de una
película con la primera de la siguiente si quedan cerca en píxeles, aunque
sean de experimentos distintos. Por eso acá se trackea cada película por
separado (frame 0..99 reinicia en cada una) y se combinan los resultados al
final con un track_id globalmente único.

Requiere haber corrido antes 01_segmentacion.py (usa los *_masks.npy generados).

Uso:
    ../venv/bin/python 02_tracking.py
    ../venv/bin/python 02_tracking.py --max-dist 50 --gap-frames 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from skimage.measure import regionprops_table
from laptrack import LapTrack
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MASKS_DIR = ROOT / "resultados" / "segmentacion"
DATASET_DIR = ROOT / "datasets" / "brightfield-mdamb231" / "1.1-training-source-BF-1600"
OUT_DIR = ROOT / "resultados" / "tracking"

# Ver docstring del módulo: verificado empíricamente en los metadatos TIFF.
FRAMES_PER_MOVIE = 100

# Separador para volver el track_id único entre películas: sin esto, el
# "track_id 0" que arma laptrack en la película 1 pisaría al "track_id 0" de
# la película 2 al concatenar todo en un solo CSV. Con este offset el
# track_id global queda como <película><id local de 5 cifras>, p.ej. la
# película 3, track local 42, queda 300042.
TRACK_ID_MOVIE_OFFSET = 100_000


def parse_args():
    p = argparse.ArgumentParser()
    # --max-dist: si una célula del frame N está a más de esta distancia (en
    # píxeles) de cualquier célula del frame N+1, laptrack no las conecta —
    # asume que son células distintas en vez de la misma célula que se movió.
    p.add_argument("--max-dist", type=float, default=50, help="Desplazamiento máximo entre frames consecutivos (px)")
    # "Gap closing" = si una célula desaparece un par de frames (oclusión,
    # falla de segmentación) y reaparece después, estos dos parámetros
    # controlan cuánto se le permite moverse y cuántos frames de ausencia se
    # toleran para igual reconectarla con su trayectoria original.
    p.add_argument("--gap-max-dist", type=float, default=75, help="Desplazamiento máximo al cerrar huecos (px)")
    p.add_argument("--gap-frames", type=int, default=2, help="Máximo de frames perdidos a puentear (gap closing)")
    # Trayectorias muy cortas suelen ser ruido de segmentación (una célula
    # detectada por error en 1-2 frames), no un movimiento real: se descartan.
    p.add_argument("--min-track-len", type=int, default=3, help="Descartar tracks más cortos que esto (frames)")
    return p.parse_args()


def load_regions(mask_paths):
    """Arma la tabla larga (una fila = una célula en un frame) para las
    máscaras dadas, con frame local 0..N-1 — pensado para llamarse con las
    máscaras de UNA sola película, no con el dataset completo."""
    rows = []
    for frame_idx, path in enumerate(mask_paths):
        masks = np.load(path)
        # regionprops_table analiza la máscara etiquetada (salida de Cellpose)
        # y devuelve, por cada célula (cada valor entero distinto de 0), su
        # centroide (posición y, x) y su área en píxeles — una fila por célula
        # detectada en este frame.
        props = regionprops_table(masks, properties=("label", "centroid", "area"))
        n = len(props["label"])
        for i in range(n):
            rows.append(
                {
                    "frame": frame_idx,
                    "file": path.stem.replace("_masks", ""),
                    "mask_label": props["label"][i],
                    "y": props["centroid-0"][i],
                    "x": props["centroid-1"][i],
                    "area": props["area"][i],
                }
            )
    # Esta tabla larga es el formato que espera LapTrack.predict_dataframe().
    return pd.DataFrame(rows)


def track_one_movie(mask_paths, args):
    """Corre laptrack sobre las máscaras de una sola película y devuelve el
    track_df filtrado (con frame local 0..99, sin numerar aún globalmente)."""
    df = load_regions(mask_paths)

    # LapTrack resuelve, frame por frame, un "problema de asignación lineal"
    # (LAP): dadas las células del frame N y las del frame N+1, encuentra el
    # emparejamiento que minimiza la distancia total recorrida, sujeto al
    # cutoff (no empareja si quedan más lejos que --max-dist). Se usa
    # distancia al cuadrado (sqeuclidean) porque es más barata de calcular
    # que la distancia real y da el mismo resultado para comparar cuál par es
    # más cercano.
    lt = LapTrack(
        metric="sqeuclidean",
        cutoff=args.max_dist ** 2,
        gap_closing_cutoff=args.gap_max_dist ** 2,
        gap_closing_max_frame_count=args.gap_frames,
    )
    # split_df/merge_df quedan sin usar: registrarían divisiones (mitosis) y
    # fusiones de células, pero esa detección está deshabilitada por defecto
    # en LapTrack (ver ESTADO.md, "Habilitar splitting" en ideas pendientes).
    track_df, split_df, merge_df = lt.predict_dataframe(
        df, coordinate_cols=["y", "x"], frame_col="frame", only_coordinate_cols=False
    )
    track_df = track_df.reset_index()

    # Filtro de calidad: una trayectoria de 1-2 frames casi siempre es ruido
    # (célula mal detectada un instante) y no un movimiento real medible, así
    # que se descartan antes de guardar/graficar.
    n_tracks_all = track_df["track_id"].nunique()
    track_lengths = track_df.groupby("track_id").size()
    keep_ids = track_lengths[track_lengths >= args.min_track_len].index
    track_df_filt = track_df[track_df["track_id"].isin(keep_ids)].copy()

    return df, track_df_filt, n_tracks_all


def plot_movie_trajectories(movie_num, first_file, track_df, args):
    # Visualización de control: dibuja todas las trayectorias de esta
    # película superpuestas sobre su primer frame, para inspeccionar a
    # simple vista si el tracking tiene sentido (trayectorias suaves y
    # contiguas) o si hay saltos/errores evidentes.
    fig, ax = plt.subplots(figsize=(10, 10))
    first_img = tifffile.imread(DATASET_DIR / f"{first_file}.tif")
    # Ajuste de contraste por percentiles (1-99) en vez de min/max: evita que
    # un par de píxeles extremos (ruido/saturación) aplasten el contraste de
    # toda la imagen.
    vmin, vmax = np.percentile(first_img, (1, 99))
    ax.imshow(first_img, cmap="gray", vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("tab20")
    for i, (track_id, g) in enumerate(track_df.groupby("track_id")):
        # Cada trayectoria se ordena por frame (no vienen necesariamente en
        # orden) y se dibuja como una línea; el punto marca dónde arrancó.
        g = g.sort_values("frame")
        ax.plot(g["x"], g["y"], "-", linewidth=1, color=cmap(i % 20), alpha=0.8)
        ax.plot(g["x"].iloc[0], g["y"].iloc[0], "o", color=cmap(i % 20), markersize=3)

    n_tracks = track_df["track_id"].nunique()
    ax.set_title(f"Película {movie_num:02d} — {n_tracks} tracks (>= {args.min_track_len} frames)")
    ax.axis("off")
    out_png = OUT_DIR / f"trayectorias_pelicula_{movie_num:02d}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # sorted() es clave: dentro de cada bloque de 100, garantiza el orden
    # 0001, 0002, ... — y los bloques de 100 consecutivos son exactamente las
    # 16 películas (ver docstring del módulo).
    mask_paths = sorted(MASKS_DIR.glob("*_masks.npy"))
    if not mask_paths:
        raise SystemExit(f"No hay máscaras en {MASKS_DIR}. Corré antes 01_segmentacion.py")
    if len(mask_paths) % FRAMES_PER_MOVIE != 0:
        raise SystemExit(
            f"{len(mask_paths)} máscaras no es múltiplo de {FRAMES_PER_MOVIE} "
            "(frames/película) -- revisar el dataset antes de trackear."
        )
    n_movies = len(mask_paths) // FRAMES_PER_MOVIE

    all_tracks = []
    summary_rows = []

    for movie_idx in tqdm(range(n_movies), unit="película"):
        movie_num = movie_idx + 1
        movie_paths = mask_paths[movie_idx * FRAMES_PER_MOVIE : (movie_idx + 1) * FRAMES_PER_MOVIE]

        df, track_df, n_tracks_all = track_one_movie(movie_paths, args)

        # Película (1-16) + track_id global único, para poder concatenar
        # todas las películas en un solo CSV sin que se pisen los IDs.
        track_df["movie"] = movie_num
        track_df["track_id"] = movie_num * TRACK_ID_MOVIE_OFFSET + track_df["track_id"]
        all_tracks.append(track_df)

        summary_rows.append(
            {
                "movie": movie_num,
                "celulas_detectadas": len(df),
                "tracks_totales": n_tracks_all,
                "tracks_filtrados": track_df["track_id"].nunique(),
            }
        )

        first_file = movie_paths[0].stem.replace("_masks", "")
        plot_movie_trajectories(movie_num, first_file, track_df, args)

    tracks_df = pd.concat(all_tracks, ignore_index=True)
    out_csv = OUT_DIR / "tracks.csv"
    tracks_df.to_csv(out_csv, index=False)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print(
        f"\nTotal: {summary_df['tracks_filtrados'].sum()} tracks "
        f"(>= {args.min_track_len} frames) en {n_movies} películas, "
        f"{summary_df['celulas_detectadas'].sum()} detecciones"
    )
    print(f"Tracks guardados en {out_csv}")
    print(f"Visualizaciones por película en {OUT_DIR}/trayectorias_pelicula_XX.png")


if __name__ == "__main__":
    main()
