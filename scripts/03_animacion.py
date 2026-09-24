"""
PASO 3 del pipeline: animación (GIF) de cada película, célula por célula.

Dibuja, para cada frame, la imagen original con las máscaras de Cellpose
coloreadas por track_id (mismo color = misma célula en el tiempo) y una
estela con las últimas posiciones de cada trayectoria — sirve para
inspeccionar visualmente si el tracking sigue bien a las células o si hay
saltos/errores.

Genera UN GIF POR PELÍCULA (16 en total), no uno solo para las 1600 imágenes:
02_tracking.py trackea cada película por separado y el `frame` que guarda en
tracks.csv es local a cada película (0-99, se repite en cada una), así que
mezclar todo en una sola animación pegaría el final de una película con el
arranque de la siguiente y colorearía mal las células (ver docstring de
02_tracking.py para el porqué de la separación por película).

Requiere haber corrido antes 01_segmentacion.py y 02_tracking.py.

Uso:
    ../venv/bin/python 03_animacion.py
    ../venv/bin/python 03_animacion.py --fps 4 --tail 10
    ../venv/bin/python 03_animacion.py --movies 1 15     # solo esas películas
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
MASKS_DIR = ROOT / "resultados" / "segmentacion"
DATASET_DIR = ROOT / "datasets" / "brightfield-mdamb231" / "1.1-training-source-BF-1600"
TRACKS_CSV = ROOT / "resultados" / "tracking" / "tracks.csv"
OUT_DIR = ROOT / "resultados" / "tracking"

# Mismo valor y misma justificación que en 02_tracking.py (metadatos TIFF).
FRAMES_PER_MOVIE = 100


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=float, default=5, help="Cuadros por segundo del GIF")
    p.add_argument("--tail", type=int, default=8, help="Cuántos frames de estela mostrar por célula")
    p.add_argument("--minutes-per-frame", type=float, default=5, help="Intervalo real entre frames (min)")
    p.add_argument(
        "--movies",
        type=int,
        nargs="+",
        default=None,
        help="Números de película a animar (1-16). Por defecto, todas.",
    )
    return p.parse_args()


def animate_movie(movie_num, movie_paths, movie_tracks_df, args, pbar):
    # (frame local, mask_label) -> track_id, solo para ESTA película: como ya
    # viene filtrado por movie, no hay colisión con los mask_label (que
    # arrancan en 1 en cada imagen) de otras películas.
    label_to_track = {
        (int(r.frame), int(r.mask_label)): int(r.track_id) for r in movie_tracks_df.itertuples()
    }
    cmap = plt.get_cmap("tab20")
    n_frames = len(movie_paths)

    frames_rgb = []
    for frame_idx, mask_path in enumerate(movie_paths):
        fstem = mask_path.stem.replace("_masks", "")
        img = tifffile.imread(DATASET_DIR / f"{fstem}.tif")
        masks = np.load(mask_path)

        vmin, vmax = np.percentile(img, (1, 99))
        img_norm = np.clip((img.astype(float) - vmin) / (vmax - vmin), 0, 1)
        rgb = np.stack([img_norm] * 3, axis=-1)

        overlay = rgb.copy()
        for label_val in np.unique(masks):
            if label_val == 0:
                continue
            track_id = label_to_track.get((frame_idx, int(label_val)))
            if track_id is None:
                continue  # célula filtrada (track corto) o sin trackear
            color = np.array(cmap(track_id % 20)[:3])
            region = masks == label_val
            overlay[region] = 0.45 * rgb[region] + 0.55 * color

        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(overlay)

        past = movie_tracks_df[movie_tracks_df["frame"] <= frame_idx]
        for track_id, g in past.groupby("track_id"):
            g = g.sort_values("frame").tail(args.tail)
            if len(g) >= 2:
                ax.plot(g["x"], g["y"], "-", linewidth=1, color=cmap(track_id % 20), alpha=0.9)

        t_min = frame_idx * args.minutes_per_frame
        ax.set_title(f"Película {movie_num:02d} — t = {t_min:.0f} min (frame {frame_idx + 1}/{n_frames})")
        ax.axis("off")
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames_rgb.append(buf)
        plt.close(fig)
        pbar.update(1)

    out_gif = OUT_DIR / f"animacion_pelicula_{movie_num:02d}.gif"
    imageio.mimsave(out_gif, frames_rgb, duration=1000 / args.fps, loop=0)
    return out_gif


def main():
    args = parse_args()
    tracks_df = pd.read_csv(TRACKS_CSV)

    # derivar el orden de frames directamente de las máscaras (igual que en
    # 02_tracking.py); no usar la columna "file" del CSV porque pandas la
    # castea a int y pierde los ceros a la izquierda del nombre de archivo
    mask_paths = sorted(MASKS_DIR.glob("*_masks.npy"))
    if len(mask_paths) % FRAMES_PER_MOVIE != 0:
        raise SystemExit(
            f"{len(mask_paths)} máscaras no es múltiplo de {FRAMES_PER_MOVIE} (frames/película)."
        )
    n_movies = len(mask_paths) // FRAMES_PER_MOVIE

    movie_nums = args.movies if args.movies else list(range(1, n_movies + 1))
    total_frames = len(movie_nums) * FRAMES_PER_MOVIE

    out_gifs = []
    with tqdm(total=total_frames, unit="frame") as pbar:
        for movie_num in movie_nums:
            movie_idx = movie_num - 1
            movie_paths = mask_paths[movie_idx * FRAMES_PER_MOVIE : (movie_idx + 1) * FRAMES_PER_MOVIE]
            movie_tracks_df = tracks_df[tracks_df["movie"] == movie_num]
            pbar.set_description(f"película {movie_num:02d}")
            out_gif = animate_movie(movie_num, movie_paths, movie_tracks_df, args, pbar)
            out_gifs.append(out_gif)

    for g in out_gifs:
        print(f"Animación guardada en {g}")


if __name__ == "__main__":
    main()
