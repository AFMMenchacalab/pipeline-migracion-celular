"""
PASO 0 (CAMAD): decodificar los 16 AVI de CAMAD a stacks de numpy listos
para segmentar.

Datos del paper del dataset (Iheme et al. 2024, IEEE Data Descriptions,
doi 10.1109/IEEEDATA.2024.3481394):
  - 600 frames por experimento, 5 h de observación real -> 30 s/frame (el
    script oficial de extracción usa "frames_per_minute = 2", consistente).
  - Contraste de fase, objetivo 40x, 0.117 um/px, 2568x1912 px.
  - Las células se sembraron en L15 + 0.35% BSA sin suero, así que las 5 h
    cubren adhesión + esparcimiento + migración temprana.

Decisiones:
  - Reducción 4x por promedio de área (ffmpeg scale flags=area) -> 642x478,
    0.468 um/px. Ver lib/config.py (CAMAD_DOWNSAMPLE) para el porqué.
  - Frames negros: algunos videos traen "blackaddedfor30sec" en el nombre
    (se rellenó con negro para completar 30 s de video a 20 fps). Un frame
    cuya intensidad media es < 5/255 se marca como inválido y no se
    segmenta; el tracking lo trata como un hueco real en el tiempo (el
    índice de frame se conserva, así que dt sigue siendo correcto).
  - Se decodifican varios videos en paralelo (un ffmpeg por video).

Salida: ~/microscopio_cache/camad_frames/expNN.npy  (600, 478, 642) uint8
        ~/microscopio_cache/camad_frames/expNN_validos.npy  (600,) bool
        resultados/v2/camad/frames_validos.csv
"""
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, CAMAD_DOWNSAMPLE, CAMAD_CONDICION, cache_dir, res_dir  # noqa: E402

W_FULL, H_FULL = 2568, 1912
W, H = W_FULL // CAMAD_DOWNSAMPLE, H_FULL // CAMAD_DOWNSAMPLE
UMBRAL_NEGRO = 5.0


def procesar(video):
    video = Path(video)
    exp = int(video.name.split("_")[0][3:])
    out = cache_dir("camad_frames")
    dest = out / f"exp{exp:02d}.npy"
    if dest.exists() and (out / f"exp{exp:02d}_validos.npy").exists():
        stack = np.load(dest, mmap_mode="r")
    else:
        cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", str(video),
               "-vf", f"scale={W}:{H}:flags=area,format=gray",
               "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        stack = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
        np.save(dest, stack)
    medias = stack.reshape(len(stack), -1).mean(axis=1)
    validos = medias >= UMBRAL_NEGRO
    np.save(out / f"exp{exp:02d}_validos.npy", validos)
    # frames repetidos exactos (congelados) también se reportan
    dif = np.abs(np.diff(stack.astype(np.int16), axis=0)).reshape(len(stack) - 1, -1).mean(axis=1)
    return {"exp": exp, "condicion": CAMAD_CONDICION[exp], "video": video.name,
            "n_frames": len(stack), "n_validos": int(validos.sum()),
            "primer_invalido": int(np.argmin(validos)) if not validos.all() else -1,
            "n_frames_identicos_al_anterior": int((dif == 0).sum()),
            "intensidad_media": float(medias[validos].mean())}


def main():
    videos = sorted(Path(DATASETS["camad"]["videos"]).glob("*.avi"))
    with ProcessPoolExecutor(max_workers=8) as ex:
        filas = list(ex.map(procesar, videos))
    df = pd.DataFrame(filas).sort_values("exp")
    df.to_csv(res_dir("camad") / "frames_validos.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
