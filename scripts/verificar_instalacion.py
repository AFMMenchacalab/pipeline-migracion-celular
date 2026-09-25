"""
Verifica que la instalación funcione (lo corren instalar.sh / instalar.ps1).

1. Muestra Python, PyTorch y la GPU que va a usar Cellpose (CUDA, ROCm o CPU).
2. Carga Cellpose-SAM; la primera vez descarga los pesos (~1.2 GB) a
   ~/.cellpose/models.
3. Segmenta una imagen sintética con células conocidas y comprueba que las
   encuentra, midiendo el tiempo.
4. Prueba el tracking (laptrack) con dos puntos que se mueven.

Si algo falla, termina con código 1 y un mensaje que dice qué revisar.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    print(f"Python {sys.version.split()[0]}")
    try:
        import torch
    except ImportError:
        sys.exit("ERROR: PyTorch no está instalado (volver a correr el instalador).")
    from lib import en_vivo as EV
    d = EV.dispositivo()
    print(f"PyTorch {d.get('torch')}  ->  {d['backend']}: {d['nombre']}")
    if d["backend"] == "CPU":
        print("AVISO: sin GPU la segmentación tarda ~1 min por imagen. Si la PC tiene NVIDIA, "
              "revisar que el driver esté instalado (nvidia-smi) y reinstalar con --cuda.")

    print("Cargando Cellpose-SAM (la primera vez descarga el modelo)…", flush=True)
    t0 = time.time()
    model = EV.cargar_modelo()
    print(f"  modelo listo en {time.time() - t0:.0f} s")

    # imagen sintética: 12 "células" (discos claros con borde oscuro) sobre fondo con ruido
    rng = np.random.default_rng(0)
    H = W = 512
    yy, xx = np.mgrid[:H, :W]
    img = np.full((H, W), 100.0) + rng.normal(0, 3, (H, W))
    centros = [(80 + 120 * (i % 4), 90 + 150 * (i // 4)) for i in range(12)]
    for cy, cx in centros:
        r = np.hypot(yy - cy, xx - cx)
        img += 60 * (r < 22) - 40 * ((r >= 22) & (r < 26))
    t0 = time.time()
    masks, _, _ = model.eval(img.astype(np.float32), batch_size=8)
    dt = time.time() - t0
    n = int(masks.max())
    print(f"Segmentación de prueba: {n} células encontradas (hay 12) en {dt:.1f} s")
    if not 9 <= n <= 15:
        sys.exit("ERROR: la segmentación de prueba no encontró las células esperadas.")

    import pandas as pd
    from laptrack import LapTrack
    det = pd.DataFrame({"frame": [0, 0, 1, 1, 2, 2], "y": [10, 50, 11, 51, 12, 52.0],
                        "x": [10, 50, 12, 49, 14, 48.0]})
    tr, _, _ = LapTrack(cutoff=25).predict_dataframe(det, coordinate_cols=["y", "x"], frame_col="frame",
                                                     only_coordinate_cols=False)
    if tr["track_id"].nunique() != 2:
        sys.exit("ERROR: el tracking de prueba no dio las 2 trayectorias esperadas.")
    print("Tracking de prueba: 2 trayectorias, correcto")
    print("\nTODO BIEN.")


if __name__ == "__main__":
    main()
