"""
PASO 1 del pipeline: segmentación de células.

Para cada imagen de microscopía (TIFF en escala de grises), Cellpose predice
una "máscara": una imagen del mismo tamaño donde cada célula detectada queda
pintada con un número entero distinto (0 = fondo). Esa máscara es la entrada
que después usa 02_tracking.py para seguir a cada célula de un frame al
siguiente.

Por cada imagen de entrada se generan dos archivos en resultados/segmentacion/:
    <nombre>_masks.npy   -> la máscara (array de enteros) para el tracking
    <nombre>_overlay.png -> imagen de control visual con los contornos dibujados

Paraleliza el procesamiento entre varias imágenes usando multiprocessing: cada
worker carga su propia copia del modelo y limita sus threads internos de
PyTorch para no sobre-suscribir los núcleos (workers * threads-per-worker ~= núcleos).
Con --gpu se usa un solo proceso en vez de varios (ver comentario en main()).

Uso:
    ../venv/bin/python 01_segmentacion.py --n 50                       # 50 imágenes, auto workers
    ../venv/bin/python 01_segmentacion.py --n -1 --workers 8           # todas, 8 procesos
    ../venv/bin/python 01_segmentacion.py --n 50 --workers 1           # modo secuencial (debug)
    ../venv/bin/python 01_segmentacion.py --n -1 --gpu                 # todas, en GPU (ROCm/CUDA)
"""
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import tifffile
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "datasets" / "brightfield-mdamb231" / "1.1-training-source-BF-1600"
OUT_DIR = ROOT / "resultados" / "segmentacion"

_model = None
_args = None


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5, help="Número de imágenes a procesar (-1 = todas)")
    p.add_argument("--gpu", action="store_true", help="Usar GPU si está disponible")
    p.add_argument("--flow-threshold", type=float, default=0.4)
    p.add_argument("--cellprob-threshold", type=float, default=0.0)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Procesos paralelos (0 = automático según núcleos disponibles, 1 = secuencial)",
    )
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=0,
        help="Threads de PyTorch por worker (0 = automático)",
    )
    return p.parse_args()


def _init_worker(args, threads_per_worker):
    # Se ejecuta UNA vez por proceso worker (no una vez por imagen), gracias
    # a `initializer=` de ProcessPoolExecutor. Así el modelo de Cellpose
    # (~1.15 GB de pesos) se carga una sola vez por proceso y se reutiliza
    # para todas las imágenes que le toquen, en vez de recargarlo por imagen.
    #
    # Fijar las variables de entorno ANTES de importar torch/BLAS: si se llama
    # torch.set_num_threads() después de importar, OpenMP/MKL/OpenBLAS ya
    # fijaron su propio pool de threads al importar y lo ignoran, causando
    # sobre-suscripción (N workers x todos los núcleos cada uno).
    t = str(max(1, threads_per_worker))
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t

    import torch
    from cellpose import models

    torch.set_num_threads(max(1, threads_per_worker))
    global _model, _args
    _args = args
    _model = models.CellposeModel(gpu=args.gpu)


def _process_one(path_str):
    # matplotlib se importa acá adentro (no al tope del archivo) porque cada
    # worker es un proceso nuevo con su propio backend gráfico: "Agg" es un
    # backend sin ventana (solo dibuja a archivo), necesario porque estos
    # procesos corren en paralelo sin entorno gráfico.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cellpose import plot as cp_plot

    path = Path(path_str)
    img = tifffile.imread(path)

    # Acá ocurre la segmentación en sí: la red neuronal de Cellpose predice,
    # para cada píxel, hacia dónde "fluye" (flows) el centro de la célula a
    # la que pertenece; luego Cellpose sigue esos flujos y agrupa píxeles que
    # convergen al mismo punto en una única célula (masks). flow_threshold y
    # cellprob_threshold controlan qué tan estricto es al aceptar una célula
    # como válida (más alto = más exigente = menos falsos positivos, pero
    # también puede perder células reales tenues).
    masks, flows, styles = _model.eval(
        img,
        flow_threshold=_args.flow_threshold,
        cellprob_threshold=_args.cellprob_threshold,
    )

    # Cada célula en `masks` está pintada con un entero 1..N (0 = fondo), así
    # que el valor máximo del array es directamente la cantidad de células
    # detectadas en esta imagen.
    n_cells = int(masks.max())
    stem = path.stem
    # .npy guarda el array de máscaras tal cual, sin pérdida ni compresión con
    # pérdida (a diferencia de un PNG/JPG) — es lo que va a leer 02_tracking.py.
    np.save(OUT_DIR / f"{stem}_masks.npy", masks)

    # Overlay = imagen original + contorno de cada célula detectada dibujado
    # encima. Sirve solo para inspección visual humana (control de calidad de
    # la segmentación), no lo usa ningún script posterior.
    fig = plt.figure(figsize=(12, 4))
    cp_plot.show_segmentation(fig, img, masks, flows[0])
    fig.savefig(OUT_DIR / f"{stem}_overlay.png", dpi=150)
    plt.close(fig)

    return path.name, n_cells


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # sorted() es clave: garantiza que las imágenes se procesen en orden
    # (0001.tif, 0002.tif, ...). 02_tracking.py depende de ese mismo orden
    # para saber qué máscara corresponde a qué frame.
    paths = sorted(DATASET_DIR.glob("*.tif"))
    if args.n != -1:
        paths = paths[: args.n]

    n_cores = os.cpu_count() or 4
    if args.gpu and args.workers == 0:
        # Un solo proceso: N procesos cargando cada uno su propio modelo en
        # la misma GPU compiten por VRAM y por el motor de cómputo sin
        # ganar nada (a diferencia de CPU, acá no hay núcleos libres que
        # repartir entre workers).
        workers = 1
        threads_per_worker = args.threads_per_worker if args.threads_per_worker > 0 else n_cores
    else:
        # Validado empíricamente en un AMD Ryzen 12c/24t: 6 workers x 4 threads
        # (~n_cores/4) rinde mejor que más workers con menos threads cada uno,
        # porque la inferencia de la CNN está limitada por ancho de banda de
        # memoria, no solo por núcleos libres.
        workers = args.workers if args.workers > 0 else max(1, min(len(paths), n_cores // 4))
        threads_per_worker = args.threads_per_worker if args.threads_per_worker > 0 else max(1, n_cores // workers)

    print(
        f"Procesando {len(paths)} imágenes de {DATASET_DIR} "
        f"con {workers} workers x {threads_per_worker} threads (núcleos detectados: {n_cores})"
    )

    total_cells = 0
    path_strs = [str(p) for p in paths]

    # Reparte las imágenes entre los `workers` procesos (cada uno con su
    # modelo ya cargado por _init_worker). as_completed() va entregando cada
    # resultado apenas termina, sin importar el orden en que terminen los
    # procesos, por eso la barra de progreso de abajo avanza imagen por
    # imagen a medida que se van completando (no en el orden original).
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(args, threads_per_worker),
    ) as executor:
        futures = [executor.submit(_process_one, p) for p in path_strs]
        # tqdm dibuja la barra de progreso con velocidad (img/s) y ETA en
        # base a cuántos futures ya se resolvieron sobre el total.
        pbar = tqdm(as_completed(futures), total=len(futures), unit="img")
        for fut in pbar:
            name, n_cells = fut.result()
            total_cells += n_cells
            pbar.set_postfix(archivo=name, celulas=n_cells)

    print(f"Resultados guardados en {OUT_DIR}")
    if paths:
        print(f"Promedio: {total_cells / len(paths):.1f} células/imagen sobre {len(paths)} imágenes")


if __name__ == "__main__":
    main()
