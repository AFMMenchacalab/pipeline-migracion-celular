"""
PASO 1 (v2): segmentación con Cellpose-SAM en GPU, genérica para los tres
datasets (bf, sirdna, camad).

Cambios respecto de 01_segmentacion.py (v1) y por qué:

1. fp16 en vez de bf16. Cellpose 4 carga la red en bfloat16 por defecto,
   pero la RX 6800 XT (RDNA2, gfx1030) no tiene unidades nativas de bf16 y
   lo emula: 4.1 s/imagen. En fp32 baja a 2.8 s y en fp16 (que RDNA2 sí
   acelera, "packed math") a ~1.7 s. Se verificó que las máscaras fp16 son
   las mismas que fp32: AP@IoU0.5 = 0.995-1.000 en 3 imágenes de prueba
   (0050, 0777 y 1450, esta última la película densa). Resultado: la
   segmentación completa de BF pasa de 2 h a ~45 min.

2. Se guardan también los flows (dP, cellprob) en float16 comprimido. Los
   umbrales flow_threshold / cellprob_threshold SOLO afectan el
   post-procesamiento (la dinámica que agrupa píxeles en células, ~0.1 s en
   GPU), no la red (~1.7 s). Guardando los flows, el ajuste de umbrales
   (12_ajuste_segmentacion.py) se hace sin volver a correr la red, y la
   segmentación final con los umbrales elegidos también.

3. Lectura y escritura en hilos aparte (prefetch/escritura asíncrona): la
   GPU no espera al disco ni a la compresión de los .npz.

4. Salida por frame en ~/microscopio_cache/masks/<dataset>/mMM/fFFF.npz
   (uint16 comprimido, ~100 kB en vez de 4 MB por .npy int32), fuera de la
   carpeta de Syncthing. Es reanudable: si se corta, al relanzarlo saltea
   los frames ya hechos.

5. Ya no se genera un overlay PNG por frame (v1 generaba 1600, ~3 GB): los
   controles visuales se hacen sobre una muestra en los scripts de
   validación.

Uso:
    ../venv/bin/python 11_segmentar_gpu.py --dataset bf
    ../venv/bin/python 11_segmentar_gpu.py --dataset camad --no-flows
"""
import argparse
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from lib.config import cache_dir  # noqa: E402
from lib.fuentes import listar_frames, leer_frame, ruta_mascara  # noqa: E402


def cargar_modelo(fp16=True):
    import torch
    from cellpose import models
    m = models.CellposeModel(gpu=True, use_bfloat16=False)
    if fp16:
        m.net.half()
        m.net.dtype = torch.float16
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["bf", "sirdna", "camad"])
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--no-flows", action="store_true", help="No guardar flows (ahorra disco)")
    ap.add_argument("--flows-every", type=int, default=1, help="Guardar flows solo cada N frames")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--movies", type=str, default="", help="Lista de películas, ej. 1,2,3")
    args = ap.parse_args()

    items = listar_frames(args.dataset)
    if args.movies:
        keep = {int(x) for x in args.movies.split(",")}
        items = [it for it in items if it["movie"] in keep]
    pendientes = [it for it in items if not ruta_mascara(args.dataset, it["movie"], it["frame"]).exists()]
    print(f"[{args.dataset}] {len(items)} frames, {len(pendientes)} pendientes", flush=True)
    if not pendientes:
        return

    model = cargar_modelo()
    flows_dir = cache_dir("flows", args.dataset)

    # prefetch de imágenes en un hilo: la lectura (TIFF o mmap del stack)
    # se solapa con la inferencia de la imagen anterior
    q_in = queue.Queue(maxsize=16)

    def lector():
        for it in pendientes:
            q_in.put((it, leer_frame(it)))
        q_in.put(None)

    threading.Thread(target=lector, daemon=True).start()

    def escribir(it, masks, dP, cellprob):
        np.savez_compressed(ruta_mascara(args.dataset, it["movie"], it["frame"]),
                            masks=masks.astype(np.uint16))
        if dP is not None:
            d = flows_dir / f"m{it['movie']:02d}"
            d.mkdir(exist_ok=True)
            np.savez_compressed(d / f"f{it['frame']:03d}.npz",
                                dP=dP.astype(np.float16), cellprob=cellprob.astype(np.float16))

    pool = ThreadPoolExecutor(max_workers=6)
    t0 = time.time()
    n = 0
    n_cel = 0
    futs = []
    while True:
        x = q_in.get()
        if x is None:
            break
        it, img = x
        masks, flows, _ = model.eval(img, batch_size=args.batch_size,
                                     flow_threshold=args.flow_threshold,
                                     cellprob_threshold=args.cellprob_threshold)
        guardar_flows = (not args.no_flows) and (it["frame"] % args.flows_every == 0)
        futs.append(pool.submit(escribir, it, masks,
                                flows[1] if guardar_flows else None,
                                flows[2] if guardar_flows else None))
        n += 1
        n_cel += int(masks.max())
        if n % 50 == 0 or n == len(pendientes):
            el = time.time() - t0
            print(f"[{args.dataset}] {n}/{len(pendientes)}  {el / n:.2f} s/img  "
                  f"ETA {(len(pendientes) - n) * el / n / 60:.1f} min  "
                  f"({n_cel / n:.1f} células/img)", flush=True)
        # evitar que la cola de escritura crezca sin límite
        if len(futs) > 64:
            futs = [f for f in futs if not f.done()]
    for f in futs:
        f.result()
    pool.shutdown()
    print(f"[{args.dataset}] LISTO en {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
