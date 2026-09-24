"""
Ajuste y validación de la segmentación brightfield usando los NÚCLEOS
(SiR-DNA) del mismo campo como referencia independiente.

Por qué esto es nuevo e importante: en v1 no había ninguna referencia para
el dataset brightfield (LFCT, el único con ground truth, no se pudo
descargar). El mismo registro de Zenodo (10074471) trae, para cada una de
las 1600 imágenes BF, la imagen de fluorescencia de núcleos (SiR-DNA)
adquirida en el mismo instante y campo. Un núcleo = una célula, así que
cruzando máscaras BF con núcleos se obtiene, sin anotar nada a mano:
  - células BF con 1 núcleo (bien segmentadas),
  - células BF sin núcleo (basura/halo),
  - células BF con >= 2 núcleos (fusiones = sub-segmentación),
  - núcleos sin célula BF (células perdidas).
Esto sirve para validar y también para elegir cellprob_threshold y
flow_threshold (se barren sobre los flows guardados por 11_segmentar_gpu.py,
sin volver a correr la red).

Evaluación honesta: leave-one-movie-out (se eligen los umbrales con 15
películas y se evalúan en la restante).

Aviso: la segmentación de núcleos también tiene errores (núcleos en
mitosis, núcleos tenues fuera de foco). Las métricas son "contra núcleos
segmentados automáticamente", no contra anotación humana; se inspecciona
una muestra visual (figura de salida) para confirmar que la referencia es
razonable.

Salidas: resultados/v2/validacion_seg_bf/{grilla.csv, loeo.csv, eleccion.json,
         por_frame_elegido.csv, ejemplo_*.png}
Con --aplicar: recalcula TODAS las máscaras BF con los umbrales elegidos.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import cache_dir, res_dir  # noqa: E402
from lib.fuentes import listar_frames, cargar_mascara, ruta_mascara, leer_frame  # noqa: E402
from lib.metricas_seg import metricas_vs_nucleos  # noqa: E402

CELLPROB = [-1.0, 0.0, 1.0]
FLOW = [0.4, 0.8, 0.0]
CADA = 10  # frames de la grilla: 1 de cada 10 (10 por película, 160 en total)


def cargar_flows(movie, frame):
    with np.load(cache_dir("flows", "bf", f"m{movie:02d}") / f"f{frame:03d}.npz") as z:
        return z["dP"].astype(np.float32), z["cellprob"].astype(np.float32)


def figura_ejemplo(movie, frame, masks_bf, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.segmentation import find_boundaries
    it_bf = [i for i in listar_frames("bf") if i["movie"] == movie and i["frame"] == frame][0]
    it_nu = [i for i in listar_frames("sirdna") if i["movie"] == movie and i["frame"] == frame][0]
    bf = leer_frame(it_bf).astype(float)
    nu = leer_frame(it_nu).astype(float)
    nm = cargar_mascara("sirdna", movie, frame)
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    for ax, im, t in ((axs[0], bf, "Brightfield + contornos Cellpose (BF)"),
                      (axs[1], nu, "SiR-DNA + contornos de núcleos")):
        lo, hi = np.percentile(im, (1, 99.5))
        ax.imshow(im, cmap="gray", vmin=lo, vmax=hi)
        ax.axis("off")
        ax.set_title(t)
    b = find_boundaries(masks_bf)
    axs[0].imshow(np.ma.masked_where(~b, b), cmap="autumn", alpha=0.9)
    bn = find_boundaries(nm)
    axs[1].imshow(np.ma.masked_where(~bn, bn), cmap="cool", alpha=0.9)
    # centroides de núcleos también sobre BF
    from skimage.measure import regionprops
    for r in regionprops(nm):
        axs[0].plot(r.centroid[1], r.centroid[0], ".", color="cyan", ms=3)
    fig.suptitle(f"Película {movie}, frame {frame}: puntos cian = núcleos SiR-DNA")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aplicar", action="store_true")
    args = ap.parse_args()
    import torch
    from cellpose import dynamics
    dev = torch.device("cuda")
    out = res_dir("validacion_seg_bf")

    pool = ThreadPoolExecutor(max_workers=12)
    if (out / "grilla.csv").exists() and "--rehacer" not in sys.argv:
        # la grilla ya se calculó (solo depende de los flows): se reutiliza
        filas = pd.read_csv(out / "grilla.csv").to_dict("records")
    else:
        items = [it for it in listar_frames("bf") if it["frame"] % CADA == 0]
        filas = []
        t0 = time.time()
        for n, it in enumerate(items):
            dP, cp = cargar_flows(it["movie"], it["frame"])
            nuc = cargar_mascara("sirdna", it["movie"], it["frame"])
            futs = []
            for cpt in CELLPROB:
                for ft in FLOW:
                    m = dynamics.compute_masks(dP, cp, niter=200, cellprob_threshold=cpt,
                                               flow_threshold=ft, min_size=15, device=dev)
                    futs.append((cpt, ft, pool.submit(metricas_vs_nucleos, m, nuc)))
            for cpt, ft, f in futs:
                r = f.result()
                r.update(movie=it["movie"], frame=it["frame"], cellprob=cpt, flow=ft)
                filas.append(r)
            if n % 40 == 0:
                print(f"{n}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(filas)
    df.to_csv(out / "grilla.csv", index=False)

    params = ["cellprob", "flow"]
    # sumar conteos por película y recalcular métricas (micro-promedio dentro de película)
    agg = df.groupby(params + ["movie"])[["tp", "fp", "fn", "fusiones", "n_nucleos"]].sum().reset_index()
    agg["f1"] = 2 * agg.tp / (2 * agg.tp + agg.fp + agg.fusiones + agg.fn)
    agg["precision"] = agg.tp / (agg.tp + agg.fp + agg.fusiones)
    agg["recall"] = agg.tp / agg.n_nucleos
    loeo = []
    for mv in sorted(agg.movie.unique()):
        tr = agg[agg.movie != mv].groupby(params)["f1"].mean()
        best = tr.idxmax()
        te = agg[(agg.movie == mv) & (agg.cellprob == best[0]) & (agg.flow == best[1])].iloc[0]
        loeo.append({"movie_test": mv, "cellprob": best[0], "flow": best[1],
                     "f1": te.f1, "precision": te.precision, "recall": te.recall})
    loeo = pd.DataFrame(loeo)
    loeo.to_csv(out / "loeo.csv", index=False)
    rank = agg.groupby(params)[["f1", "precision", "recall"]].mean().sort_values("f1", ascending=False)
    rank.to_csv(out / "ranking_parametros.csv")
    # Regla de desempate (decidida al ver que el F1 máximo y el de los umbrales
    # por defecto diferían en 0.0007): diferencias de F1 menores que 1/10 del
    # desvío del F1 entre películas se consideran empate, y entre las
    # combinaciones empatadas se elige la de MAYOR PRECISIÓN. Para estadística
    # de movimiento un objeto falso es peor que una célula perdida: el falso
    # genera una trayectoria falsa (rapidez y ángulos de ruido), la perdida
    # solo achica un poco la muestra.
    sd_f1 = agg[(agg.cellprob == rank.index[0][0]) & (agg.flow == rank.index[0][1])].f1.std()
    tol = 0.1 * sd_f1
    empatadas = rank[rank.f1 >= rank.f1.iloc[0] - tol]
    best = empatadas.sort_values("precision", ascending=False).index[0]
    base = agg[(agg.cellprob == 0.0) & (agg.flow == 0.4)]
    eleccion = {"cellprob_threshold": float(best[0]), "flow_threshold": float(best[1]),
                "regla": "máximo F1; empate si la diferencia < 0.1 x SD(F1 entre películas); entre empatadas, mayor precisión",
                "tolerancia_f1": float(tol), "empatadas": [list(map(float, i)) for i in empatadas.index],
                "metricas_elegida_por_pelicula_media": rank.loc[best].to_dict(),
                "maximo_f1": {"cellprob": float(rank.index[0][0]), "flow": float(rank.index[0][1]),
                              **rank.iloc[0].to_dict()},
                "metricas_loeo_media": loeo[["f1", "precision", "recall"]].mean().to_dict(),
                "metricas_loeo_sd": loeo[["f1", "precision", "recall"]].std().to_dict(),
                "metricas_v1_default(cp0,flow0.4)": base[["f1", "precision", "recall"]].mean().to_dict(),
                "fusiones_v1": int(base.fusiones.sum()), "fp_v1": int(base.fp.sum()),
                "fusiones_elegida": int(agg[(agg.cellprob == best[0]) & (agg.flow == best[1])].fusiones.sum()),
                "fp_elegida": int(agg[(agg.cellprob == best[0]) & (agg.flow == best[1])].fp.sum())}
    (out / "eleccion.json").write_text(json.dumps(eleccion, indent=2, ensure_ascii=False))
    print(json.dumps(eleccion, indent=2, ensure_ascii=False))
    print(rank.head(10).to_string())

    if args.aplicar:
        cpt, ft = eleccion["cellprob_threshold"], eleccion["flow_threshold"]
        todos = listar_frames("bf")
        pf = []
        def guardar(it, m):
            np.savez_compressed(ruta_mascara("bf", it["movie"], it["frame"]), masks=m.astype(np.uint16))
            nuc = cargar_mascara("sirdna", it["movie"], it["frame"])
            r = metricas_vs_nucleos(m, nuc)
            r.update(movie=it["movie"], frame=it["frame"])
            return r
        futs = []
        for it in todos:
            dP, cp = cargar_flows(it["movie"], it["frame"])
            m = dynamics.compute_masks(dP, cp, niter=200, cellprob_threshold=cpt,
                                       flow_threshold=ft, min_size=15, device=dev)
            futs.append(pool.submit(guardar, it, m))
        pf = pd.DataFrame([f.result() for f in futs])
        pf.to_csv(out / "por_frame_elegido.csv", index=False)
        print("máscaras BF recalculadas con", cpt, ft)
        for mv, fr in ((1, 50), (15, 50), (8, 0)):
            figura_ejemplo(mv, fr, cargar_mascara("bf", mv, fr), out / f"ejemplo_pelicula{mv:02d}_frame{fr:03d}.png")


if __name__ == "__main__":
    main()
