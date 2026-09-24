"""
Ajuste y validación de la segmentación de CAMAD contra la anotación manual
(96 imágenes = 16 experimentos x 6 frames, cada 10 min durante los primeros
50 min del video).

Qué cambia respecto de 07_validacion_segmentacion.py (v1) y por qué:

1. Instancias de referencia desde los ROI de ImageJ, no desde componentes
   conexas de la máscara binaria. v1 reconstruía "células" con
   skimage.label sobre el PNG binario: dos células anotadas que se tocan
   quedaban como UNA sola célula de referencia. Los .roi traen el polígono
   de cada célula por separado, así que acá cada célula es su propia
   instancia (se rasteriza el polígono).

2. Métricas para ground truth PARCIAL. El paper del dataset aclara que
   "solo se incluyeron las células que no estaban en contacto con otras".
   Por eso en v1 la precisión daba muy baja: contaba como falso positivo a
   células reales no anotadas. Acá las métricas se calculan desde el lado
   de las células anotadas (ver lib/metricas_seg.py): recall, IoU,
   sobre-segmentación, sub-segmentación y una "precisión local" restringida
   a las zonas anotadas.

3. Se buscan la escala y los umbrales en vez de fijarlos. La red se corre
   UNA vez por (imagen, escala) y los umbrales (cellprob_threshold,
   flow_threshold) se barren sobre los flows ya calculados (la dinámica en
   GPU tarda ~0.1 s, la red ~1-8 s).

La elección se hace con validación cruzada por experimento (leave-one-
experiment-out): para cada experimento se elige la mejor combinación
usando los OTROS 15 y se evalúa en el que quedó afuera. Así el número
reportado no está inflado por haber elegido los parámetros sobre los mismos
datos con que se evalúa.

Salidas: resultados/v2/validacion_seg_camad/{grilla.csv, loeo.csv, eleccion.json}

Con --aplicar (después de 11_segmentar_gpu.py --dataset camad, que guarda
los flows): recalcula las máscaras de los 9600 frames con los umbrales
elegidos, sin volver a correr la red. Solo si la escala elegida coincide
con la de extracción (CAMAD_DOWNSAMPLE); si no, lo avisa y deja las
máscaras por defecto.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import roifile
import tifffile
from skimage.draw import polygon as draw_polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, res_dir  # noqa: E402
from lib.metricas_seg import metricas_gt_parcial  # noqa: E402

ESCALAS = [1, 2, 4, 6]               # factor de reducción respecto de la resolución original
# a resolución completa (escala 1, lo que hacía v1) solo se evalúan los
# umbrales por defecto: es la referencia de comparación, y cada pasada de
# dinámica a 2568x1912 cuesta ~1 s
SOLO_DEFAULT_EN_ESCALA_1 = True
CELLPROB = [-2.0, -1.0, 0.0, 1.0]
FLOW = [0.4, 0.8, 0.0]          # 0.0 = sin control de consistencia de flujo
# criterio para elegir: recall medio penalizando partir/fusionar células
def puntaje(df):
    return df["recall"] - 0.5 * df["sobre_seg"] - 0.5 * df["sub_seg"]


def gt_instancias(exp, k, shape):
    rois = roifile.roiread(DATASETS["camad"]["gt_rois"] / f"exp{exp}_roi.zip")
    lab = np.zeros(shape, np.uint16)
    n = 0
    for r in rois:
        if r.position != k:
            continue
        xy = r.coordinates()
        rr, cc = draw_polygon(xy[:, 1], xy[:, 0], shape)
        n += 1
        lab[rr, cc] = n
    return lab


def aplicar():
    import torch
    from cellpose import dynamics
    from concurrent.futures import ThreadPoolExecutor
    from lib.config import CAMAD_DOWNSAMPLE, cache_dir
    from lib.fuentes import listar_frames, ruta_mascara
    d = res_dir("validacion_seg_camad")
    el = json.loads((d / "eleccion.json").read_text())
    # Los frames ya se extrajeron a CAMAD_DOWNSAMPLE; se eligen los mejores
    # umbrales DENTRO de esa escala y se registra cuánto se pierde respecto
    # del óptimo global (si fuera mucho, convendría re-extraer a otra escala).
    rk = pd.read_csv(d / "ranking_parametros.csv")
    # el puntaje de la grilla no ve la basura (objetos que no son células
    # anotadas); se suma la precisión local para no elegir combinaciones
    # permisivas (p.ej. sin control de flujo) que empatan en recall pero
    # llenan el campo de fragmentos
    rk["puntaje_final"] = rk.puntaje + 0.5 * rk.precision_local.fillna(0)
    en_escala = rk[rk.escala == CAMAD_DOWNSAMPLE].sort_values("puntaje_final", ascending=False).iloc[0]
    cpt, ft = float(en_escala.cellprob), float(en_escala.flow)
    aplicada = {"escala": CAMAD_DOWNSAMPLE, "cellprob_threshold": cpt, "flow_threshold": ft,
                "puntaje": float(en_escala.puntaje), "puntaje_final": float(en_escala.puntaje_final),
                "recall": float(en_escala.recall),
                "iou_medio": float(en_escala.iou_medio), "precision_local": float(en_escala.precision_local),
                "optimo_global": {"escala": el["escala"], "cellprob": el["cellprob_threshold"],
                                  "flow": el["flow_threshold"],
                                  "mejor_puntaje_final_cualquier_escala": rk.sort_values(
                                      "puntaje_final", ascending=False).iloc[0][["escala", "cellprob", "flow",
                                                                                 "puntaje_final"]].to_dict()}}
    (d / "eleccion_aplicada.json").write_text(json.dumps(aplicada, indent=2, ensure_ascii=False))
    print(json.dumps(aplicada, indent=2, ensure_ascii=False), flush=True)
    if (cpt, ft) == (0.0, 0.4):
        print("umbrales elegidos = por defecto; nada que recalcular", flush=True)
        return
    dev = torch.device("cuda")
    pool = ThreadPoolExecutor(8)
    futs = []
    t0 = time.time()
    for n, it in enumerate(listar_frames("camad")):
        f = cache_dir("flows", "camad", f"m{it['movie']:02d}") / f"f{it['frame']:03d}.npz"
        with np.load(f) as z:
            dP, cp = z["dP"].astype(np.float32), z["cellprob"].astype(np.float32)
        m = dynamics.compute_masks(dP, cp, niter=200, cellprob_threshold=cpt, flow_threshold=ft,
                                   min_size=15, device=dev)
        futs.append(pool.submit(np.savez_compressed, ruta_mascara("camad", it["movie"], it["frame"]),
                                masks=m.astype(np.uint16)))
        if n % 1000 == 0:
            print(f"aplicar {n} ({time.time() - t0:.0f}s)", flush=True)
    for f in futs:
        f.result()
    print(f"máscaras CAMAD recalculadas con cellprob={cpt}, flow={ft}", flush=True)


def main():
    if "--aplicar" in sys.argv:
        return aplicar()
    import torch
    from cellpose import models, dynamics
    m = models.CellposeModel(gpu=True, use_bfloat16=False)
    m.net.half()
    m.net.dtype = torch.float16
    dev = torch.device("cuda")

    out = res_dir("validacion_seg_camad")
    filas = []
    t0 = time.time()
    for exp in range(1, 17):
        for k in range(1, 7):
            img = tifffile.imread(DATASETS["camad"]["gt_images"] / f"exp{exp}" / "images" / f"exp{exp}_{k}.tif")
            if img.ndim == 3:
                img = img[..., 0] if img.shape[-1] in (3, 4) else img[0]
            gt = gt_instancias(exp, k, img.shape)
            if gt.max() == 0:
                continue
            for s in ESCALAS:
                if s == 1:
                    im_s = img
                else:
                    im_s = cv2.resize(img, (img.shape[1] // s, img.shape[0] // s), interpolation=cv2.INTER_AREA)
                _, flows, _ = m.eval(im_s, compute_masks=False, batch_size=16)
                dP, cp = flows[1], flows[2]
                combos = [(cpt, ft) for cpt in CELLPROB for ft in FLOW]
                if s == 1 and SOLO_DEFAULT_EN_ESCALA_1:
                    combos = [(0.0, 0.4)]
                for cpt, ft in combos:
                    if True:
                        pred = dynamics.compute_masks(dP, cp, niter=200, cellprob_threshold=cpt,
                                                      flow_threshold=ft, min_size=15, device=dev)
                        if s != 1:
                            pred = cv2.resize(pred.astype(np.uint16), (img.shape[1], img.shape[0]),
                                              interpolation=cv2.INTER_NEAREST)
                        met = metricas_gt_parcial(gt, pred)
                        met.update(exp=exp, k=k, escala=s, cellprob=cpt, flow=ft)
                        filas.append(met)
            print(f"exp{exp} k{k} listo ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(filas).to_csv(out / "grilla.csv", index=False)

    df = pd.DataFrame(filas)
    df.to_csv(out / "grilla.csv", index=False)
    params = ["escala", "cellprob", "flow"]
    # promedio por experimento primero (cada experimento pesa igual)
    por_exp = df.groupby(params + ["exp"])[["recall", "iou_medio", "sobre_seg", "sub_seg", "precision_local"]].mean().reset_index()
    por_exp["puntaje"] = puntaje(por_exp)

    # leave-one-experiment-out
    loeo = []
    for exp in sorted(por_exp["exp"].unique()):
        train = por_exp[por_exp["exp"] != exp].groupby(params)["puntaje"].mean()
        best = train.idxmax()
        test = por_exp[(por_exp["exp"] == exp) & (por_exp[params].apply(tuple, axis=1) == best)].iloc[0]
        loeo.append({"exp_test": exp, "escala": best[0], "cellprob": best[1], "flow": best[2],
                     **{c: test[c] for c in ["recall", "iou_medio", "sobre_seg", "sub_seg", "precision_local"]}})
    loeo = pd.DataFrame(loeo)
    loeo.to_csv(out / "loeo.csv", index=False)

    global_ = por_exp.groupby(params)[["recall", "iou_medio", "sobre_seg", "sub_seg", "precision_local", "puntaje"]].mean()
    global_ = global_.sort_values("puntaje", ascending=False)
    global_.to_csv(out / "ranking_parametros.csv")
    best = global_.index[0]
    base = global_.loc[(1, 0.0, 0.4)] if (1, 0.0, 0.4) in global_.index else None
    eleccion = {
        "escala": int(best[0]), "cellprob_threshold": float(best[1]), "flow_threshold": float(best[2]),
        "metricas_elegida_in_sample": global_.iloc[0].to_dict(),
        "metricas_loeo_media": loeo[["recall", "iou_medio", "sobre_seg", "sub_seg", "precision_local"]].mean().to_dict(),
        "metricas_v1_equivalente(escala1,cp0,flow0.4)": base.to_dict() if base is not None else None,
        "escala4_cp0_flow0.4": global_.loc[(4, 0.0, 0.4)].to_dict(),
    }
    (out / "eleccion.json").write_text(json.dumps(eleccion, indent=2, ensure_ascii=False))
    print(json.dumps(eleccion, indent=2, ensure_ascii=False))
    print(global_.head(15).to_string())


if __name__ == "__main__":
    main()
