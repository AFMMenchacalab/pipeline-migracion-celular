"""
Cellpose 3 (cyto3, red U-Net) contra Cellpose-SAM (cpsam, transformer ViT-L):
tamaño, costo de cálculo, tiempo y calidad de segmentación sobre las mismas
imágenes brightfield, evaluadas contra los núcleos SiR-DNA.

Cellpose 3 se instala APARTE (pip install --target <carpeta> cellpose==3.1.1.2
--no-deps) para no reemplazar el Cellpose 4 del pipeline; se pasa la carpeta
con --cp3. Corre en CPU con 4 hilos (como tendría una Raspberry Pi 5).

Imágenes: el frame 50 de cada una de las 16 películas. Los valores de
Cellpose-SAM salen de la grilla de 13_ajuste_segmentacion_bf.py (mismos
frames, umbrales por defecto).

Uso: ../venv/bin/python 25_cellpose3_vs_sam.py --cp3 /ruta/a/cp3
Salida: resultados/v2/cellpose3_vs_sam/{comparacion.json, por_imagen.csv}
"""
import argparse
import json
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--cp3", required=True)
ap.add_argument("--frame", type=int, default=50)
args = ap.parse_args()
sys.path.insert(0, args.cp3)                        # Cellpose 3 primero
sys.path.insert(1, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
import cellpose  # noqa: E402
from cellpose import models  # noqa: E402

from lib.config import res_dir  # noqa: E402
from lib.fuentes import listar_frames, leer_frame, cargar_mascara  # noqa: E402
from lib.metricas_seg import metricas_vs_nucleos  # noqa: E402

assert "3." in getattr(cellpose, "version", "3") or Path(cellpose.__file__).is_relative_to(args.cp3), cellpose.__file__
torch.set_num_threads(4)
out = res_dir("cellpose3_vs_sam")

m = models.Cellpose(gpu=False, model_type="cyto3")
net = m.cp.net
n_param = sum(p.numel() for p in net.parameters())
x = torch.zeros(1, 2, 224, 224)
with FlopCounterMode(display=False) as fc, torch.no_grad():
    net(x)
gflop_tile = fc.get_total_flops() / 1e9

filas = []
items = [it for it in listar_frames("bf") if it["frame"] == args.frame]
m.eval(leer_frame(items[0]), diameter=None, channels=[0, 0])   # calentamiento
for it in items:
    img = leer_frame(it)
    t = time.time()
    masks, flows, styles, diam = m.eval(img, diameter=None, channels=[0, 0])
    dt = time.time() - t
    nuc = cargar_mascara("sirdna", it["movie"], it["frame"])
    r = metricas_vs_nucleos(masks, nuc)
    r.update(movie=it["movie"], frame=it["frame"], segundos_cpu4=dt, diametro_estimado_px=float(diam))
    filas.append(r)
    print(it["movie"], f"{dt:.1f}s", f"F1={r['f1']:.3f}", flush=True)
c3 = pd.DataFrame(filas)
c3.to_csv(out / "por_imagen_cellpose3.csv", index=False)

g = pd.read_csv(res_dir("validacion_seg_bf") / "grilla.csv")
sam = g[(g.cellprob == 0) & (g.flow == 0.4) & (g.frame == args.frame)].set_index("movie")


def micro(df):
    tp, fp, fn, fu, nn = df.tp.sum(), df.fp.sum(), df.fn.sum(), df.fusiones.sum(), df.n_nucleos.sum()
    return {"f1": 2 * tp / (2 * tp + fp + fu + fn), "precision": tp / (tp + fp + fu), "recall": tp / nn,
            "fusiones": int(fu), "celulas_sin_nucleo": int(fp), "nucleos_sin_celula": int(df.nucleos_sin_celula.sum())}


from scipy.stats import wilcoxon  # noqa: E402
pares = c3.set_index("movie").join(sam[["f1"]], rsuffix="_sam")
res = {
    "cellpose3": {"modelo": "cyto3 (U-Net residual, Cellpose 3.1)", "parametros_millones": n_param / 1e6,
                  "gflop_por_bloque_224": gflop_tile, **micro(c3),
                  "segundos_por_imagen_cpu4": float(c3.segundos_cpu4.median()),
                  "diametro_estimado_px_mediana": float(c3.diametro_estimado_px.median())},
    "cellpose_sam": {"modelo": "cpsam (encoder ViT-L de SAM, Cellpose 4)", **micro(sam.reset_index())},
    "f1_por_imagen_mediana": {"cellpose3": float(pares.f1.median()), "cellpose_sam": float(pares.f1_sam.median())},
    "p_wilcoxon_pareado_f1": float(wilcoxon(pares.f1, pares.f1_sam).pvalue),
    "n_imagenes": len(c3),
}
(out / "comparacion.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps(res, indent=2, ensure_ascii=False))
