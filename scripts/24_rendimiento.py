"""
Tiempo de procesamiento por imagen en esta PC (GPU y CPU) y estimación para
una Raspberry Pi 5 (8 GB), para decidir si el análisis se puede hacer
mientras se adquiere (en el intervalo entre fotos).

Cómo se estima la Pi 5 (no se midió en una Pi):
  1) Se cuentan los FLOPs reales de Cellpose-SAM con torch.utils.flop_counter:
     727 GFLOP por bloque de 256x256; una imagen de 1024x1022 son 25 bloques
     (con el solapamiento por defecto) = 18.2 TFLOP.
  2) En esta PC, Cellpose en CPU con 4 hilos rinde ~48% de lo que rinde una
     multiplicación de matrices pura con los mismos hilos (240 vs 504 GFLOPS).
  3) La Pi 5 (4 x Cortex-A76 a 2.4 GHz) rinde 30.2 GFLOPS en Linpack
     (J. Geerling, github.com/geerlingguy/top500-benchmark/issues/18).
     Con la misma eficiencia del 48%: ~15 GFLOPS efectivos (rango 12-25).
  4) Tiempo Pi = FLOPs de la imagen / GFLOPS efectivos.
Detecciones + tracking + estadística: escalados por rendimiento de un solo
núcleo (~3.5x más lento en la Pi que un núcleo P del i5-13600K).

Uso: ../venv/bin/python 24_rendimiento.py            (figura desde los valores medidos)
     ../venv/bin/python 24_rendimiento.py --medir    (vuelve a medir en esta PC; ~5 min)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import res_dir  # noqa: E402

OUT = res_dir("rendimiento")

# Mediciones del 2026-09-24 (i5-13600K + RX 6800 XT; la CPU estaba parcialmente
# ocupada por el pipeline, así que los tiempos de CPU son una cota superior).
MEDIDO = {
    "cpu": "Intel Core i5-13600K (6P+8E núcleos, 20 hilos, AVX2)",
    "gpu": "AMD Radeon RX 6800 XT (ROCm 7.2, PyTorch 2.14)",
    "gflop_por_tile_256": 727.1,
    "tiles": {"1024x1022 (BF)": 25, "642x478 (CAMAD 4x)": 12, "512x511": 9},
    "gpu_fp16_s": {"1024x1022 (BF)": 1.7},   # otras tallas: escaladas por número de bloques (CAMAD medido: 0.83 s con la GPU compartida)
    "gpu_bf16_s": {"1024x1022 (BF)": 4.1},
    "gpu_fp32_s": {"1024x1022 (BF)": 2.8},
    "cpu_4hilos_s": {"1024x1022 (BF)": 75.7, "512x511": 28.1},
    "cpu_14hilos_s": {"1024x1022 (BF)": 62.2, "512x511": 21.1},
    "matmul_gflops_4hilos": 504.3,
    "detecciones_s_por_frame": 0.09,
    "tracking_s_por_frame": 0.05,
    "estadistica_s_por_pelicula": 25.0,
    "pi5_linpack_gflops": 30.2,
    "factor_un_nucleo_pi": 3.5,
}


def medir():
    import torch
    import tifffile
    import cv2
    from cellpose import models
    from torch.utils.flop_counter import FlopCounterMode
    img = tifffile.imread(str(Path(__file__).resolve().parents[1] / "datasets" / "brightfield-mdamb231"
                              / "1.1-training-source-BF-1600" / "0050.tif"))
    m = models.CellposeModel(gpu=False, use_bfloat16=False)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m.net(torch.zeros(1, 3, 256, 256))
    MEDIDO["gflop_por_tile_256"] = fc.get_total_flops() / 1e9
    torch.set_num_threads(4)
    a, b = torch.randn(2048, 2048), torch.randn(2048, 2048)
    t = time.time()
    for _ in range(10):
        a @ b
    MEDIDO["matmul_gflops_4hilos"] = 2 * 2048 ** 3 * 10 / (time.time() - t) / 1e9
    for nombre, im in (("1024x1022 (BF)", img), ("512x511", cv2.resize(img, (512, 511), interpolation=cv2.INTER_AREA))):
        m.eval(im)
        t = time.time()
        m.eval(im)
        MEDIDO["cpu_4hilos_s"][nombre] = time.time() - t


def estimar():
    M = MEDIDO
    efic = (M["gflop_por_tile_256"] * M["tiles"]["1024x1022 (BF)"] / M["cpu_4hilos_s"]["1024x1022 (BF)"]) / M["matmul_gflops_4hilos"]
    pi_eff = {"central": M["pi5_linpack_gflops"] * efic, "rapido": 25.0, "lento": 12.0}
    filas = []
    for tam, nt in M["tiles"].items():
        tflop = M["gflop_por_tile_256"] * nt / 1000
        otros_pc = M["detecciones_s_por_frame"] + M["tracking_s_por_frame"]
        otros_pi = otros_pc * M["factor_un_nucleo_pi"]
        filas.append({
            "imagen": tam, "tflop_por_imagen": tflop,
            "pc_gpu_fp16_s": M["gpu_fp16_s"].get(tam, M["gpu_fp16_s"]["1024x1022 (BF)"] * nt / 25),
            "pc_cpu_4hilos_s": M["cpu_4hilos_s"].get(tam, M["cpu_4hilos_s"]["1024x1022 (BF)"] * nt / 25),
            "pi5_s": tflop * 1000 / pi_eff["central"] + otros_pi,
            "pi5_s_min": tflop * 1000 / pi_eff["rapido"] + otros_pi,
            "pi5_s_max": tflop * 1000 / pi_eff["lento"] + otros_pi,
            "otros_pasos_pc_s": otros_pc, "otros_pasos_pi_s": otros_pi,
        })
    return efic, pi_eff, filas


def figura(filas):
    from lib import estilo as S
    from lib.estilo import plt
    fig, ax = plt.subplots(figsize=(10, 4.2))
    etiquetas = [r["imagen"] for r in filas]
    y = np.arange(len(filas))
    h = 0.26
    ax.barh(y - h, [r["pc_gpu_fp16_s"] for r in filas], h, color=S.CAT[0], label="PC, GPU RX 6800 XT (fp16)")
    ax.barh(y, [r["pc_cpu_4hilos_s"] for r in filas], h, color=S.CAT[1], label="PC, CPU (4 hilos)")
    ax.barh(y + h, [r["pi5_s"] for r in filas], h, color=S.CAT[2], label="Raspberry Pi 5 (estimado)",
            xerr=[[r["pi5_s"] - r["pi5_s_min"] for r in filas], [r["pi5_s_max"] - r["pi5_s"] for r in filas]],
            error_kw=dict(ecolor=S.TINTA2, capsize=3, lw=1))
    for x, lab in ((30, "intervalo CAMAD (30 s)"), (300, "intervalo brightfield (5 min)")):
        ax.axvline(x, color=S.TINTA2, ls="--", lw=1)
        ax.text(x * 1.05, len(filas) - 0.45, lab, fontsize=8, color=S.TINTA2, va="top")
    ax.set_xscale("log")
    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.set_xlabel("segundos por imagen (segmentación + detección + tracking), escala logarítmica")
    ax.legend(loc="lower right")
    ax.set_title("Tiempo de procesamiento por imagen vs intervalo entre fotos")
    S.guardar(fig, res_dir("figuras") / "rendimiento")


def main():
    if "--medir" in sys.argv:
        medir()
    efic, pi_eff, filas = estimar()
    salida = {"medido": MEDIDO, "eficiencia_cellpose_vs_matmul": efic, "pi5_gflops_efectivos": pi_eff, "tabla": filas}
    (OUT / "rendimiento.json").write_text(json.dumps(salida, indent=2, ensure_ascii=False))
    figura(filas)
    for r in filas:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    print("eficiencia", round(efic, 3), "Pi GFLOPS efectivos", {k: round(v, 1) for k, v in pi_eff.items()})


if __name__ == "__main__":
    main()
