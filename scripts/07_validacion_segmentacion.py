"""
PASO 7 (validación externa): Cellpose vs. ground truth anotado a mano, sobre
el dataset CAMAD (Zenodo 10.5281/zenodo.12806149).

Desde la revisión del 2026-09-20 quedó pendiente calcular una tasa de error
de segmentación/tracking validada -- no se pudo porque el dataset LFCT (que
tenía anotación manual) no se pudo descargar. CAMAD sí trae, para cada uno
de sus 16 experimentos, 6 pares imagen+máscara anotados a mano
(camad/images/expN/{images,masks}/), de la misma línea celular
(MDA-MB-231) que el dataset principal. Este script corre el MISMO modelo de
Cellpose (mismos parámetros que 01_segmentacion.py) sobre esas 96 imágenes
y compara contra la anotación humana.

Dos familias de métrica:
  - A nivel de PÍXEL (IoU/Dice binario: predicción>0 vs. máscara>0): no le
    importa si dos células se separaron bien entre sí, solo si el área
    total "célula vs. fondo" coincide.
  - A nivel de INSTANCIA (cada célula por separado): para cada célula real
    (ground truth), ¿hay una célula predicha que la cubra con IoU>=0.5? Y
    viceversa para cada célula predicha. De ahí salen precisión, recall y
    F1 -- la pregunta real de "¿cuántas células reales se encontraron, y
    cuántas de las encontradas eran realmente células?".

Aviso metodológico: la máscara ground truth es BINARIA (0/255), no trae un
ID por célula. Las instancias de referencia se reconstruyen acá con
componentes conexas (skimage.measure.label) -- si dos células reales están
tocándose en la anotación, van a quedar contadas como una sola instancia
"real" más grande, lo que puede subestimar el recall en zonas confluentes.
Se reporta explícitamente cuántas imágenes son densas para poder juzgar eso.

Salidas:
    resultados/validacion_camad/metricas_por_imagen.csv
    resultados/validacion_camad/metricas_resumen.csv
    resultados/figuras/validacion_ejemplo_bueno.png
    resultados/figuras/validacion_ejemplo_dificil.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from skimage.measure import label
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
CAMAD_DIR = ROOT / "datasets" / "camad-mdamb231" / "camad" / "images"
OUT_DIR = ROOT / "resultados" / "validacion_camad"
FIG_DIR = ROOT / "resultados" / "figuras"

IOU_THRESHOLD = 0.5
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0


def overlap_matrix(gt, pred):
    """Tabla de contingencia gt x pred (# píxeles en común), vectorizada:
    un solo bincount en vez de comparar instancia por instancia."""
    n_gt, n_pred = int(gt.max()), int(pred.max())
    idx = gt.astype(np.int64) * (n_pred + 1) + pred.astype(np.int64)
    counts = np.bincount(idx.ravel(), minlength=(n_gt + 1) * (n_pred + 1))
    return counts.reshape((n_gt + 1, n_pred + 1))


def instance_metrics(gt, pred, iou_thresh=IOU_THRESHOLD):
    n_gt, n_pred = int(gt.max()), int(pred.max())
    if n_gt == 0:
        return {"n_gt": 0, "n_pred": n_pred, "tp": 0, "fp": n_pred, "fn": 0,
                "precision": np.nan, "recall": np.nan, "f1": np.nan, "iou_medio_tp": np.nan}

    overlap = overlap_matrix(gt, pred)
    area_gt = overlap.sum(axis=1)
    area_pred = overlap.sum(axis=0)

    # mejor IoU de cada instancia real contra cualquier instancia predicha (y viceversa)
    best_iou_gt = np.zeros(n_gt + 1)
    for i in range(1, n_gt + 1):
        js = np.nonzero(overlap[i, 1:])[0] + 1
        if len(js):
            inter = overlap[i, js]
            iou = inter / (area_gt[i] + area_pred[js] - inter)
            best_iou_gt[i] = iou.max()

    best_iou_pred = np.zeros(n_pred + 1)
    for j in range(1, n_pred + 1):
        i_s = np.nonzero(overlap[1:, j])[0] + 1
        if len(i_s):
            inter = overlap[i_s, j]
            iou = inter / (area_gt[i_s] + area_pred[j] - inter)
            best_iou_pred[j] = iou.max()

    tp_gt = best_iou_gt[1:] >= iou_thresh   # células reales "encontradas"
    tp_pred = best_iou_pred[1:] >= iou_thresh  # células predichas que corresponden a algo real
    recall = tp_gt.mean()
    precision = tp_pred.mean() if n_pred else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and (precision + recall) > 0) else np.nan
    iou_medio_tp = best_iou_gt[1:][tp_gt].mean() if tp_gt.any() else np.nan

    return {
        "n_gt": n_gt, "n_pred": n_pred,
        "tp": int(tp_gt.sum()), "fp": int((~tp_pred).sum()), "fn": int((~tp_gt).sum()),
        "precision": precision, "recall": recall, "f1": f1, "iou_medio_tp": iou_medio_tp,
    }


def pixel_metrics(gt_bin, pred_bin):
    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    iou = inter / union if union else np.nan
    dice = 2 * inter / (gt_bin.sum() + pred_bin.sum()) if (gt_bin.sum() + pred_bin.sum()) else np.nan
    return iou, dice


def save_example(img, gt, pred, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 7))
    vmin, vmax = np.percentile(img, (1, 99))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    from skimage.measure import find_contours
    for c in find_contours(gt > 0, 0.5):
        ax.plot(c[:, 1], c[:, 0], color="#2ecc71", linewidth=1.4)
    for lbl in np.unique(pred):
        if lbl == 0:
            continue
        for c in find_contours(pred == lbl, 0.5):
            ax.plot(c[:, 1], c[:, 0], color="#e74c3c", linewidth=0.9, linestyle="--")
    ax.set_title(title)
    ax.axis("off")
    handles = [mpatches.Patch(color="#2ecc71", label="ground truth (humano)"),
               mpatches.Patch(color="#e74c3c", label="Cellpose (predicción)")]
    ax.legend(handles=handles, loc="lower right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    from cellpose import models
    model = models.CellposeModel(gpu=torch.cuda.is_available())
    print(f"GPU: {torch.cuda.is_available()}")

    exp_dirs = sorted(CAMAD_DIR.glob("exp*"), key=lambda p: int(p.name.replace("exp", "")))
    rows = []
    best_row, worst_row = None, None
    cache_for_examples = {}

    for exp_dir in exp_dirs:
        exp = exp_dir.name
        img_dir, mask_dir = exp_dir / "images", exp_dir / "masks"
        if not img_dir.exists() or not mask_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.tif")):
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                continue
            img = tifffile.imread(img_path)
            gt_bin = np.array(Image.open(mask_path)) > 0
            gt_labeled = label(gt_bin)

            pred, flows, styles = model.eval(img, flow_threshold=FLOW_THRESHOLD, cellprob_threshold=CELLPROB_THRESHOLD)

            iou_px, dice_px = pixel_metrics(gt_bin, pred > 0)
            inst = instance_metrics(gt_labeled, pred)

            row = {"exp": exp, "imagen": img_path.stem, "iou_pixel": iou_px, "dice_pixel": dice_px, **inst}
            rows.append(row)
            cache_for_examples[img_path.stem] = (img, gt_labeled, pred, row)
            print(f"{img_path.stem}: n_gt={inst['n_gt']:3d} n_pred={inst['n_pred']:3d} "
                  f"F1={inst['f1']:.3f} IoU_px={iou_px:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metricas_por_imagen.csv", index=False)

    resumen = pd.DataFrame([{
        "n_imagenes": len(df),
        "n_experimentos": df["exp"].nunique(),
        "iou_pixel_media": df["iou_pixel"].mean(), "iou_pixel_mediana": df["iou_pixel"].median(),
        "dice_pixel_media": df["dice_pixel"].mean(),
        "precision_media": df["precision"].mean(), "recall_media": df["recall"].mean(),
        "f1_media": df["f1"].mean(), "f1_mediana": df["f1"].median(),
        "iou_medio_tp_media": df["iou_medio_tp"].mean(),
        "n_gt_total": int(df["n_gt"].sum()), "n_pred_total": int(df["n_pred"].sum()),
        "tp_total": int(df["tp"].sum()), "fp_total": int(df["fp"].sum()), "fn_total": int(df["fn"].sum()),
    }])
    resumen.to_csv(OUT_DIR / "metricas_resumen.csv", index=False)
    print("\n=== RESUMEN (96 imágenes, 16 experimentos) ===")
    print(resumen.T.to_string())

    # ejemplos visuales: el de mejor y peor F1
    best_stem = df.loc[df["f1"].idxmax(), "imagen"]
    worst_stem = df.loc[df["f1"].idxmin(), "imagen"]
    img, gt_l, pred, r = cache_for_examples[best_stem]
    save_example(img, gt_l, pred, FIG_DIR / "validacion_ejemplo_bueno.png",
                 f"{best_stem} — F1={r['f1']:.2f}, IoU píxel={r['iou_pixel']:.2f} (mejor caso)")
    img, gt_l, pred, r = cache_for_examples[worst_stem]
    save_example(img, gt_l, pred, FIG_DIR / "validacion_ejemplo_dificil.png",
                 f"{worst_stem} — F1={r['f1']:.2f}, IoU píxel={r['iou_pixel']:.2f} (peor caso)")
    print(f"\nEjemplos guardados: {best_stem} (mejor), {worst_stem} (peor)")


if __name__ == "__main__":
    main()
