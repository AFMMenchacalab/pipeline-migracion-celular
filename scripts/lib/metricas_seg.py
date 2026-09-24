"""
Métricas de segmentación por instancia, compartidas por los scripts de
ajuste/validación.
"""
import numpy as np


def matriz_solapamiento(gt, pred):
    """Tabla de contingencia (n_gt+1) x (n_pred+1) de píxeles en común,
    con un solo bincount (sin loops por instancia). Fila/columna 0 = fondo."""
    n_gt, n_pred = int(gt.max()), int(pred.max())
    idx = gt.astype(np.int64).ravel() * (n_pred + 1) + pred.astype(np.int64).ravel()
    return np.bincount(idx, minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)


def metricas_gt_parcial(gt, pred, iou_min=0.5):
    """Métricas pensadas para un ground truth PARCIAL (CAMAD anota solo las
    células aisladas, no todas las del campo): una predicción que no toca
    ninguna célula anotada NO se puede contar como falso positivo, porque
    puede ser una célula real no anotada. Por eso se evalúa todo desde el
    lado de las células anotadas:

      recall        fracción de células GT con una predicción de IoU >= iou_min
      iou_medio     IoU medio de la mejor predicción por célula GT (0 si ninguna)
      sobre_seg     fracción de células GT partidas: >= 2 predicciones cubren
                    cada una >= 20% de la célula
      sub_seg       fracción de células GT fusionadas: su mejor predicción
                    cubre también >= 20% de OTRA célula GT
      precision_local  entre las predicciones que caen mayormente (>50% de
                    su área) sobre células GT, fracción que matchea con IoU>=iou_min
                    (mide fragmentos/basura DENTRO de las zonas anotadas)
    """
    ov = matriz_solapamiento(gt, pred)
    n_gt, n_pred = ov.shape[0] - 1, ov.shape[1] - 1
    if n_gt == 0:
        return None
    a_gt = ov.sum(1)
    a_pr = ov.sum(0)
    inter = ov[1:, 1:].astype(float)
    union = a_gt[1:, None] + a_pr[None, 1:] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    best = iou.max(1) if n_pred else np.zeros(n_gt)
    frac_gt_cubierta = inter / a_gt[1:, None]          # qué fracción de la GT i cubre la pred j
    sobre = (frac_gt_cubierta >= 0.2).sum(1) >= 2 if n_pred else np.zeros(n_gt, bool)
    sub = np.zeros(n_gt, bool)
    if n_pred:
        jbest = iou.argmax(1)
        for i in range(n_gt):
            col = frac_gt_cubierta[:, jbest[i]]
            sub[i] = ((col >= 0.2).sum() >= 2) and best[i] > 0
    # predicciones "locales": > 50% de su área dentro de células GT
    if n_pred:
        frac_pred_en_gt = inter.sum(0) / np.maximum(a_pr[1:], 1)
        local = frac_pred_en_gt > 0.5
        match_pred = iou.max(0) >= iou_min
        prec_local = match_pred[local].mean() if local.any() else np.nan
    else:
        prec_local = np.nan
    return {
        "n_gt": n_gt, "n_pred": n_pred,
        "recall": float((best >= iou_min).mean()),
        "iou_medio": float(best.mean()),
        "iou_medio_matcheadas": float(best[best >= iou_min].mean()) if (best >= iou_min).any() else np.nan,
        "sobre_seg": float(sobre.mean()),
        "sub_seg": float(sub.mean()),
        "precision_local": float(prec_local),
    }


def asignar_nucleos(cel, nuc, frac_min=0.5):
    """Para cada núcleo, la célula (máscara BF) que contiene la mayoría de
    sus píxeles (0 si más de la mitad del núcleo cae en fondo).
    Devuelve (celula_de_nucleo[1..n_nuc], n_nucleos_por_celula[1..n_cel])."""
    ov = matriz_solapamiento(nuc, cel)          # filas = núcleos, cols = células
    area_nuc = ov.sum(1)
    j = ov[:, 1:].argmax(1) + 1 if ov.shape[1] > 1 else np.zeros(ov.shape[0], int)
    frac = ov[np.arange(ov.shape[0]), j] / np.maximum(area_nuc, 1)
    cel_de_nuc = np.where(frac >= frac_min, j, 0)[1:]
    n_por_cel = np.bincount(cel_de_nuc[cel_de_nuc > 0], minlength=int(cel.max()) + 1)[1:]
    return cel_de_nuc, n_por_cel


def metricas_vs_nucleos(cel, nuc):
    """Segmentación de células (BF) evaluada contra núcleos (fluorescencia)
    del MISMO campo: cada núcleo = una célula real.
      TP  = células con exactamente 1 núcleo
      FP  = células sin núcleo (basura, halo, o núcleo no detectado)
      FN  = núcleos sin célula + núcleos "extra" en células con >= 2 núcleos
    """
    cel_de_nuc, n_por_cel = asignar_nucleos(cel, nuc)
    tp = int((n_por_cel == 1).sum())
    fp = int((n_por_cel == 0).sum())
    fusion = int((n_por_cel >= 2).sum())
    fn = int((cel_de_nuc == 0).sum()) + int((n_por_cel[n_por_cel >= 2] - 1).sum()) + fusion
    # (en una fusión ninguna de las células reales quedó bien: todos sus
    # núcleos cuentan como FN y la máscara fusionada no suma TP)
    n_nuc = len(cel_de_nuc)
    return {"n_nucleos": n_nuc, "n_celulas": len(n_por_cel), "tp": tp, "fp": fp, "fn": fn,
            "fusiones": fusion, "nucleos_sin_celula": int((cel_de_nuc == 0).sum()),
            "precision": tp / max(tp + fp + fusion, 1), "recall": tp / max(n_nuc, 1),
            "f1": 2 * tp / max(2 * tp + fp + fusion + fn, 1)}
