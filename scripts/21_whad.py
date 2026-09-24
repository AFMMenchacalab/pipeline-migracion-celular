"""
WHAD (Wound Healing Assay Dataset, Iheme et al. 2024): cinética de cierre
de herida a partir de las máscaras MANUALES del dataset (no hace falta
segmentar: la herida ya está delineada por expertos en cada frame).

Datos (paper del dataset): contraste de fase, 10x, 0.454 um/px, 1 frame/h
durante 48-72 h, mitomicina C (bloquea proliferación, así que el cierre es
por migración). Condiciones:
  MCF10A: CC = control, NC = sobreexpresión de NICD (Notch1 activo),
          C61 = shRNA contra CYR61, N61 = NICD + shRNA CYR61
          (Ilhan et al. 2020: CYR61 media la migración inducida por Notch1)
  MCF7:   LacZ = control, SEMA6D = sobreexpresión de SEMA6D
          (Gunyuz et al. 2022: SEMA6D promueve migración y desprendimiento)

Qué se mide por posición (P1..P4):
  - ancho medio de la herida = área negra / largo de la herida (la herida
    atraviesa toda la imagen; se detecta si es vertical u horizontal);
  - velocidad de avance del frente = -d(ancho)/dt / 2 (dos frentes),
    ajuste lineal en las primeras 12 h;
  - área relativa A(t)/A(0) y tiempo al 50 % de cierre;
  - células desprendidas (máscaras *_Cyy): número y área por frame.
Aviso: las posiciones de un mismo pocillo NO son réplicas biológicas
independientes (mismo pocillo, mismo día). Se reportan como tales y los
tests entre condiciones son exploratorios.

Salidas: resultados/v2/whad/*.csv, figuras whad_*.png
"""
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS_DIR, res_dir  # noqa: E402

WHAD = DATASETS_DIR / "camad-mdamb231" / "whad" / "masks"
UM = 0.454
DT_H = 1.0


def parsear(p):
    rel = p.relative_to(WHAD).parts
    linea = rel[0]
    nombre = p.name
    t = int(re.search(r"_t(\d+)_", nombre).group(1))
    det = re.search(r"_mask_C(\d+)\.png$", nombre)
    if linea == "MCF10A":
        cond = rel[3]
        grupo = f"{rel[1]}/{rel[2]}"
        pos = rel[4]
    else:
        cond = "SEMA6D" if "sema6d" in rel[3].lower() else "LacZ"
        grupo = f"{rel[1]}/{rel[2]}"
        pos = re.search(r"_p(\d+)_t", nombre).group(1)
    return dict(linea=linea, condicion=cond, grupo=grupo, posicion=pos, t=t,
                desprendida=1 if det else 0, id_desprendida=int(det.group(1)) if det else -1, path=str(p))


def medir(fila):
    m = np.array(Image.open(fila["path"]))
    if m.ndim == 3:
        m = m[..., 0]
    if fila["desprendida"]:
        return {**fila, "area_desprendida_um2": float((m > 0).sum() * UM ** 2)}
    herida = m == 0
    filas_con = herida.any(axis=1).mean()
    cols_con = herida.any(axis=0).mean()
    largo_px = m.shape[0] if filas_con >= cols_con else m.shape[1]
    return {**fila, "frac_herida": float(herida.mean()),
            "ancho_um": float(herida.sum() / largo_px * UM), "orientacion": "vertical" if filas_con >= cols_con else "horizontal"}


def main():
    out = res_dir("whad")
    filas = [parsear(p) for p in WHAD.rglob("*.png")]
    with ProcessPoolExecutor(16) as ex:
        res = pd.DataFrame(list(ex.map(medir, filas, chunksize=16)))
    res.drop(columns=["path"]).to_csv(out / "mediciones_crudas.csv", index=False)
    her = res[res.desprendida == 0].copy()
    her["id"] = her.linea + "|" + her.grupo + "|" + her.condicion + "|" + her.posicion.astype(str)
    her = her.sort_values(["id", "t"])
    her["area_rel"] = her.ancho_um / her.groupby("id").ancho_um.transform("first")
    des = res[res.desprendida > 0].copy()
    des["id"] = des.linea + "|" + des.grupo + "|" + des.condicion + "|" + des.posicion.astype(str)
    des_t = des.groupby(["id", "t"]).agg(n_desprendidas=("desprendida", "size"),
                                        area_desprendida_um2=("area_desprendida_um2", "sum")).reset_index()
    her = her.merge(des_t, on=["id", "t"], how="left").fillna({"n_desprendidas": 0, "area_desprendida_um2": 0})
    her.to_csv(out / "herida_por_frame.csv", index=False)
    pos = []
    for pid, g in her.groupby("id"):
        g = g.sort_values("t")
        tt = g.t.to_numpy() * DT_H
        w = g.ancho_um.to_numpy()
        ok = tt <= 12
        vel = -np.polyfit(tt[ok], w[ok], 1)[0] / 2 if ok.sum() >= 3 else np.nan
        rel = g.area_rel.to_numpy()
        t50 = float(np.interp(0.5, rel[::-1], tt[::-1])) if rel.min() <= 0.5 else np.nan
        pos.append({"id": pid, "linea": g.linea.iloc[0], "condicion": g.condicion.iloc[0], "grupo": g.grupo.iloc[0],
                    "posicion": g.posicion.iloc[0], "n_frames": len(g), "duracion_h": tt[-1],
                    "ancho_inicial_um": w[0], "velocidad_frente_um_h": vel,
                    "cierre_12h_pct": 100 * (1 - np.interp(12, tt, rel)) if tt[-1] >= 12 else np.nan,
                    "t50_h": t50, "cierre_final_pct": 100 * (1 - rel[-1]),
                    "desprendidas_media": g.n_desprendidas.mean(),
                    "desprendidas_max": g.n_desprendidas.max()})
    pos = pd.DataFrame(pos)
    pos.to_csv(out / "por_posicion.csv", index=False)
    print(pos.groupby(["linea", "condicion"])[["velocidad_frente_um_h", "cierre_12h_pct", "t50_h",
                                               "desprendidas_media"]].agg(["mean", "std", "count"]).round(2).to_string())
    # tests exploratorios (posiciones como unidad, con la advertencia del docstring)
    from scipy import stats
    tests = []
    for linea, ref in (("MCF10A", "CC"), ("MCF7", "LacZ")):
        p = pos[pos.linea == linea]
        for c in sorted(p.condicion.unique()):
            if c == ref:
                continue
            for m in ["velocidad_frente_um_h", "cierre_12h_pct", "desprendidas_media"]:
                a, b = p[p.condicion == c][m].dropna(), p[p.condicion == ref][m].dropna()
                if len(a) >= 2 and len(b) >= 2:
                    tests.append({"linea": linea, "condicion": c, "vs": ref, "metrica": m, "media": a.mean(),
                                  "media_ref": b.mean(), "n": len(a), "n_ref": len(b),
                                  "p_mannwhitney": stats.mannwhitneyu(a, b).pvalue})
    pd.DataFrame(tests).to_csv(out / "tests_exploratorios.csv", index=False)
    print(pd.DataFrame(tests).round(4).to_string())

    # figura
    from lib import estilo as S
    from lib.estilo import plt
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    for i, (linea, conds) in enumerate((("MCF10A", ["CC", "NC", "C61", "N61"]), ("MCF7", ["LacZ", "SEMA6D"]))):
        for k, c in enumerate(conds):
            g = her[(her.linea == linea) & (her.condicion == c)]
            for _, gg in g.groupby("id"):
                axs[i, 0].plot(gg.t * DT_H, gg.area_rel, color=S.CAT[k], lw=0.8, alpha=0.6)
            m = g.groupby("t").area_rel.mean()
            axs[i, 0].plot(m.index * DT_H, m.values, color=S.CAT[k], lw=2.4, label=c)
            d = g.groupby("t").n_desprendidas.mean()
            axs[i, 1].plot(d.index * DT_H, d.values, color=S.CAT[k], lw=2, label=c)
        axs[i, 0].set_ylabel("área de herida relativa A(t)/A(0)")
        axs[i, 0].set_title(f"{linea}: cierre de herida (líneas finas = posiciones)")
        axs[i, 0].legend()
        axs[i, 1].set_ylabel("células/grupos desprendidos por frame")
        axs[i, 1].set_title(f"{linea}: desprendimiento")
        axs[i, 1].legend()
        axs[i, 0].set_xlabel("tiempo (h)")
        axs[i, 1].set_xlabel("tiempo (h)")
    fig.tight_layout()
    S.guardar(fig, res_dir("figuras") / "whad_cierre")


if __name__ == "__main__":
    main()
