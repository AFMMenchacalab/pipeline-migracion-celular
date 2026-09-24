"""
Validación del tracking brightfield contra el tracking de NÚCLEOS (SiR-DNA)
del mismo campo.

Idea: los núcleos marcados con fluorescencia son objetos chicos, de alto
contraste y bien separados entre sí aun cuando las células se tocan, así
que su tracking es mucho menos ambiguo que el de las células en
brightfield. Se usa como referencia ("pseudo ground truth"):

  1. En cada frame se asocia cada célula BF con el núcleo que contiene
     (solo asociaciones 1 a 1: célula con exactamente un núcleo).
  2. Eslabón BF (misma trayectoria BF, frame t -> siguiente detección):
     es CORRECTO si los dos núcleos asociados pertenecen a la misma
     trayectoria nuclear. Precisión = correctos / evaluables.
  3. Eslabón nuclear: se RECUPERA si las dos células BF asociadas
     pertenecen a la misma trayectoria BF. Recall = recuperados / evaluables.
  4. Pureza de trayectoria BF: fracción de sus detecciones asociadas que
     pertenecen a su núcleo "dominante" (1 = la trayectoria nunca saltó a
     otra célula). Completitud: fracción de una trayectoria nuclear que
     cubre su trayectoria BF dominante (1 = no se fragmentó).

Limitación honesta: el tracking nuclear no es perfecto (mitosis, núcleos
tenues), así que estas métricas son una cota inferior de la calidad real
del tracking BF. Pero comparan todas las variantes de tracking con la
misma referencia, que es lo que se necesita para elegir entre ellas.

Salidas: resultados/v2/validacion_tracking/{por_pelicula.csv, resumen.csv, eleccion.json}
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import res_dir  # noqa: E402
from lib.fuentes import cargar_mascara  # noqa: E402
from lib.metricas_seg import asignar_nucleos  # noqa: E402

REF = "dist"   # variante de tracking usada para los núcleos


def mapa_pelicula(movie, frames):
    filas = []
    for f in frames:
        cel = cargar_mascara("bf", movie, f)
        nuc = cargar_mascara("sirdna", movie, f)
        cel_de_nuc, n_por_cel = asignar_nucleos(cel, nuc)
        for nl, cl in enumerate(cel_de_nuc, start=1):
            if cl > 0 and n_por_cel[cl - 1] == 1:
                filas.append((f, cl, nl))
    return pd.DataFrame(filas, columns=["frame", "label_bf", "label_nuc"])


def eslabones(tr):
    tr = tr.sort_values(["track_id", "frame"])
    nxt = tr.groupby("track_id").shift(-1)
    ok = nxt["frame"].notna()
    return pd.DataFrame({"f0": tr.loc[ok, "frame"].to_numpy(), "l0": tr.loc[ok, "label"].to_numpy(),
                         "f1": nxt.loc[ok, "frame"].astype(int).to_numpy(),
                         "l1": nxt.loc[ok, "label"].astype(int).to_numpy()})


def evaluar(args):
    movie, variantes = args
    nuc_tr = pd.read_csv(res_dir("sirdna", "tracking", REF) / "tracks.csv.gz",
                         usecols=["movie", "frame", "label", "track_id"])
    nuc_tr = nuc_tr[nuc_tr.movie == movie]
    frames = sorted(nuc_tr.frame.unique())
    mp = mapa_pelicula(movie, frames)
    bf2nuc = {(f, b): n for f, b, n in mp.itertuples(index=False)}
    nuc2bf = {(f, n): b for f, b, n in mp.itertuples(index=False)}
    nuc_track = {(f, l): t for f, l, t in nuc_tr[["frame", "label", "track_id"]].itertuples(index=False)}
    en = eslabones(nuc_tr)
    filas = []
    for var in variantes:
        bf_tr = pd.read_csv(res_dir("bf", "tracking", var) / "tracks.csv.gz",
                            usecols=["movie", "frame", "label", "track_id"])
        bf_tr = bf_tr[bf_tr.movie == movie]
        bf_track = {(f, l): t for f, l, t in bf_tr[["frame", "label", "track_id"]].itertuples(index=False)}
        eb = eslabones(bf_tr)
        # precisión de eslabones BF
        ok = tot = 0
        for f0, l0, f1, l1 in eb.itertuples(index=False):
            n0, n1 = bf2nuc.get((f0, l0)), bf2nuc.get((f1, l1))
            if n0 is None or n1 is None:
                continue
            t0, t1 = nuc_track.get((f0, n0)), nuc_track.get((f1, n1))
            if t0 is None or t1 is None:
                continue
            tot += 1
            ok += int(t0 == t1)
        prec = ok / max(tot, 1)
        # recall de eslabones nucleares
        rec_ok = rec_tot = 0
        for f0, l0, f1, l1 in en.itertuples(index=False):
            b0, b1 = nuc2bf.get((f0, l0)), nuc2bf.get((f1, l1))
            if b0 is None or b1 is None:
                continue
            rec_tot += 1
            t0, t1 = bf_track.get((f0, b0)), bf_track.get((f1, b1))
            rec_ok += int(t0 is not None and t0 == t1)
        rec = rec_ok / max(rec_tot, 1)
        # pureza y completitud
        bf_tr = bf_tr.assign(nuc_track=[nuc_track.get((f, bf2nuc.get((f, l)))) if (f, l) in bf2nuc else None
                                        for f, l in zip(bf_tr.frame, bf_tr.label)])
        m = bf_tr.dropna(subset=["nuc_track"])
        if len(m):
            dom = m.groupby("track_id")["nuc_track"].agg(lambda s: s.value_counts().iloc[0] / len(s))
            w = m.groupby("track_id").size()
            pureza = float((dom * w).sum() / w.sum())
            cov = m.groupby(["nuc_track", "track_id"]).size().groupby(level=0).max()
            largo_nuc = nuc_tr.groupby("track_id").size()
            comp = float(cov.sum() / largo_nuc.loc[cov.index].sum())
        else:
            pureza = comp = np.nan
        filas.append({"movie": movie, "variante": var, "precision_eslabon": prec, "recall_eslabon": rec,
                      "f1_eslabon": 2 * prec * rec / max(prec + rec, 1e-9), "pureza": pureza,
                      "completitud": comp, "eslabones_evaluados": tot,
                      "largo_medio_bf": float(bf_tr.groupby("track_id").size().mean()),
                      "largo_medio_nuc": float(nuc_tr.groupby("track_id").size().mean())})
    return filas


def main():
    out = res_dir("validacion_tracking")
    variantes = sorted(p.name for p in res_dir("bf", "tracking").iterdir() if (p / "tracks.csv.gz").exists())
    movies = list(range(1, 17))
    with ProcessPoolExecutor(16) as ex:
        filas = [f for r in ex.map(evaluar, [(m, variantes) for m in movies]) for f in r]
    df = pd.DataFrame(filas)
    df.to_csv(out / "por_pelicula.csv", index=False)
    res = df.groupby("variante")[["precision_eslabon", "recall_eslabon", "f1_eslabon", "pureza",
                                  "completitud", "largo_medio_bf", "largo_medio_nuc"]].agg(["mean", "std"])
    res.to_csv(out / "resumen.csv")
    print(res.round(4).to_string())
    medias = df.groupby("variante")["f1_eslabon"].mean()
    mejor = medias.idxmax()
    # test pareado por película de la mejor contra v1
    from scipy.stats import wilcoxon
    a = df[df.variante == mejor].set_index("movie")["f1_eslabon"]
    b = df[df.variante == "v1"].set_index("movie")["f1_eslabon"]
    p = float(wilcoxon(a, b).pvalue) if mejor != "v1" else np.nan
    el = {"elegida": mejor, "f1_media": float(medias[mejor]), "f1_v1": float(medias.get("v1", np.nan)),
          "p_wilcoxon_vs_v1_pareado_por_pelicula": p, "referencia_nuclear": REF}
    (out / "eleccion.json").write_text(json.dumps(el, indent=2))
    print(el)


if __name__ == "__main__":
    main()
