"""
Estadística completa de un experimento del microscopio propio con el mismo
motor que el pipeline v2 (16_estadisticas.analizar_pelicula): MSD, α corto
y largo, PRW, VACF, espectro, EAD y SE (crudas y corregidas), alineamiento
entre vecinas, sensibilidad al paso y forma.

Unidad de réplica: cada cámara es una caja de Petri distinta (misma línea y
mismas condiciones), así que cada una cuenta como una réplica, igual que
cada película en el pipeline v2. Además se repite todo por ventanas de
tiempo (por defecto 12 h) dentro de cada caja, para ver cómo cambia a lo
largo del experimento.

Entrada: lo que deja lib/propio.analizar (analisis/propio/<exp>/<cam>/
tracks.csv, o <cam>/hasta_N/ con --max-grupo).
Salida: analisis/propio/<exp>/estadisticas[/hasta_N]/
  por_pelicula.csv (una fila por caja), por_ventana.csv (caja × ventana),
  por_celula.csv, msd.csv, vacf.csv, ead_tau.csv, corr_espacial.csv,
  sensibilidad_paso.csv, parametros.json

Uso:
  venv/bin/python scripts/36_estadisticas_propio.py 2026-10-02_1618_prueba_7 [--max-grupo 3]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, cache_dir  # noqa: E402
import importlib  # noqa: E402

EST = importlib.import_module("16_estadisticas")

VENTANA_H = 12


def registrar(um, dt_s):
    """El experimento como un dataset más, con los parámetros de brightfield
    (mismo intervalo de 5 min y escala parecida)."""
    DATASETS["propio"] = {"um_per_px": um, "dt_s": dt_s, "area_min_um2": 50.0}
    EST.PARAMS["propio"] = dict(paso=1, msd_max_lag=60, alfa_corto=(5, 30), alfa_largo=(60, 240), max_hueco=3)
    EST.PASOS_SENSIB["propio"] = [1, 2, 3, 4]


def _tarea(args):
    um, dt_s = args[-2:]
    registrar(um, dt_s)
    return EST.analizar_pelicula(args[:-2])


def preparar(tr):
    tr = tr.copy()
    tr["aspect_ratio"] = tr["major_axis_length"] / tr["minor_axis_length"].replace(0, np.nan)
    tr["perimetro_um"] = tr["perimeter"] * np.sqrt(tr["area_um2"] / tr["area"].clip(lower=1))
    tr["circularidad"] = (4 * np.pi * tr["area_um2"] / tr["perimetro_um"].clip(lower=1e-6) ** 2).clip(upper=1)
    tr["borde"] = tr["borde"].astype(bool)
    return tr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experimento")
    ap.add_argument("--max-grupo", type=int, default=None)
    ap.add_argument("--ventana-h", type=float, default=VENTANA_H)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    exp, mg = args.experimento, args.max_grupo
    an = cache_dir("analisis", "propio") / exp
    masks = cache_dir("masks", "propio") / exp
    cams = sorted(p.name for p in an.glob("cam*") if (p / "resumen.json").exists())
    out = an / "estadisticas" / (f"hasta_{mg}" if mg else "")
    out.mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(masks / "segmentacion.csv").drop_duplicates(["camara", "fecha"], keep="last")

    tareas, info = [], []
    for i, cam in enumerate(cams, start=1):
        d = an / cam / (f"hasta_{mg}" if mg else "")
        res = json.loads((d / "resumen.json").read_text(encoding="utf-8"))
        um, dt_s = res["um_por_px"], res["intervalo_min"] * 60
        tr = preparar(pd.read_csv(d / "tracks.csv"))
        tr["movie"] = i
        z = np.load(next((masks / cam).glob("*.npz")))
        h, w = z["masks"].shape
        area_mm2 = h * w * um ** 2 / 1e6
        cel_ciclo = seg[seg["camara"] == cam]["celulas"].to_numpy()
        tareas.append((("propio", i, tr, float(cel_ciclo.mean()), area_mm2, um, dt_s)))
        info.append({"movie": i, "camara": cam, "ventana": "completo", "t0_h": 0.0,
                     "t1_h": float(tr["t_s"].max() / 3600)})
        # ventanas de tiempo
        frames_v = int(round(args.ventana_h * 3600 / dt_s))
        nv = int(np.ceil((tr["frame"].max() + 1) / frames_v))
        for k in range(nv):
            sub = tr[(tr["frame"] >= k * frames_v) & (tr["frame"] < (k + 1) * frames_v)]
            if sub["track_id"].nunique() < 5:
                continue
            mv = i * 100 + k + 1
            sub = sub.assign(movie=mv)
            n_ciclo = cel_ciclo[k * frames_v:(k + 1) * frames_v]
            tareas.append(("propio", mv, sub, float(n_ciclo.mean()) if len(n_ciclo) else np.nan, area_mm2, um, dt_s))
            info.append({"movie": mv, "camara": cam, "ventana": k + 1, "t0_h": k * args.ventana_h,
                         "t1_h": (k + 1) * args.ventana_h})
    registrar(*tareas[0][-2:])
    t0 = time.time()
    print(f"{len(tareas)} análisis ({len(cams)} cajas + ventanas de {args.ventana_h:g} h)…", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(_tarea, tareas))
    info = pd.DataFrame(info)
    peli = pd.DataFrame([r["peli"] for r in res]).merge(info, on="movie")
    peli[peli["ventana"] == "completo"].to_csv(out / "por_pelicula.csv", index=False)
    peli[peli["ventana"] != "completo"].to_csv(out / "por_ventana.csv", index=False)
    nombres = {"cel": "por_celula", "msd": "msd", "vacf": "vacf", "fps": "fps", "ead": "ead_tau",
               "corr": "corr_espacial", "sens": "sensibilidad_paso", "pasos": "pasos_muestra"}
    for k, nombre in nombres.items():
        partes = [r[k] for r in res if r[k] is not None and len(r[k])]
        if partes:
            pd.concat(partes, ignore_index=True).to_csv(out / f"{nombre}.csv", index=False)
    (out / "parametros.json").write_text(json.dumps(
        {**{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in EST.COMUN.items()},
         **EST.PARAMS["propio"], "ventana_h": args.ventana_h, "max_grupo": mg,
         "camaras": {str(i): c for i, c in enumerate(cams, start=1)}}, indent=2))
    cols = ["camara", "ventana", "n_segmentos", "rapidez_um_min", "alfa_corto", "alfa_largo",
            "ead1_cel_corr", "se_fps_completa_corr", "prw_P_min", "densidad_cel_mm2"]
    print(peli[cols].round(3).to_string())
    print(f"listo en {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
