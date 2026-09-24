"""
PASO 2 (v2): detecciones + tracking, para cualquier dataset, en paralelo
(una película por proceso: 16 películas -> 16 procesos a la vez).

1) Detecciones: regionprops de cada máscara (centroide, área, perímetro,
   ejes, solidez, excentricidad, orientación, si toca el borde). Se
   guardan una sola vez por dataset en resultados/v2/<ds>/detecciones.csv.gz
   (se reusan para todas las variantes de tracking).
2) Tracking con laptrack (ver lib/tracking.py), parámetros en micrómetros.

Variantes con nombre (--variante) para la validación contra núcleos
(15_validacion_tracking.py); la elegida se usa con --variante final:
    v1         cutoff 50 px (=32.5 um en BF), gap 2 frames / 75 px, sin tamaño ni mitosis (igual a 02_tracking.py)
    dist       cutoff físico, gap 3 frames
    dist_tam   + penalización de cambio de tamaño
    dist_tam_corte  + corte de trayectorias en saltos anómalos (ver lib/tracking.py)

Uso:
    ../venv/bin/python 14_tracking.py --dataset bf --variante dist_tam_div
    ../venv/bin/python 14_tracking.py --dataset bf --todas
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, res_dir  # noqa: E402
from lib.fuentes import listar_frames  # noqa: E402
from lib.tracking import detecciones_pelicula, trackear, a_micrometros, cortar_saltos  # noqa: E402


def variantes(dataset):
    um = DATASETS[dataset]["um_per_px"]
    dt_min = DATASETS[dataset]["dt_s"] / 60
    # escala de desplazamiento máximo por frame: 25 um en 5 min (~5 um/min,
    # varias veces la rapidez máxima de MDA-MB-231); en CAMAD (0.5 min)
    # proporcional pero con piso de 6 um (media célula).
    link = max(6.0, 25.0 * dt_min / 5.0)
    base = {
        "v1": dict(max_link_um=50 * um, gap_frames=2, gap_link_um=75 * um, size_weight=0.0, split_um=None),
        "dist": dict(max_link_um=link, gap_frames=3, gap_link_um=link * 1.5, size_weight=0.0, split_um=None),
        "dist_tam": dict(max_link_um=link, gap_frames=3, gap_link_um=link * 1.5, size_weight=1.0, split_um=None),
        "dist_tam_corte": dict(max_link_um=link, gap_frames=3, gap_link_um=link * 1.5, size_weight=1.0,
                               split_um=None, cortar=True),
        "dist_corte": dict(max_link_um=link, gap_frames=3, gap_link_um=link * 1.5, size_weight=0.0,
                           split_um=None, cortar=True),
    }
    if dataset == "camad":
        # 30 s/frame: huecos de hasta 6 frames (3 min) siguen siendo cortos
        for v in base.values():
            v["gap_frames"] = 6 if v["gap_frames"] == 3 else v["gap_frames"]
        # Variante usada en CAMAD (decidida con el diagnóstico de exp1): Cellpose
        # pierde células durante tramos largos (cuando la trayectoria se corta, la
        # célula reaparece a ~5 um una mediana de 20 frames = 10 min después), y la
        # penalización por tamaño fragmenta más las trayectorias porque durante la
        # adhesión el área cambia mucho (62 trayectorias con tamaño vs 34 sin él en
        # exp1). Se cierran huecos de hasta 20 frames (10 min) dentro de 8 um:
        # a ~0.5 um/min una célula recorre ~5 um en 10 min, y en los campos
        # dispersos la vecina más cercana está a ~50-65 um (en exp4/5, densos,
        # a ~13-15 um; ahí el riesgo de confundir células es mayor y se advierte).
        base["camad_huecos"] = dict(max_link_um=6.0, gap_frames=20, gap_link_um=8.0, size_weight=0.0, split_um=None)
    return base


def _det(args):
    ds, mv = args
    return detecciones_pelicula(ds, mv)


def _trk(args):
    det, ds, params = args
    params = dict(params)
    cortar = params.pop("cortar", False)
    tr, sp = trackear(det, ds, **params)
    n_cortes = 0
    if cortar:
        tr, n_cortes = cortar_saltos(tr, ds)
    return tr, sp, n_cortes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--variante", default=None)
    ap.add_argument("--todas", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    ds = args.dataset
    out = res_dir(ds)

    det_path = out / "detecciones.csv.gz"
    movies = sorted({it["movie"] for it in listar_frames(ds)})
    t0 = time.time()
    if det_path.exists():
        det = pd.read_csv(det_path)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            det = pd.concat(list(ex.map(_det, [(ds, m) for m in movies])), ignore_index=True)
        det.to_csv(det_path, index=False)
        print(f"[{ds}] detecciones: {len(det)} en {time.time() - t0:.0f}s", flush=True)

    vs = variantes(ds)
    nombres = list(vs) if args.todas else [args.variante]
    for nombre in nombres:
        params = vs[nombre]
        t1 = time.time()
        grupos = [(g, ds, params) for _, g in det.groupby("movie")]
        with ProcessPoolExecutor(args.workers) as ex:
            res = list(ex.map(_trk, grupos))
        tr = pd.concat([r[0] for r in res], ignore_index=True)
        sp = [r[1] for r in res if len(r[1])]
        sp = pd.concat(sp, ignore_index=True) if sp else pd.DataFrame(
            columns=["parent_track_id", "child_track_id", "movie"])
        tr = a_micrometros(tr, ds)
        d = res_dir(ds, "tracking", nombre)
        tr.to_csv(d / "tracks.csv.gz", index=False)
        sp.to_csv(d / "divisiones.csv", index=False)
        (d / "parametros.json").write_text(json.dumps(params, indent=2))
        resumen = tr.groupby("movie").agg(tracks=("track_id", "nunique"), detecciones=("frame", "size"))
        resumen.to_csv(d / "resumen.csv")
        n_cortes = sum(r[2] for r in res)
        print(f"[{ds}/{nombre}] {tr.track_id.nunique()} tracks, {n_cortes} cortes por salto, "
              f"{time.time() - t1:.0f}s  params={params}", flush=True)


if __name__ == "__main__":
    main()
