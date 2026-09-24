"""
Validación de la implementación de las entropías (SE y EAD) con
trayectorias simuladas de caminata aleatoria persistente (PRW), como en
Liu et al. 2021/2024, y análisis de qué se puede medir con NUESTRO muestreo.

(a) Reproducción de Liu 2021 (Fig. 2e): SE del FPS para P = 0.5, 2, 8, 32
    min, dt = 0.2 min, 4800 pasos, S = 0.5 um/min. Paper: ~0.90, 0.75,
    0.60, 0.45.
(b) Reproducción de Liu 2024 (Fig. 2D-F): perfiles EAD(tau) y la relación
    de EAD1 y TL1 con P.
(c) Sesgo por longitud: EAD1 cruda vs corregida en función del número de
    ángulos por célula (m), para movimiento aleatorio y persistente. Es la
    justificación de la corrección propia.
(d) Régimen real: Delta = 5 min, 60 pasos, ruido de localización sigma y
    rapidez S del orden de lo medido. ¿Qué rango de P distinguen EAD1 y SE?

Salidas: resultados/v2/simulaciones/*.csv y figuras sim_*.png/pdf
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import res_dir  # noqa: E402
from lib import entropia as E  # noqa: E402


def prw(P, S, dt, N, rng, sigma=0.0):
    """Liu 2021 ec. 6-9: dx = a dx_prev + F W, a = 1 - dt/P, F = sqrt(S^2 dt^3 / P).
    Con dt > P el proceso discreto no está definido -> se simula con paso fino
    y se submuestrea."""
    sub = max(1, int(np.ceil(5 * dt / P)))
    h = dt / sub
    a = 1 - h / P
    F = np.sqrt(S ** 2 * h ** 3 / P)
    n = N * sub + 1
    d = np.zeros((n, 2))
    d[0] = rng.standard_normal(2) * S * h / np.sqrt(2)
    w = rng.standard_normal((n, 2)) * F
    for i in range(1, n):
        d[i] = a * d[i - 1] + w[i]
    xy = np.cumsum(d, 0)[::sub]
    return xy + sigma * rng.standard_normal(xy.shape)


def tarea_a(P):
    rng = np.random.default_rng(int(P * 100))
    se, eads = [], []
    for _ in range(100):
        xy = prw(P, 0.5, 0.2, 4800, rng)
        v = np.diff(xy, axis=0) / 0.2
        se.append(E.se_fps(v))
        cr, co, _ = E.ead_perfil(v, 250, 36)
        eads.append(co)
    eads = np.array(eads)
    return {"P_min": P, "se_media": np.mean(se), "se_sd": np.std(se)}, \
        pd.DataFrame({"P_min": P, "tau_min": np.arange(1, 251) * 0.2,
                      "ead_media": np.nanmean(eads, 0), "ead_sd": np.nanstd(eads, 0)})


def tarea_c(args):
    P, m = args
    rng = np.random.default_rng(int(P * 10) + m)
    cr, co = [], []
    for _ in range(400):
        xy = prw(P, 1.0, 5.0, m + 1, rng)
        v = np.diff(xy, axis=0)
        th = E.angulos_lag(v, 1)
        e = E.entropia_norm(E.histograma_angulos(th, 12))
        cr.append(e)
        co.append(e / E.ead_nula(len(th), 12))
    return {"P_min": P, "m_angulos": m, "ead1_cruda": np.mean(cr), "ead1_corr": np.mean(co),
            "ead1_cruda_sd": np.std(cr), "ead1_corr_sd": np.std(co)}


def tarea_d(args):
    P, sigma = args
    rng = np.random.default_rng(int(P * 7) + int(sigma * 100))
    S = 0.85                    # um/min, orden de la rapidez medida en BF
    e1, se, tl = [], [], []
    for _ in range(300):
        xy = prw(P, S, 5.0, 60, rng, sigma=sigma)
        v = np.diff(xy, axis=0) / 5.0
        cr, co, _ = E.ead_perfil(v, 12, 12)
        e1.append(co[0])
        tl.append(E.tl1(co, 0.95) * 5)
        s, n = E.se_ventanas(v, 16)
        se.append(s / E.se_nula(16))
    return {"P_min": P, "sigma_um": sigma, "ead1_corr": np.nanmean(e1), "ead1_corr_sd": np.nanstd(e1),
            "se_corr": np.nanmean(se), "se_corr_sd": np.nanstd(se),
            "tl1_mediana_min": np.nanmedian(tl), "tl1_censurado": float(np.mean(np.isnan(tl)))}


def main():
    out = res_dir("simulaciones")
    with ProcessPoolExecutor(12) as ex:
        ra = list(ex.map(tarea_a, [0.5, 2, 8, 32]))
        rc = list(ex.map(tarea_c, [(P, m) for P in (0.01, 5, 20) for m in (10, 15, 20, 30, 50, 80, 120, 200, 400)]))
        rd = list(ex.map(tarea_d, [(P, s) for P in (1, 2.5, 5, 10, 15, 20, 30, 45, 60, 90, 120)
                                   for s in (0.0, 1.0, 2.0)]))
    a = pd.DataFrame([r[0] for r in ra])
    a["se_paper_aprox"] = [0.90, 0.75, 0.60, 0.45]
    a.to_csv(out / "a_reproduccion_liu2021.csv", index=False)
    pd.concat([r[1] for r in ra]).to_csv(out / "b_perfiles_ead_liu2024.csv", index=False)
    pd.DataFrame(rc).to_csv(out / "c_sesgo_longitud.csv", index=False)
    pd.DataFrame(rd).to_csv(out / "d_regimen_real.csv", index=False)
    print(a.to_string())
    print(pd.DataFrame(rc).round(3).to_string())
    print(pd.DataFrame(rd).round(3).to_string())


if __name__ == "__main__":
    main()
