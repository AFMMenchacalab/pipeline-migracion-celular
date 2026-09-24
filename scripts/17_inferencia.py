"""
PASO 4 (v2): inferencia estadística sobre las tablas de 16_estadisticas.py.

  A) Dataset BF (y núcleos): una sola condición, 16 películas.
     - media, SD, SEM e IC95% bootstrap entre películas de cada métrica;
     - pruebas contra el nulo con las películas como réplicas (Wilcoxon
       de un grupo): ¿hay persistencia (EAD/SE corregidas < 1)?, ¿hay
       acoplamiento rapidez-persistencia (rho > 0)?, ¿las vecinas se
       mueven más alineadas que las lejanas?;
     - heterogeneidad entre campos: fracción de la varianza por célula
       explicada por la película (ICC);
     - relación con la densidad celular entre películas (Spearman, Holm).
  B) BF vs núcleos: concordancia película a película (Lin CCC, diferencia
     media, Wilcoxon pareado). Responde "¿las conclusiones cambian si se
     usa el pipeline sin marcador (brightfield) en vez de núcleos?".
  C) CAMAD: comparación entre sustratos (solo MDA-MB-231; exp8/9, de
     identidad celular incierta, van aparte; ver lib/config.py). Al nivel de experimento: Kruskal-Wallis y ANOVA por
     permutación. Al nivel de célula: modelo mixto con efecto aleatorio
     por experimento, contrastes contra Vidrio. Holm entre métricas.
     Aviso: n = 2-4 experimentos por sustrato -> potencia MUY baja; los
     resultados son exploratorios.

Uso: ../venv/bin/python 17_inferencia.py --bf dist_tam --sirdna dist --camad dist_tam
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import CAMAD_ORDEN_CONDICIONES, res_dir  # noqa: E402
from lib import inferencia as I  # noqa: E402

warnings.filterwarnings("ignore")

METRICAS = ["rapidez_um_min", "direccionalidad_1h", "prw_P_min", "prw_D_um2_min", "prw_S_um_min",
            "prw_sigma_um", "alfa_corto", "alfa_largo", "vacf_lag1", "cos_giro_medio", "rho_rapidez_cosgiro",
            "se_fps_completa", "se_fps_completa_corr", "se_ventana", "se_ventana_corr", "se_wavelet",
            "se_wavelet_corr", "ead1_cel", "ead1_cel_corr", "tl1_cel_mediana_min", "ead1_ens", "ead1_ens_corr",
            "tl1_ens_min", "ead_t_pct_0_03", "ead_t_pct_03_06", "ead_t_pct_06_1", "ead_t_corr_pct_06_1",
            "corr_dir_0_50um", "corr_dir_100_200um", "corr_se_vecinos", "corr_se_lejanos",
            "area_um2", "aspect_ratio", "circularidad", "solidez", "densidad_cel_mm2", "n_segmentos"]
METRICAS_CELULA = ["rapidez_um_min", "direccionalidad_1h", "se_ventana_corr", "se_wavelet_media_corr",
                   "ead1_corr", "tl1_min", "area_um2", "aspect_ratio", "circularidad"]


def seccion_un_grupo(ds, var):
    d = res_dir(ds, "estadisticas", var)
    peli = pd.read_csv(d / "por_pelicula.csv")
    cel = pd.read_csv(d / "por_celula.csv")
    out = res_dir("inferencia", ds)
    filas = [{"metrica": m, **I.resumen(peli[m])} for m in METRICAS if m in peli]
    tab = pd.DataFrame(filas)
    tab.to_csv(out / "resumen_bootstrap.csv", index=False)

    # pruebas contra el nulo, películas como réplicas
    pr = []
    def w1(nombre, x, mu, alt, que):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 5:
            return
        r = stats.wilcoxon(x - mu, alternative=alt)
        pr.append({"prueba": nombre, "hipotesis": que, "n_peliculas": len(x), "mediana": np.median(x),
                   "valor_nulo": mu, "p": r.pvalue})
    w1("EAD1 ensamble corregida < 1", peli.ead1_ens_corr, 1, "less", "hay persistencia direccional a 1 paso")
    w1("EAD1 por célula corregida < 1", peli.ead1_cel_corr, 1, "less", "persistencia en célula individual")
    w1("SE ventana corregida < 1", peli.se_ventana_corr, 1, "less", "espectro de velocidad no es ruido blanco")
    w1("SE wavelet corregida < 1", peli.se_wavelet_corr, 1, "less", "idem con wavelet")
    w1("cos(giro) medio > 0", peli.cos_giro_medio, 0, "greater", "la dirección tiende a mantenerse")
    w1("rho(rapidez, cos giro) > 0", peli.rho_rapidez_cosgiro, 0, "greater",
       "las células más rápidas giran menos (acoplamiento rapidez-persistencia, UCSP)")
    w1("correlación de dirección 0-50 um > 0", peli.corr_dir_0_50um, 0, "greater", "vecinas alineadas")
    if peli.corr_se_vecinos.notna().sum() >= 5:
        dd = (peli.corr_se_vecinos - peli.corr_se_lejanos).dropna()
        r = stats.wilcoxon(dd, alternative="greater")
        pr.append({"prueba": "corr SE(t) vecinas > lejanas (pareado)", "hipotesis": "migración correlacionada",
                   "n_peliculas": len(dd), "mediana": np.median(dd), "valor_nulo": 0, "p": r.pvalue})
    pr = pd.DataFrame(pr)
    if len(pr):
        pr["p_holm"] = I.holm(pr.p)
    pr.to_csv(out / "pruebas_nulo.csv", index=False)

    # heterogeneidad entre campos
    icc = pd.DataFrame([{"metrica": m, "icc_pelicula": I.icc_peliculas(cel, m),
                         "kruskal_p_entre_peliculas": stats.kruskal(*[g[m].dropna() for _, g in cel.groupby("movie")
                                                                       if g[m].notna().sum() > 2]).pvalue}
                        for m in METRICAS_CELULA if m in cel])
    icc.to_csv(out / "heterogeneidad_peliculas.csv", index=False)

    # relación con densidad (entre películas)
    rel = []
    for m in ["rapidez_um_min", "prw_P_min", "ead1_ens_corr", "se_ventana_corr", "direccionalidad_1h",
              "corr_dir_0_50um", "area_um2", "aspect_ratio"]:
        if m in peli:
            r = stats.spearmanr(peli.densidad_cel_mm2, peli[m], nan_policy="omit")
            rel.append({"metrica": m, "rho_vs_densidad": r.statistic, "p": r.pvalue})
    rel = pd.DataFrame(rel)
    rel["p_holm"] = I.holm(rel.p)
    rel.to_csv(out / "relacion_densidad.csv", index=False)
    return tab, pr, icc, rel


def seccion_bf_vs_nucleos(var_bf, var_nuc):
    a = pd.read_csv(res_dir("bf", "estadisticas", var_bf) / "por_pelicula.csv").set_index("movie")
    b = pd.read_csv(res_dir("sirdna", "estadisticas", var_nuc) / "por_pelicula.csv").set_index("movie")
    filas = []
    for m in METRICAS:
        if m not in a or m not in b or m in ("area_um2", "aspect_ratio", "circularidad", "solidez"):
            continue
        x, y = a[m], b.loc[a.index, m]
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue
        p = stats.wilcoxon(x[ok], y[ok]).pvalue if (x[ok] != y[ok]).any() else 1.0
        filas.append({"metrica": m, "media_bf": x[ok].mean(), "media_nucleos": y[ok].mean(),
                      "diferencia_media": (x[ok] - y[ok]).mean(), "diferencia_relativa_%": 100 * (x[ok] - y[ok]).mean() / abs(y[ok].mean()),
                      "pearson_r": stats.pearsonr(x[ok], y[ok]).statistic, "ccc_lin": I.ccc_lin(x[ok], y[ok]),
                      "p_wilcoxon_pareado": p, "n": int(ok.sum())})
    t = pd.DataFrame(filas)
    t["p_holm"] = I.holm(t.p_wilcoxon_pareado)
    t.to_csv(res_dir("inferencia") / "bf_vs_nucleos.csv", index=False)
    return t


def seccion_camad(var):
    d = res_dir("camad", "estadisticas", var)
    peli = pd.read_csv(d / "por_pelicula.csv")
    cel = pd.read_csv(d / "por_celula.csv")
    out = res_dir("inferencia", "camad")
    mda = peli[peli.linea == "MDA-MB-231"]
    celm = cel[cel.condicion != "Matriz 231 (exp8-9)"]
    # descriptivo por condición (experimentos como réplicas)
    desc = []
    for cond in CAMAD_ORDEN_CONDICIONES:
        g = peli[peli.condicion == cond]
        for m in METRICAS:
            if m in g:
                desc.append({"condicion": cond, "metrica": m, **I.resumen(g[m])})
    pd.DataFrame(desc).to_csv(out / "descriptivo_por_condicion.csv", index=False)
    # omnibus a nivel experimento
    omni = []
    for m in METRICAS:
        if m not in mda or mda[m].notna().sum() < 6:
            continue
        grupos = [g[m].dropna().to_numpy() for _, g in mda.groupby("condicion") if g[m].notna().sum() > 0]
        try:
            kw = stats.kruskal(*grupos).pvalue
        except ValueError:
            kw = np.nan
        f, pperm = I.anova_permutacion(mda[m], mda.condicion, n_perm=5000)
        omni.append({"metrica": m, "kruskal_p": kw, "anova_perm_F": f, "anova_perm_p": pperm,
                     "n_experimentos": int(mda[m].notna().sum())})
    omni = pd.DataFrame(omni)
    omni["kruskal_p_holm"] = I.holm(omni.kruskal_p)
    omni["anova_perm_p_holm"] = I.holm(omni.anova_perm_p)
    omni.to_csv(out / "omnibus_experimentos.csv", index=False)
    # modelo mixto por célula
    mm, varc = [], []
    for m in METRICAS_CELULA:
        if m not in celm:
            continue
        r = I.modelo_mixto(celm, m, "condicion", "Vidrio")
        if r is None:
            continue
        tab, ve, vr = r
        mm.append(tab)
        varc.append({"metrica": m, "var_entre_experimentos": ve, "var_residual": vr,
                     "icc": ve / (ve + vr) if np.isfinite(ve) else np.nan})
    if mm:
        mm = pd.concat(mm, ignore_index=True)
        mm["p_holm"] = I.holm(mm.p)
        mm.to_csv(out / "modelo_mixto_vs_vidrio.csv", index=False)
    pd.DataFrame(varc).to_csv(out / "componentes_varianza.csv", index=False)
    return omni, mm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf")
    ap.add_argument("--sirdna")
    ap.add_argument("--camad")
    args = ap.parse_args()
    if args.bf:
        tab, pr, icc, rel = seccion_un_grupo("bf", args.bf)
        print("== BF ==\n", tab.round(4).to_string(), "\n", pr.round(5).to_string(), "\n", icc.round(3).to_string(),
              "\n", rel.round(4).to_string())
    if args.sirdna:
        tab, pr, icc, rel = seccion_un_grupo("sirdna", args.sirdna)
        print("== núcleos ==\n", tab.round(4).to_string(), "\n", pr.round(5).to_string())
    if args.bf and args.sirdna:
        print("== BF vs núcleos ==\n", seccion_bf_vs_nucleos(args.bf, args.sirdna).round(4).to_string())
    if args.camad:
        omni, mm = seccion_camad(args.camad)
        print("== CAMAD ==\n", omni.round(4).to_string())
        if isinstance(mm, pd.DataFrame):
            print(mm.round(4).to_string())


if __name__ == "__main__":
    main()
