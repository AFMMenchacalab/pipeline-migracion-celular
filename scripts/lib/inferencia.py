"""
Herramientas de inferencia usadas por 17_inferencia.py.

Principio (igual que v1 desde el arreglo #6): la réplica independiente es la
película / experimento. Las células de un mismo campo comparten pocillo,
iluminación, densidad y hasta se empujan entre sí, así que tratarlas como
independientes (n = miles) produce p-valores ridículamente chicos. Por eso:
  - intervalos de confianza por bootstrap sobre películas;
  - comparaciones entre condiciones al nivel de experimento (Kruskal-Wallis
    y ANOVA por permutación, válidos con n chico);
  - modelo mixto (efecto aleatorio por experimento) cuando se usan valores
    por célula, que es la forma correcta de usar toda la información sin
    pseudo-replicar.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

RNG = np.random.default_rng(0)


def bootstrap_ic(x, n_boot=5000, alpha=0.05, fn=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    b = fn(x[idx], axis=1)
    return tuple(np.quantile(b, [alpha / 2, 1 - alpha / 2]))


def resumen(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0, media=np.nan, sd=np.nan, sem=np.nan, ic95_inf=np.nan, ic95_sup=np.nan, mediana=np.nan)
    lo, hi = bootstrap_ic(x)
    return dict(n=len(x), media=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                sem=x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                ic95_inf=lo, ic95_sup=hi, mediana=np.median(x))


def holm(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    orden = np.argsort(pv)
    m = len(pv)
    adj = np.empty(m)
    run = 0
    for r, i in enumerate(orden):
        run = max(run, (m - r) * pv[i])
        adj[i] = min(1.0, run)
    out[ok] = adj
    return out


def ccc_lin(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    sxy = np.cov(a, b, ddof=1)[0, 1]
    return 2 * sxy / (a.var(ddof=1) + b.var(ddof=1) + (a.mean() - b.mean()) ** 2)


def anova_permutacion(valores, grupos, n_perm=20000):
    """p-valor de la F de un ANOVA de una vía por permutación de etiquetas
    (al nivel de experimento). No asume normalidad; válido con n chico."""
    v = np.asarray(valores, float)
    g = np.asarray(grupos)
    ok = np.isfinite(v)
    v, g = v[ok], g[ok]
    niveles = np.unique(g)
    if len(niveles) < 2:
        return np.nan, np.nan

    def F(gg):
        media = v.mean()
        ssb = sum((gg == k).sum() * (v[gg == k].mean() - media) ** 2 for k in niveles)
        ssw = sum(((v[gg == k] - v[gg == k].mean()) ** 2).sum() for k in niveles)
        dfb, dfw = len(niveles) - 1, len(v) - len(niveles)
        return (ssb / dfb) / (ssw / dfw) if ssw > 0 else np.inf

    f0 = F(g)
    cnt = sum(F(RNG.permutation(g)) >= f0 for _ in range(n_perm))
    return f0, (cnt + 1) / (n_perm + 1)


def modelo_mixto(df, y, grupo_col, ref, exp_col="movie"):
    """y ~ condición (referencia `ref`) + (1 | experimento), con statsmodels.
    Devuelve tabla de coeficientes vs la referencia."""
    import statsmodels.formula.api as smf
    d = df[[y, grupo_col, exp_col]].dropna().copy()
    d = d.rename(columns={y: "y", grupo_col: "g", exp_col: "e"})
    if d["g"].nunique() < 2 or len(d) < 10:
        return None
    try:
        m = smf.mixedlm(f"y ~ C(g, Treatment('{ref}'))", d, groups=d["e"]).fit(reml=True, method="lbfgs")
    except Exception:
        return None
    filas = []
    ci = m.conf_int()
    for nombre in m.params.index:
        if not nombre.startswith("C(g"):
            continue
        cond = nombre.split("[T.")[-1].rstrip("]")
        filas.append({"metrica": y, "condicion": cond, "vs": ref, "diferencia": m.params[nombre],
                      "ic95_inf": ci.loc[nombre, 0], "ic95_sup": ci.loc[nombre, 1], "p": m.pvalues[nombre]})
    var_exp = float(m.cov_re.iloc[0, 0]) if m.cov_re.size else np.nan
    return pd.DataFrame(filas), var_exp, float(m.scale)


def icc_peliculas(df, y, exp_col="movie"):
    """Fracción de la varianza (por célula) explicada por la película
    (correlación intraclase de un modelo de intercepto aleatorio)."""
    import statsmodels.formula.api as smf
    d = df[[y, exp_col]].dropna().rename(columns={y: "y", exp_col: "e"})
    if d["e"].nunique() < 3:
        return np.nan
    try:
        m = smf.mixedlm("y ~ 1", d, groups=d["e"]).fit(reml=True, method="lbfgs")
        vb = float(m.cov_re.iloc[0, 0])
        return vb / (vb + float(m.scale))
    except Exception:
        return np.nan
