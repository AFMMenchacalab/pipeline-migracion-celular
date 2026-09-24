"""
Medidas de persistencia basadas en entropía de Shannon, según:

 [SE]  Liu Y, Jiao Y, et al. "Shannon entropy for time-varying persistence
       of cell migration". Biophys J 120:2552-2565 (2021).
       doi:10.1016/j.bpj.2021.04.026
       - SE a partir del espectro de potencias de Fourier (FPS) de la
         velocidad: p_k = FPS(f_k)/sum(FPS), H = -sum p log2 p, SE = H/log2(n).
         SE=1 -> movimiento difusivo ("aleatorio"); SE=0 -> balístico.
       - SE variable en el tiempo usando la transformada wavelet continua
         (Morlet, w0=6) en vez de Fourier: en cada instante se normaliza el
         espectro local (WPS) sobre las escalas y se calcula SE(t),
         descartando lo que cae dentro del cono de influencia (COI).

 [EAD] Liu Y, Jiao Y, et al. "An entropy-based approach for assessing the
       directional persistence of cell migration". Biophys J 123:730-744
       (2024). doi:10.1016/j.bpj.2024.02.010
       - Ángulo entre v_i y v_{i+tau} en [-pi, pi], histograma en n bins,
         EAD(tau) = H/log2(n). Parámetros conjuntos TL1 (lag al que la EAD
         llega al umbral, "cuánto tarda en perderse la dirección") y EAD1
         (EAD a tau=1). EAD variable en el tiempo con ventana deslizante.

MEJORA METODOLÓGICA propia (documentada en el reporte): corrección por
tamaño de muestra. La entropía de un histograma estimada con pocas muestras
está sesgada hacia ABAJO: con m = 20 ángulos repartidos en 36 bins, aun si
los ángulos son perfectamente aleatorios la EAD no puede superar
log2(20)/log2(36) = 0.84, y su valor esperado es ~0.78. En los papers las
trayectorias son largas (cientos de pasos), pero acá una célula típica
tiene 20-100 pasos: sin corregir, la EAD por célula mediría "cuán corta es
la trayectoria" tanto como "cuán persistente es". Por eso además del valor
crudo (comparable con los papers) se reporta la versión CORREGIDA:

    EAD_corr = EAD_cruda / E[EAD de m ángulos uniformes en n bins]

(el denominador se calcula por Monte Carlo y se cachea por (m, n)). Con
esto EAD_corr ~ 1 para movimiento sin dirección preferida,
independientemente de m. Lo mismo para SE: se divide por la SE esperada
de una velocidad de ruido blanco con la misma cantidad de puntos.
"""
from functools import lru_cache

import numpy as np

_RNG_SEED = 12345


# ---------------------------------------------------------------- entropía base
def entropia_norm(counts):
    counts = np.asarray(counts, float)
    n = len(counts)
    tot = counts.sum()
    if tot <= 0 or n < 2:
        return np.nan
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum() / np.log2(n))


@lru_cache(maxsize=None)
def ead_nula(m, n_bins, reps=4000):
    """E[entropía normalizada] de m ángulos iid uniformes en n_bins."""
    if m < 2:
        return np.nan
    rng = np.random.default_rng(_RNG_SEED + m * 1000 + n_bins)
    idx = rng.integers(0, n_bins, size=(reps, m)) + (np.arange(reps)[:, None] * n_bins)
    c = np.bincount(idx.ravel(), minlength=reps * n_bins).reshape(reps, n_bins) / m
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(c > 0, c * np.log2(c), 0).sum(1) / np.log2(n_bins)
    return float(h.mean())


# ---------------------------------------------------------------- ángulos
def angulos_lag(v, tau):
    """Ángulo con signo entre v_i y v_{i+tau} (radianes, [-pi, pi])."""
    if len(v) <= tau:
        return np.empty(0)
    a, b = v[:-tau], v[tau:]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = (a * b).sum(1)
    ok = (np.abs(a).sum(1) > 0) & (np.abs(b).sum(1) > 0)
    return np.arctan2(cross, dot)[ok]


def histograma_angulos(theta, n_bins):
    return np.histogram(theta, bins=n_bins, range=(-np.pi, np.pi))[0]


def ead_perfil(v, tau_max, n_bins, min_m=8):
    """EAD cruda y corregida para tau = 1..tau_max de UNA trayectoria.
    Devuelve arrays (cruda, corregida, m) de largo tau_max (NaN si m < min_m)."""
    cr = np.full(tau_max, np.nan)
    co = np.full(tau_max, np.nan)
    ms = np.zeros(tau_max, int)
    for t in range(1, tau_max + 1):
        th = angulos_lag(v, t)
        m = len(th)
        ms[t - 1] = m
        if m < min_m:
            continue
        e = entropia_norm(histograma_angulos(th, n_bins))
        cr[t - 1] = e
        co[t - 1] = e / ead_nula(m, n_bins)
    return cr, co, ms


def tl1(perfil, umbral):
    """Primer lag (en unidades de paso, 1-based) en que el perfil llega al
    umbral. Si nunca llega, devuelve NaN (censurado)."""
    idx = np.nonzero(np.asarray(perfil) >= umbral)[0]
    return float(idx[0] + 1) if len(idx) else np.nan


def ead_temporal(v, ventana, n_bins):
    """EAD a tau=1 en ventana deslizante de `ventana` ángulos consecutivos.
    Devuelve (indice_centro, cruda, corregida)."""
    th = angulos_lag(v, 1)
    if len(th) < ventana:
        return np.empty(0), np.empty(0), np.empty(0)
    nul = ead_nula(ventana, n_bins)
    cen, cr = [], []
    for i in range(len(th) - ventana + 1):
        cr.append(entropia_norm(histograma_angulos(th[i:i + ventana], n_bins)))
        cen.append(i + ventana / 2 + 0.5)   # en unidades de paso desde el inicio
    cr = np.array(cr)
    return np.array(cen), cr, cr / nul


# ---------------------------------------------------------------- SE por Fourier
def se_fps(v):
    """SE normalizada del espectro de potencias de la velocidad (vectorial),
    con la media restada (autocovarianza, como la VAC del paper) y sin la
    componente k=0."""
    v = np.asarray(v, float)
    if len(v) < 6:
        return np.nan
    V = np.fft.rfft(v - v.mean(0), axis=0)
    P = (np.abs(V) ** 2).sum(1)[1:]
    return entropia_norm(P)


@lru_cache(maxsize=None)
def se_nula(n, reps=2000):
    """E[SE] de una velocidad de ruido blanco gaussiano con n puntos."""
    rng = np.random.default_rng(_RNG_SEED + n)
    return float(np.nanmean([se_fps(rng.standard_normal((n, 2))) for _ in range(reps)]))


def se_ventanas(v, L):
    """SE promedio sobre ventanas NO solapadas de exactamente L velocidades
    (misma cantidad de frecuencias para todas las células -> SE comparable
    entre células de distinta longitud de trayectoria)."""
    n = len(v) // L
    if n == 0:
        return np.nan, 0
    vals = [se_fps(v[i * L:(i + 1) * L]) for i in range(n)]
    return float(np.nanmean(vals)), n


# ---------------------------------------------------------------- wavelet
def cwt_morlet(x, dt, dj=0.25, s0=None, w0=6.0):
    """Transformada wavelet continua de Morlet (Torrence & Compo 1998),
    x real o complejo. Devuelve (W[escala, t], escalas, periodos, coi)."""
    x = np.asarray(x)
    N = len(x)
    s0 = s0 or 2 * dt
    J = int(np.floor(np.log2(N * dt / s0) / dj))
    escalas = s0 * 2 ** (dj * np.arange(J + 1))
    npad = int(2 ** np.ceil(np.log2(N)) * 2)
    xf = np.fft.fft(x - x.mean(), n=npad)
    k = 2 * np.pi * np.fft.fftfreq(npad, d=dt)
    W = np.empty((len(escalas), N), complex)
    for i, s in enumerate(escalas):
        psi = np.pi ** -0.25 * np.sqrt(2 * np.pi * s / dt) * np.exp(-0.5 * (s * k - w0) ** 2) * (k > 0)
        W[i] = np.fft.ifft(xf * psi)[:N]
    ff = 4 * np.pi / (w0 + np.sqrt(2 + w0 ** 2))
    periodos = ff * escalas
    t_borde = np.minimum(np.arange(N), np.arange(N)[::-1]) * dt
    coi = ff / np.sqrt(2) * np.maximum(t_borde, 1e-12)  # periodo máximo válido en cada t
    return W, escalas, periodos, coi


def se_wavelet(v, dt, min_escalas=4, dj=0.25):
    """SE(t) a partir del espectro wavelet local de la velocidad vectorial:
    WPS = |W_vx|^2 + |W_vy|^2; en cada t se normaliza sobre las escalas FUERA
    del COI y se calcula la entropía normalizada por log2(#escalas válidas).
    Devuelve array de largo len(v) (NaN donde hay < min_escalas válidas)."""
    v = np.asarray(v, float)
    Wx, esc, per, coi = cwt_morlet(v[:, 0], dt, dj=dj)
    Wy, *_ = cwt_morlet(v[:, 1], dt, dj=dj)
    P = np.abs(Wx) ** 2 + np.abs(Wy) ** 2
    out = np.full(len(v), np.nan)
    for t in range(len(v)):
        ok = per <= coi[t]
        if ok.sum() >= min_escalas:
            out[t] = entropia_norm(P[ok, t])
    return out


@lru_cache(maxsize=None)
def se_wavelet_nula(n, dt, reps=200):
    rng = np.random.default_rng(_RNG_SEED + 7 * n)
    acc = np.array([se_wavelet(rng.standard_normal((n, 2)), dt) for _ in range(reps)])
    return np.nanmean(acc, axis=0)
