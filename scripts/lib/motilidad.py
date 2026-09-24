"""
Medidas clásicas de motilidad (por segmento de trayectoria uniforme).

Cambios de fondo respecto de 04_estadisticas.py (v1) y por qué:

- PRW CON término de ruido de localización:
      MSD(tau) = 4 D [tau - P (1 - e^{-tau/P})] + 4 sigma^2
  v1 ajustaba el PRW sin el 4 sigma^2. El error del centroide agrega una
  constante a TODO el MSD, que a lags cortos se confunde con movimiento
  difusivo y achica artificialmente el tiempo de persistencia P. Es la
  razón más probable de los 4 ajustes "degenerados" (P ~ 0) de v1: con el
  ruido como parámetro libre ese colapso no hace falta.
- Direccionalidad a ventana FIJA de tiempo. El cociente distancia neta /
  recorrido baja solo al alargarse la trayectoria (cualquier caminata
  tiende a 0 con el tiempo), así que comparar células de distinta
  duración mezcla duración con persistencia. Acá se calcula en ventanas de
  duración fija (1 h) y se promedia.
- MSD promediado en tiempo y ensamble (TEA-MSD) por película, con piso de
  pares por lag.
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


def msd_segmento(xy, max_lag):
    n = len(xy)
    L = min(max_lag, n - 1)
    out = np.full(max_lag, np.nan)
    cnt = np.zeros(max_lag, int)
    for lag in range(1, L + 1):
        d = xy[lag:] - xy[:-lag]
        out[lag - 1] = (d ** 2).sum(1).mean()
        cnt[lag - 1] = len(d)
    return out, cnt


def tea_msd(segs, max_lag, min_pares=50):
    """MSD promediado en tiempo y ensamble: suma de desplazamientos^2 sobre
    todos los pares de todas las células / número de pares, por lag."""
    s = np.zeros(max_lag)
    c = np.zeros(max_lag)
    n_cel = np.zeros(max_lag)
    for xy in segs:
        m, k = msd_segmento(xy, max_lag)
        ok = k > 0
        s[ok] += m[ok] * k[ok]
        c[ok] += k[ok]
        n_cel[ok] += 1
    msd = np.where(c >= min_pares, s / np.maximum(c, 1), np.nan)
    return msd, c, n_cel


def prw_msd(tau, D, P, sig2):
    return 4 * D * (tau - P * (1 - np.exp(-tau / P))) + 4 * sig2


def ajustar_prw(tau, msd):
    ok = np.isfinite(msd) & (msd > 0)
    tau, msd = tau[ok], msd[ok]
    if len(tau) < 5:
        return dict(D=np.nan, P=np.nan, sigma=np.nan, S=np.nan, r2=np.nan)
    p0 = [msd[-1] / (4 * tau[-1]), tau[len(tau) // 4], max(msd[0] / 8, 1e-4)]
    try:
        popt, _ = curve_fit(prw_msd, tau, msd, p0=p0, sigma=msd,  # errores relativos (escala log)
                            bounds=([1e-8, tau[0] * 0.05, 0], [np.inf, tau[-1] * 20, np.inf]),
                            maxfev=20000)
    except Exception:
        return dict(D=np.nan, P=np.nan, sigma=np.nan, S=np.nan, r2=np.nan)
    D, P, sig2 = popt
    pred = prw_msd(tau, *popt)
    r2 = 1 - np.sum((np.log(msd) - np.log(pred)) ** 2) / np.sum((np.log(msd) - np.log(msd).mean()) ** 2)
    # S = rapidez RMS del modelo (en 2D, D = S^2 P / 2)
    return dict(D=D, P=P, sigma=np.sqrt(sig2), S=np.sqrt(2 * D / P), r2=r2)


def alfa_local(tau, msd):
    """Pendiente log-log local (diferencias centradas)."""
    lt, lm = np.log(tau), np.log(msd)
    return np.gradient(lm, lt)


def alfa_rango(tau, msd, tmin, tmax):
    ok = (tau >= tmin) & (tau <= tmax) & np.isfinite(msd) & (msd > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(tau[ok]), np.log(msd[ok]), 1)[0])


def vacf_ensamble(vels, max_lag, min_pares=50):
    """VACF normalizada C(tau) = <v(t).v(t+tau)> / <v.v> (tau=0..max_lag)."""
    s = np.zeros(max_lag + 1)
    c = np.zeros(max_lag + 1)
    for v in vels:
        n = len(v)
        for lag in range(0, min(max_lag, n - 1) + 1):
            d = (v[:n - lag] * v[lag:]).sum(1)
            s[lag] += d.sum()
            c[lag] += len(d)
    C = np.where(c >= min_pares, s / np.maximum(c, 1), np.nan)
    return C / C[0], c


def fps_ensamble(vels, L, dt):
    """Espectro de potencias promedio sobre ventanas de L velocidades."""
    acc = []
    for v in vels:
        for i in range(len(v) // L):
            w = v[i * L:(i + 1) * L]
            V = np.fft.rfft(w - w.mean(0), axis=0) * dt
            acc.append((np.abs(V) ** 2).sum(1) / (L * dt))
    if not acc:
        return None, None
    f = np.fft.rfftfreq(L, d=dt)
    return f, np.mean(acc, axis=0)


def direccionalidad_ventana(xy, n_pasos):
    """Promedio de distancia neta / recorrido en ventanas no solapadas de
    n_pasos pasos."""
    vals = []
    for i in range(0, len(xy) - n_pasos, n_pasos):
        w = xy[i:i + n_pasos + 1]
        rec = np.linalg.norm(np.diff(w, axis=0), axis=1).sum()
        if rec > 0:
            vals.append(np.linalg.norm(w[-1] - w[0]) / rec)
    return float(np.mean(vals)) if vals else np.nan


def correlacion_espacial_velocidad(det_frame, bins_um):
    """Para las células de UN frame (x, y, vx, vy en um y um/s), correlación
    media de las DIRECCIONES (v^_i . v^_j) de pares, por bin de distancia.
    Devuelve (suma, conteo) por bin para acumular entre frames."""
    xy = det_frame[:, :2]
    v = det_frame[:, 2:4]
    nv = np.linalg.norm(v, axis=1)
    ok = nv > 0
    xy, u = xy[ok], v[ok] / nv[ok, None]
    s = np.zeros(len(bins_um) - 1)
    c = np.zeros(len(bins_um) - 1)
    if len(xy) < 2:
        return s, c
    tree = cKDTree(xy)
    pares = tree.query_pairs(bins_um[-1], output_type="ndarray")
    if len(pares) == 0:
        return s, c
    d = np.linalg.norm(xy[pares[:, 0]] - xy[pares[:, 1]], axis=1)
    dot = (u[pares[:, 0]] * u[pares[:, 1]]).sum(1)
    b = np.digitize(d, bins_um) - 1
    ok = (b >= 0) & (b < len(s))
    np.add.at(s, b[ok], dot[ok])
    np.add.at(c, b[ok], 1)
    return s, c
