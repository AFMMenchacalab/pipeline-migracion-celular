"""
Forma de cada célula con LOCO-EFA (Lobe Contribution Elliptic Fourier
Analysis, Sánchez-Corrales et al. 2018), tal como se usó en Vázquez-Cisneros
et al., Technologies 14(1):60 (2026), para obtener la Shape Complexity (SC).

Cada célula entra SOLA: se toma su máscara (masks == label), se recorta, se
saca su contorno y se coloca en un lienzo cuadrado de LIENZO × LIENZO px,
centrada y conservando la proporción (la caja de la célula ocupa MARGEN del
lienzo). Así todas comparten el mismo soporte y la SC no depende del
tamaño de la foto ni de la escala de la cámara.

Pasos (forma_celula):
  1. Contorno cerrado de la máscara (rellena huecos, queda la pieza más
     grande), escalado al lienzo y orientado en sentido antihorario.
  2. Coeficientes elípticos de Fourier (Kuhl y Giardina 1982) con el
     contorno parametrizado por longitud de arco.
  3. Cada armónico n es una elipse = dos círculos que giran en sentidos
     contrarios (radios λ+ₙ, λ−ₙ; fases ζ+ₙ, ζ−ₙ). Con números complejos
     z(θ) = Σ C_k e^{ikθ}: C_n es el círculo positivo y C_-n el negativo.
  4. Modo de lóbulos 𝓛ₙ (n ≥ 2): una forma con n lóbulos tiene simetría de
     orden n, es decir, solo trae los k ≡ 1 (mod n): k = 1+n y k = 1−n. Por
     eso en 𝓛ₙ contribuyen los armónicos n+1 y n−1 de la EFA (ecuación 1):
        𝓛ₙ = | C_{n+1} e^{i(n+1)t0} + C_{1−n} e^{i(1−n)t0} |
     con t0 = el instante en que la elipse del primer armónico pasa por su
     eje mayor (así el punto de partida no depende de dónde se empezó a
     recorrer el contorno ni de cómo está girada la célula).
     𝓛₁ es el círculo principal, |C_1| (tamaño de la célula).
  5. Descriptores:
        SC        = Σ_{n=N0}^{N} 𝓛ₙ        (N0 = 3, N = 50; ecuación 3)
        entropia  = −Σ fₗ log2 fₗ, fₗ = 𝓛ₗ / Σ 𝓛  (l = 1…N; ecuación 2)
        cd_px     = diferencia acumulada: Σ_{n=2}^{N} XOR(n), con XOR(n) = px
                    que difieren entre el contorno real y el reconstruido con
                    los modos ≤ n, en px del lienzo de LIENZO × LIENZO
        lobulos_dominante = n ≥ 2 con mayor λ+ₙ + λ−ₙ (número de lóbulos
                    más marcado; no depende del punto de partida)

Lo que NO está verificado contra el código original (MATLAB):
  · la convención exacta de las fases de la ecuación (1): aquí la fase de
    cada círculo se mide después de llevar el punto de partida al eje mayor
    de la elipse del primer armónico (ver arriba);
  · si en el artículo la caja de la célula llenaba el lienzo (aquí sí, al
    MARGEN) o se dejó el tamaño original;
  · el tratamiento de 𝓛₁ en la entropía (aquí es |C_1|, y entra en la suma).
Antes de comparar valores de SC con los del artículo hay que correr
`scripts/39_formas_loco_efa.py --probar` y, con las máscaras de las 24
células de la Fig. 5, comparar contra la Fig. 6a.

Máscaras a baja resolución (`reducir` > 1 en la segmentación) dan contornos
más toscos; los armónicos altos serían ruido de escalonado. `suavizar` (σ en
px de la máscara) suaviza antes de sacar el contorno.
"""
import numpy as np
import pandas as pd

LIENZO = 512
MARGEN = 0.9
N_MODOS = 50
N_SC0 = 3
LIENZO_XOR = 128
MIN_PUNTOS = 24
LOBULOS_GUARDADOS = 12      # columnas L1…L12 en la tabla


# ---------------------------------------------------------------- contorno
def contorno(mascara, lienzo=LIENZO, margen=MARGEN, suavizar=0.0):
    """Contorno (N, 2) en (x, y) de una máscara binaria de una sola célula,
    escalado al lienzo, centrado y antihorario. None si no hay contorno."""
    from scipy import ndimage as ndi
    from skimage.measure import find_contours
    m = ndi.binary_fill_holes(np.asarray(mascara) > 0)
    lab, n = ndi.label(m)
    if n > 1:
        tam = np.bincount(lab.ravel())
        tam[0] = 0
        m = lab == tam.argmax()
    m = np.pad(m, 2)
    img = ndi.gaussian_filter(m.astype(float), suavizar) if suavizar > 0 else m.astype(float)
    cs = find_contours(img, 0.5)
    if not cs:
        return None
    c = max(cs, key=len)
    xy = np.column_stack([c[:, 1], c[:, 0]])        # x = columna, y = fila
    if len(xy) > 1 and np.allclose(xy[0], xy[-1]):
        xy = xy[:-1]
    if len(xy) < MIN_PUNTOS:
        return None
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    ext = float((hi - lo).max())
    if ext <= 0:
        return None
    xy = (xy - (lo + hi) / 2) * (lienzo * margen / ext) + lienzo / 2
    return _antihorario(xy)


def _antihorario(xy):
    x, y = xy[:, 0], xy[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return xy[::-1].copy() if area < 0 else xy


# --------------------------------------------------------------------- EFA
def coef_efa(xy, n_arm):
    """Coeficientes elípticos de Fourier (a, b, c, d) de los armónicos
    1…n_arm y el centro (x0, y0) de un contorno cerrado (N, 2), con el
    contorno parametrizado por longitud de arco (Kuhl y Giardina)."""
    p = np.vstack([xy, xy[:1]])
    d = np.diff(p, axis=0)
    dt = np.hypot(d[:, 0], d[:, 1])
    ok = dt > 1e-12
    d, dt, p = d[ok], dt[ok], p[:-1][ok]
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    n = np.arange(1, n_arm + 1)[:, None]
    ang = 2 * np.pi * n * t[None, :] / T
    dc, ds = np.diff(np.cos(ang), axis=1), np.diff(np.sin(ang), axis=1)
    k = T / (2 * np.pi ** 2 * n[:, 0] ** 2)
    vx, vy = d[:, 0] / dt, d[:, 1] / dt
    a, b = k * (dc @ vx), k * (ds @ vx)
    c, dd = k * (dc @ vy), k * (ds @ vy)
    x0 = float(np.sum(dt * (p[:, 0] + (p[:, 0] + d[:, 0])) / 2) / T)
    y0 = float(np.sum(dt * (p[:, 1] + (p[:, 1] + d[:, 1])) / 2) / T)
    return a, b, c, dd, x0, y0


def circulos(a, b, c, d):
    """Cada armónico n como dos círculos: C_n gira en sentido positivo
    (radio |C_n| = λ+ₙ, fase arg C_n = ζ+ₙ) y C_-n en el negativo
    (λ−ₙ, ζ−ₙ). Devuelve (Cpos, Cneg), índice 0 = armónico 1."""
    return 0.5 * ((a + d) + 1j * (c - b)), 0.5 * ((a - d) + 1j * (c + b))


def modos_loco(Cpos, Cneg, n_modos=N_MODOS):
    """Amplitudes 𝓛ₙ (n = 1…n_modos) y radios λ+ₙ, λ−ₙ de los círculos que
    forman cada modo de lóbulos. Hacen falta n_modos + 1 armónicos."""
    n = np.arange(2, n_modos + 1)
    A, B = Cpos[n], Cneg[n - 2]                # k = n+1 y k = -(n-1)
    t0 = (np.angle(Cneg[0]) - np.angle(Cpos[0])) / 2
    L = np.empty(n_modos)
    L[0] = abs(Cpos[0])
    L[1:] = np.abs(A * np.exp(1j * (n + 1) * t0) + B * np.exp(-1j * (n - 1) * t0))
    lam_pos = np.concatenate([[abs(Cpos[0])], np.abs(A)])
    lam_neg = np.concatenate([[abs(Cneg[0])], np.abs(B)])
    return L, lam_pos, lam_neg


def shape_complexity(L, n0=N_SC0):
    return float(L[n0 - 1:].sum())


def entropia(L):
    f = L / L.sum()
    f = f[f > 0]
    return float(-(f * np.log2(f)).sum())


# --------------------------------------------------------------------- XOR
def _raster(x, y, lado):
    from skimage.draw import polygon
    m = np.zeros((lado, lado), bool)
    rr, cc = polygon(y, x, (lado, lado))
    m[rr, cc] = True
    return m


def diferencias_xor(xy, Cpos, Cneg, x0, y0, n_modos=N_MODOS, lienzo=LIENZO,
                    lienzo_xor=LIENZO_XOR, puntos=720):
    """XOR(n) para n = 2…n_modos: píxeles que difieren entre el contorno y su
    reconstrucción con los modos ≤ n, en px del lienzo de `lienzo`. Se calcula
    en un lienzo más chico (`lienzo_xor`) y se reescala."""
    f = lienzo_xor / lienzo
    ref = _raster(xy[:, 0] * f, xy[:, 1] * f, lienzo_xor)
    th = np.linspace(0, 2 * np.pi, puntos, endpoint=False)
    z = Cpos[0] * np.exp(1j * th) + Cpos[1] * np.exp(2j * th)    # modo 1 (+ armónico 2)
    xor = np.empty(n_modos - 1)
    for i, n in enumerate(range(2, n_modos + 1)):
        z = z + Cpos[n] * np.exp(1j * (n + 1) * th) + Cneg[n - 2] * np.exp(-1j * (n - 1) * th)
        rec = _raster((z.real + x0) * f, (z.imag + y0) * f, lienzo_xor)
        xor[i] = np.count_nonzero(ref ^ rec)
    return xor / f ** 2, float(ref.sum()) / f ** 2


# ------------------------------------------------------------------- célula
def forma_de_contorno(xy, n_modos=N_MODOS, con_xor=True, lienzo=LIENZO, lienzo_xor=LIENZO_XOR):
    a, b, c, d, x0, y0 = coef_efa(xy, n_modos + 1)
    Cpos, Cneg = circulos(a, b, c, d)
    L, lp, ln = modos_loco(Cpos, Cneg, n_modos)
    r = {"sc": shape_complexity(L), "entropia": entropia(L),
         "lobulos_dominante": int(np.argmax(lp[1:] + ln[1:]) + 2)}
    for n in range(1, min(LOBULOS_GUARDADOS, n_modos) + 1):
        r[f"L{n}"] = float(L[n - 1])
    if con_xor:
        xor, area = diferencias_xor(xy, Cpos, Cneg, x0, y0, n_modos, lienzo, lienzo_xor)
        r["cd_px"] = float(xor.sum())
        r["xor_final_rel"] = float(xor[-1] / max(area, 1.0))
        r["area_lienzo_px"] = area
    return r


def forma_celula(mascara, n_modos=N_MODOS, con_xor=True, lienzo=LIENZO, margen=MARGEN,
                 lienzo_xor=LIENZO_XOR, suavizar=0.0):
    """Descriptores LOCO-EFA de la máscara binaria de UNA célula (ya recortada
    o no). None si el contorno es demasiado chico."""
    xy = contorno(mascara, lienzo, margen, suavizar)
    if xy is None:
        return None
    r = forma_de_contorno(xy, n_modos, con_xor, lienzo, lienzo_xor)
    r["n_puntos_contorno"] = int(len(xy))
    return r


# ----------------------------------------------------------- tabla de tracks
def _ciclo(args):
    """Trabajador (un ciclo): máscaras en `ruta`, etiquetas a medir."""
    from scipy import ndimage as ndi
    ruta, etiquetas, kw = args
    m = np.load(ruta)["masks"]
    objs = ndi.find_objects(m)
    filas = []
    for lab in etiquetas:
        if lab < 1 or lab > len(objs) or objs[lab - 1] is None:
            continue
        r = forma_celula(m[objs[lab - 1]] == lab, **kw)
        if r is not None:
            filas.append({"label": int(lab), **r})
    return filas


def formas_de_tracks(tr, masks_dir, grupo_max=1, quitar_borde=True, procesos=1,
                     progreso=None, **kw):
    """Agrega a `tr` (tracks.csv de lib/propio.analizar: fecha, label, grupo,
    borde) las columnas LOCO-EFA de cada célula en cada ciclo. Solo se miden
    las células en grupos de hasta `grupo_max` y, con quitar_borde, las que no
    tocan el borde de la foto; el resto queda con NaN. kw → forma_celula."""
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    sel = tr
    if grupo_max is not None and "grupo" in tr.columns:
        sel = sel[sel["grupo"] <= grupo_max]
    if quitar_borde and "borde" in tr.columns:
        sel = sel[~sel["borde"].astype(bool)]
    masks_dir = Path(masks_dir)
    tareas = [(masks_dir / f"{fecha}.npz", g["label"].astype(int).tolist(), kw)
              for fecha, g in sel.groupby("fecha")]
    fechas = [t[0].stem for t in tareas]
    filas = []
    if procesos > 1 and len(tareas) > 1:
        with ProcessPoolExecutor(procesos) as ex:
            resultados = ex.map(_ciclo, tareas, chunksize=4)
            for i, (fecha, res) in enumerate(zip(fechas, resultados)):
                filas += [{"fecha": fecha, **r} for r in res]
                if progreso and i % 25 == 0:
                    progreso(f"forma de las células ({i}/{len(tareas)} ciclos)")
    else:
        for i, (fecha, t) in enumerate(zip(fechas, tareas)):
            filas += [{"fecha": fecha, **r} for r in _ciclo(t)]
            if progreso and i % 25 == 0:
                progreso(f"forma de las células ({i}/{len(tareas)} ciclos)")
    if not filas:
        return tr.copy()
    return tr.merge(pd.DataFrame(filas), on=["fecha", "label"], how="left")


# ------------------------------------------------------------- autoprueba
def _figura_sintetica(lobulos, eps, R=200.0, estira=1.0, n=1500, giro=0.0, inicio=0, centro=(0.0, 0.0)):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = R * (1 + eps * np.cos(lobulos * th)) if lobulos else np.full(n, R)
    z = (r * np.cos(th) * estira + 1j * r * np.sin(th)) * np.exp(1j * giro) + complex(*centro)
    return np.roll(np.column_stack([z.real, z.imag]), -inicio, axis=0)


def autoprueba():
    """Comprobaciones con formas sintéticas (sin datos). Devuelve la lista de
    fallos; vacía = todo bien. Se corre con 39_formas_loco_efa.py --probar."""
    fallos = []

    def revisar(ok, texto):
        print(("  ok  " if ok else "  FALLA  ") + texto)
        if not ok:
            fallos.append(texto)

    print("círculo: sin lóbulos")
    xy = _figura_sintetica(0, 0.0)
    r = forma_de_contorno(xy, con_xor=False)
    revisar(r["sc"] < 1e-3 * r["L1"], f"SC = {r['sc']:.2e} ≈ 0 frente a 𝓛₁ = {r['L1']:.1f}")

    print("figuras con m lóbulos: el modo dominante es m")
    for m in (3, 4, 5, 6, 8):
        xy = _figura_sintetica(m, 0.06)
        a, b, c, d, _, _ = coef_efa(xy, N_MODOS + 1)
        Cp, Cn = circulos(a, b, c, d)
        L, lp, ln = modos_loco(Cp, Cn)
        dom = int(np.argmax(lp[2:] + ln[2:]) + 3)      # n ≥ 3, como en la SC
        amp = float(lp[m - 1] + ln[m - 1])             # = 2·(Rε/2) = Rε si es exacto
        revisar(dom == m, f"m = {m}: modo dominante {dom}; λ+ + λ− = {amp:.1f} (Rε = {200 * 0.06:.1f})")
        revisar(abs(amp / (200 * 0.06) - 1) < 0.2, f"m = {m}: amplitud dentro de 20 % de Rε")

    print("más lóbulos de más amplitud → más SC")
    sc = [forma_de_contorno(_figura_sintetica(5, e), con_xor=False)["sc"] for e in (0.02, 0.05, 0.1)]
    revisar(sc[0] < sc[1] < sc[2], f"SC con ε = 0.02, 0.05, 0.10: {sc[0]:.1f} < {sc[1]:.1f} < {sc[2]:.1f}")

    print("invariancia: giro, traslación y punto de partida no cambian 𝓛ₙ")
    base = _figura_sintetica(3, 0.06, estira=1.3)
    # mismo contorno girado 1.1 rad, trasladado y empezando en otro punto
    z = (base[:, 0] + 1j * base[:, 1]) * np.exp(1.1j) + (40 - 25j)
    otra = np.roll(np.column_stack([z.real, z.imag]), -377, axis=0)
    L0 = forma_de_contorno(base, con_xor=False)
    L1 = forma_de_contorno(otra, con_xor=False)
    dif = max(abs(L0[k] - L1[k]) for k in ("sc", "entropia", "L1", "L3", "L5"))
    revisar(dif < 1e-6 * L0["L1"], f"diferencia máxima {dif:.2e}")

    print("contorno hacia el otro lado (horario) da lo mismo")
    L2 = forma_de_contorno(_antihorario(base[::-1]), con_xor=False)
    revisar(abs(L0["sc"] - L2["sc"]) < 1e-6 * L0["L1"], f"SC {L0['sc']:.3f} vs {L2['sc']:.3f}")

    print("XOR: con más modos el contorno reconstruido se parece más")
    xy = _figura_sintetica(5, 0.08, estira=1.2)
    xy = (xy - xy.mean(axis=0)) * (LIENZO * MARGEN / np.ptp(xy, axis=0).max()) + LIENZO / 2
    a, b, c, d, x0, y0 = coef_efa(xy, N_MODOS + 1)
    Cp, Cn = circulos(a, b, c, d)
    xor, area = diferencias_xor(xy, Cp, Cn, x0, y0)
    revisar(xor[-1] < xor[0] * 0.2, f"XOR(n=2) = {xor[0]:.0f} px → XOR(n=50) = {xor[-1]:.0f} px (área {area:.0f})")

    print("desde una máscara de píxeles (informativo, sin criterio)")
    from skimage.draw import polygon
    for m in (0, 3, 5):
        lado = 160
        p = _figura_sintetica(m, 0.1 if m else 0.0, R=55) + lado / 2
        mk = np.zeros((lado, lado), bool)
        rr, cc = polygon(p[:, 1], p[:, 0], mk.shape)
        mk[rr, cc] = True
        r = forma_celula(mk, lienzo_xor=128)
        print(f"     m = {m}: SC {r['sc']:.1f}, entropía {r['entropia']:.2f}, CD {r['cd_px']:.0f} px, "
              f"lóbulos dominantes {r['lobulos_dominante']}")
    return fallos
