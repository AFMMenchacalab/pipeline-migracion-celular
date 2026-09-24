"""
Arquetipos de forma: agrupar morfologías parecidas y describir cada grupo con
una geometría simple (círculo, elipse, huso, gota/abanico, triángulo,
estrella con protuberancias...).

Método:
  1. Contorno de cada célula (máscaras de Cellpose, sin volver a segmentar),
     sin las que tocan el borde de la imagen.
  2. Normalización: centro en el centroide, mismo tamaño (área 1), eje mayor
     horizontal, y orientación fijada con los momentos de tercer orden (el
     lado "más cargado" a la derecha / arriba) para que formas asimétricas
     como una gota o un triángulo queden todas mirando igual.
  3. Descriptor r(theta): distancia del centro al borde en 64 direcciones
     (para bordes con entrantes se toma el punto más lejano).
  4. Armónicos de Fourier de r(theta): A1 = un extremo más ancho (gota,
     abanico), A2 = alargamiento (elipse, huso), A3 = triangularidad,
     A4-A8 = protuberancias (estrella). Estos números son la "geometría".
  5. Agrupamiento: PCA de log r(theta) + mezcla de gaussianas (número de
     grupos elegido por BIC entre 4 y 9). Para cada grupo: forma media,
     espectro de armónicos y una geometría simplificada reconstruida con los
     armónicos dominantes; el nombre se asigna con reglas explícitas sobre
     esos armónicos.
  6. Relación con el movimiento (rapidez, persistencia) usando las películas
     como réplicas (prueba de Friedman), transiciones entre arquetipos cada
     5 min (cadena de Markov) y, en CAMAD, composición por sustrato en el
     tiempo.
  7. Forma promedio ALINEADA CON LA DIRECCIÓN DE MOVIMIENTO (frente a la
     derecha) para células rápidas vs lentas: muestra la asimetría
     frente-cola típica de la migración.

Salidas: resultados/v2/arquetipos/*.csv, figuras arq_*.png
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from skimage.measure import find_contours

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import CAMAD_CONDICION, CAMAD_ORDEN_CONDICIONES, DATASETS, cache_dir, res_dir  # noqa: E402
from lib.fuentes import cargar_mascara  # noqa: E402

N_ANG = 64
TH = np.linspace(-np.pi, np.pi, N_ANG, endpoint=False)
OUT = res_dir("arquetipos")
FIG = res_dir("figuras")


def contorno_normalizado(mask_bool, vel=None):
    """Devuelve (r_intrinseco[64], r_movimiento[64] o None, extra) de una célula."""
    pad = np.pad(mask_bool, 2)
    cs = find_contours(pad.astype(float), 0.5)
    if not cs:
        return None
    c = max(cs, key=len)[:, ::-1]              # (x, y)
    yy, xx = np.nonzero(pad)
    cx, cy = xx.mean(), yy.mean()
    area = len(xx)
    x, y = (c[:, 0] - cx) / np.sqrt(area), (c[:, 1] - cy) / np.sqrt(area)
    # momentos para el eje mayor
    px, py = (xx - cx) / np.sqrt(area), (yy - cy) / np.sqrt(area)
    cov = np.cov(np.vstack([px, py]))
    w, v = np.linalg.eigh(cov)
    ang = np.arctan2(v[1, 1], v[0, 1])        # eje mayor
    def rotar(xa, ya, a):
        ca, sa = np.cos(-a), np.sin(-a)
        return xa * ca - ya * sa, xa * sa + ya * ca
    xr, yr = rotar(x, y, ang)
    pxr, pyr = rotar(px, py, ang)
    # tercer momento: lado más cargado a la derecha (x) y arriba (y)
    if np.mean(pxr ** 3) < 0:
        xr, pxr = -xr, -pxr
    if np.mean(pyr ** 3) < 0:
        yr, pyr = -yr, -pyr

    def polar(xa, ya):
        t = np.arctan2(ya, xa)
        r = np.hypot(xa, ya)
        idx = np.clip(np.round((t + np.pi) / (2 * np.pi) * N_ANG).astype(int) % N_ANG, 0, N_ANG - 1)
        out = np.zeros(N_ANG)
        np.maximum.at(out, idx, r)
        # rellenar direcciones sin puntos por interpolación circular
        ok = out > 0
        if ok.sum() < N_ANG // 2:
            return None
        k = np.arange(N_ANG)
        out[~ok] = np.interp(k[~ok], np.r_[k[ok] - N_ANG, k[ok], k[ok] + N_ANG], np.tile(out[ok], 3))
        return out
    r_int = polar(xr, yr)
    r_mov = None
    if vel is not None and np.hypot(*vel) > 0:
        a_mov = np.arctan2(vel[1], vel[0])
        xm, ym = rotar(x, y, a_mov)
        r_mov = polar(xm, ym)
    return r_int, r_mov


def _pelicula(args):
    ds, var, movie, cada = args
    tr = pd.read_csv(res_dir(ds, "tracking", var) / "tracks.csv.gz",
                     usecols=["movie", "track_id", "frame", "label", "x", "y", "borde"])
    tr = tr[(tr.movie == movie)].sort_values(["track_id", "frame"])
    paso = 1 if ds == "bf" else 10
    g = tr.groupby("track_id")
    tr["vx"] = g.x.shift(-paso) - tr.x
    tr["vy"] = g.y.shift(-paso) - tr.y
    tr = tr[(~tr.borde) & (tr.frame % cada == 0)]
    filas, R, RM = [], [], []
    for f, gf in tr.groupby("frame"):
        m = cargar_mascara(ds, movie, f)
        from scipy.ndimage import find_objects
        sl = find_objects(m)
        for _, r in gf.iterrows():
            s = sl[int(r.label) - 1] if int(r.label) - 1 < len(sl) else None
            if s is None:
                continue
            crop = m[s] == r.label
            if crop.sum() < 30:
                continue
            vel = (r.vx, r.vy) if np.isfinite(r.vx) else None
            res = contorno_normalizado(crop, vel)
            if res is None or res[0] is None:
                continue
            filas.append((movie, int(r.track_id), int(f)))
            R.append(res[0])
            RM.append(res[1] if res[1] is not None else np.full(N_ANG, np.nan))
    return pd.DataFrame(filas, columns=["movie", "track_id", "frame"]), np.array(R), np.array(RM)


def armonicos(r):
    """Amplitud relativa de los armónicos 1..8 de r(theta) (normalizada por el término 0)."""
    F = np.fft.rfft(r, axis=-1) / N_ANG
    a0 = np.abs(F[..., 0])
    return np.abs(F[..., 1:9]) * 2 / a0[..., None]


def nombrar(A, ar):
    """Nombre geométrico a partir de los armónicos de la forma media."""
    a1, a2, a3 = A[0], A[1], A[2]
    alto = A[3:].sum()
    if a2 < 0.08 and a1 < 0.05 and a3 < 0.04:
        base = "redonda (círculo)"
    elif a3 > 0.06 and a3 > 0.5 * a2:
        base = "triangular (tres puntas)"
    elif a2 >= 0.25:
        base = "fusiforme (huso bipolar)" if A[3] > 0.04 else "alargada (elipse estirada)"
    elif a2 >= 0.08:
        base = "elíptica"
    else:
        base = "redondeada irregular"
    extras = []
    if a1 > 0.06:
        extras.append("con un extremo más ancho (gota / abanico)")
    if alto > 0.12 and "triangular" not in base:
        extras.append("con protuberancias")
    return base + (", " + ", ".join(extras) if extras else "")


def _superelipse(ar, n):
    a, b = np.sqrt(ar), 1 / np.sqrt(ar)
    return (np.abs(np.cos(TH) / a) ** n + np.abs(np.sin(TH) / b) ** n) ** (-1 / n)


def nombre_por_armonicos(A, alarg, solidez):
    """Reglas explícitas (calibradas mirando las formas medias):
    solidez < 0.72 -> ramificada; A3 >= 0.12 y A3 >= A2/2 -> triángulo;
    si no, según A2 (alargamiento): círculo / óvalo / elipse / huso; A1 alto
    -> un extremo más ancho; A3 o entrantes en husos -> protuberancia lateral."""
    a1, a2, a3 = A[0], A[1], A[2]
    if solidez < 0.72:
        return "ramificada (varias prolongaciones)", "superelipse"
    if a3 >= 0.17 and a3 >= 0.5 * a2:
        return f"triángulo redondeado ({alarg:.1f}:1)", "triangulo"
    if a3 >= 0.12 and a3 >= 0.5 * a2:
        return f"triángulo suave, tipo gota ({alarg:.1f}:1)", "triangulo"
    if a2 < 0.09:
        return "círculo", "superelipse"
    if a2 < 0.2:
        return f"óvalo ({alarg:.1f}:1)", "superelipse"
    if a2 < 0.45:
        return (f"elipse ({alarg:.1f}:1), un extremo más ancho" if a1 > 0.03 else f"elipse ({alarg:.1f}:1)"), \
            ("gota" if a1 > 0.03 else "superelipse")
    if a3 > 0.09:
        return f"huso con protuberancia lateral ({alarg:.1f}:1)", "triangulo"
    return (f"huso muy alargado ({alarg:.1f}:1)" if alarg > 4 else f"huso alargado ({alarg:.1f}:1)"), "superelipse"


def ajustar_geometria(r, solidez, familia=None):
    """Ajusta plantillas geométricas explícitas a r(theta) (en escala log, con
    factor de escala libre) y elige la de menor BIC (dentro de `familia` si se
    indica). Devuelve (nombre, r_ajustado, params)."""
    lr = np.log(r)
    cands = []
    def evalua(nombre, rt, k, params):
        lt = np.log(rt)
        c = np.mean(lr - lt)
        rss = np.sum((lr - lt - c) ** 2)
        bic = N_ANG * np.log(rss / N_ANG + 1e-12) + k * np.log(N_ANG)
        cands.append((bic, nombre, np.exp(lt + c), params))
    ars = np.linspace(1.0, 6.0, 51)
    for ar in ars:
        for n in (1.3, 1.6, 2.0, 2.6):
            base = _superelipse(ar, n)
            evalua("superelipse", base, 3, dict(ar=ar, n=n))
            for e in (0.08, 0.15, 0.25):
                evalua("gota", base * (1 + e * np.cos(TH)), 4, dict(ar=ar, n=n, e=e))
            for t in (0.08, 0.15, 0.25):
                for ph in np.linspace(0, 2 * np.pi / 3, 6, endpoint=False):
                    evalua("triangulo", base * (1 + t * np.cos(3 * TH - ph)), 5, dict(ar=ar, n=n, t=t))
    if familia is not None:
        cands = [c for c in cands if c[1] == familia] or cands
    bic, fam, rfit, par = min(cands, key=lambda x: x[0])
    ar, n = par["ar"], par["n"]
    if fam == "triangulo":
        nombre = "triángulo redondeado" + (f" alargado ({ar:.1f}:1)" if ar > 1.6 else "")
    elif fam == "gota":
        nombre = f"gota / huevo ({ar:.1f}:1, un extremo más ancho)" if ar > 1.2 else "gota corta (un lado más ancho)"
    elif ar < 1.15:
        nombre = "círculo"
    elif ar < 1.8:
        nombre = f"elipse ({ar:.1f}:1)"
    elif n <= 1.6:
        nombre = f"huso con puntas ({ar:.1f}:1)"
    else:
        nombre = f"elipse alargada ({ar:.1f}:1)"
    if solidez < 0.72:
        nombre += ", ramificada (prolongaciones)"
    return nombre, rfit, dict(familia=fam, **par)


def geometria_simple(A_complejo, k_max=3):
    """Reconstrucción de r(theta) con los armónicos 0..k_max (la 'geometría propuesta')."""
    F = np.zeros(N_ANG // 2 + 1, complex)
    F[:k_max + 1] = A_complejo[:k_max + 1]
    return np.fft.irfft(F, n=N_ANG)


def main():
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from lib import estilo as S
    from lib.estilo import plt
    vbf = (res_dir() / "variante_bf.txt").read_text().strip()
    vca = (res_dir() / "variante_camad.txt").read_text().strip()
    cf = cache_dir("arquetipos")
    if (cf / "r_theta.npz").exists() and (cf / "meta_bf.csv.gz").exists():
        z = np.load(cf / "r_theta.npz")
        Rbf, RMbf, Rca, RMca = z["Rbf"], z["RMbf"], z["Rca"], z["RMca"]
        mbf, mca = pd.read_csv(cf / "meta_bf.csv.gz"), pd.read_csv(cf / "meta_camad.csv.gz")
    else:
        tareas = [("bf", vbf, m, 1) for m in range(1, 17)] + [("camad", vca, m, 10) for m in range(1, 17)]
        with ProcessPoolExecutor(16) as ex:
            res = list(ex.map(_pelicula, tareas))
        def juntar(rr):
            meta = pd.concat([x[0] for x in rr], ignore_index=True)
            R = np.vstack([x[1] for x in rr if len(x[1])])
            RM = np.vstack([x[2] for x in rr if len(x[2])])
            return meta, R, RM
        mbf, Rbf, RMbf = juntar(res[:16])
        mca, Rca, RMca = juntar(res[16:])
        np.savez_compressed(cf / "r_theta.npz", Rbf=Rbf, RMbf=RMbf, Rca=Rca, RMca=RMca)
        mbf.to_csv(cf / "meta_bf.csv.gz", index=False)
        mca.to_csv(cf / "meta_camad.csv.gz", index=False)
    print("contornos BF:", len(Rbf), " CAMAD:", len(Rca))

    # ---- PCA + GMM sobre log r (BF; submuestra para ajustar, luego se clasifica todo)
    L = np.log(Rbf)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(L), min(40000, len(L)), replace=False)
    pca = PCA(n_components=10, random_state=0).fit(L[sub])
    Z = pca.transform(L)
    bics = {}
    for k in range(4, 10):
        gm = GaussianMixture(k, covariance_type="full", random_state=0, n_init=2).fit(Z[sub])
        bics[k] = gm.bic(Z[sub])
    # BIC suele seguir bajando con muchos datos: se elige el k donde la mejora marginal cae < 1%
    ks = sorted(bics)
    k_sel = ks[-1]
    for i in range(1, len(ks)):
        if (bics[ks[i - 1]] - bics[ks[i]]) / abs(bics[ks[i - 1]]) < 0.01:
            k_sel = ks[i - 1]
            break
    gm = GaussianMixture(k_sel, covariance_type="full", random_state=0, n_init=3).fit(Z[sub])
    mbf["arq"] = gm.predict(Z)
    mca["arq"] = gm.predict(pca.transform(np.log(Rca)))
    print("BIC:", {k: round(v) for k, v in bics.items()}, "-> k =", k_sel)

    # ---- descripción de cada arquetipo
    pasos = pd.read_csv(res_dir("morfoespacio") / "pasos_bf.csv.gz")
    mbf = mbf.merge(pasos[["movie", "track_id", "frame", "rapidez", "cos_giro", "alargamiento", "solidez", "q"]],
                    on=["movie", "track_id", "frame"], how="left")
    filas = []
    for a in range(k_sel):
        sel = mbf.arq.to_numpy() == a
        rmed = np.exp(L[sel].mean(0))
        F = np.fft.rfft(rmed) / N_ANG
        A = np.abs(F[1:9]) * 2 / np.abs(F[0])
        g = mbf[sel]
        nom, fam = nombre_por_armonicos(A, g.alargamiento.median(), g.solidez.median())
        _, _, par = ajustar_geometria(rmed, g.solidez.median(), fam)
        filas.append({"arq": a, "frac": sel.mean(), "n": int(sel.sum()), "nombre": nom,
                      "familia": par["familia"], "ar_plantilla": par["ar"], "n_plantilla": par["n"],
                      **{f"A{k + 1}": A[k] for k in range(8)},
                      "alargamiento_med": g.alargamiento.median(), "solidez_med": g.solidez.median(), "q_med": g.q.median(),
                      "rapidez_med": g.rapidez.median(), "cos_giro_med": g.cos_giro.mean()})
    arq = pd.DataFrame(filas).sort_values("rapidez_med").reset_index(drop=True)
    orden = {a: i for i, a in enumerate(arq.arq)}
    arq["id"] = [f"F{i + 1}" for i in range(len(arq))]
    mbf["id"] = mbf.arq.map(dict(zip(arq.arq, arq.id)))
    mca["id"] = mca.arq.map(dict(zip(arq.arq, arq.id)))
    # nombres repetidos -> numerar
    dup = arq.nombre.duplicated(keep=False)
    arq.loc[dup, "nombre"] = arq.loc[dup, "nombre"] + " (" + arq.loc[dup, "id"] + ")"
    arq.to_csv(OUT / "arquetipos.csv", index=False)
    print(arq.round(3).to_string(index=False))

    # ---- movimiento por arquetipo con películas como réplicas (Friedman)
    pm = mbf.groupby(["movie", "id"]).agg(rapidez=("rapidez", "mean"), cos_giro=("cos_giro", "mean"),
                                          n=("rapidez", "size")).reset_index()
    tab = pm.pivot(index="movie", columns="id", values="rapidez").dropna()
    fr = stats.friedmanchisquare(*[tab[c] for c in tab.columns])
    tab2 = pm.pivot(index="movie", columns="id", values="cos_giro").dropna()
    fr2 = stats.friedmanchisquare(*[tab2[c] for c in tab2.columns])
    pm.to_csv(OUT / "movimiento_por_arquetipo_y_pelicula.csv", index=False)
    pd.DataFrame([{"medida": "rapidez", "friedman_chi2": fr.statistic, "p": fr.pvalue, "peliculas": len(tab)},
                  {"medida": "cos_giro", "friedman_chi2": fr2.statistic, "p": fr2.pvalue, "peliculas": len(tab2)}]
                 ).to_csv(OUT / "friedman.csv", index=False)
    print("Friedman rapidez", fr, "cos_giro", fr2)

    # ---- transiciones entre arquetipos cada 5 min
    s = mbf.sort_values(["track_id", "frame"])
    nxt = s.groupby("track_id").id.shift(-1)
    ok = (s.groupby("track_id").frame.shift(-1) - s.frame) == 1
    T = pd.crosstab(s.id[ok], nxt[ok], normalize="index")
    T.to_csv(OUT / "transiciones.csv")
    perman = pd.Series(np.diag(T.values), index=T.index)
    permanencia_min = 5 / (1 - perman)
    print("permanencia media (min):", permanencia_min.round(1).to_dict())

    # ---- CAMAD: composición por sustrato y tiempo
    mca["condicion"] = mca.movie.map(CAMAD_CONDICION)
    mca["t_h"] = mca.frame * DATASETS["camad"]["dt_s"] / 3600
    comp = mca.groupby(["condicion", pd.cut(mca.t_h, [0, 1, 2, 3, 4, 5.1])], observed=True).id.value_counts(normalize=True)
    comp.rename("fraccion").reset_index().to_csv(OUT / "camad_composicion.csv", index=False)

    # ================= figuras
    ids = arq.id.tolist()
    cols = min(len(ids), 5)
    filas_fig = int(np.ceil(len(ids) / cols))
    fig, axs = plt.subplots(filas_fig, cols, figsize=(3.3 * cols, 3.7 * filas_fig), subplot_kw={"aspect": "equal"})
    axs = np.atleast_1d(axs).ravel()
    for ax, (_, a) in zip(axs, arq.iterrows()):
        sel = mbf.id.to_numpy() == a.id
        ejemplos = np.where(sel)[0]
        for j in rng.choice(ejemplos, min(25, len(ejemplos)), replace=False):
            r = Rbf[j]
            ax.plot(np.r_[r * np.cos(TH), r[0] * np.cos(TH[0])], np.r_[r * np.sin(TH), r[0] * np.sin(TH[0])],
                    color=S.NEUTRO, lw=0.4, alpha=0.5)
        rmed = np.exp(L[sel].mean(0))
        ax.fill(rmed * np.cos(TH), rmed * np.sin(TH), color=S.CAT[0], alpha=0.35)
        ax.plot(np.r_[rmed * np.cos(TH), rmed[0] * np.cos(TH[0])], np.r_[rmed * np.sin(TH), rmed[0] * np.sin(TH[0])],
                color=S.CAT[0], lw=2)
        _, rg, _ = ajustar_geometria(rmed, 1.0, a.familia)
        ax.plot(np.r_[rg * np.cos(TH), rg[0] * np.cos(TH[0])], np.r_[rg * np.sin(TH), rg[0] * np.sin(TH[0])],
                color=S.CAT[1], lw=1.6, ls="--")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis("off")
        import textwrap
        ax.set_title(f"{a.id}: " + "\n".join(textwrap.wrap(a.nombre, 30)) +
                     f"\n{100 * a.frac:.0f}% de las observaciones · {a.rapidez_med:.2f} µm/min", fontsize=7.5)
    for ax in axs[len(ids):]:
        ax.axis("off")
    fig.suptitle("Arquetipos de forma (brightfield), ordenados de más lentos a más rápidos. Azul: forma media; gris: 25 células del grupo; naranja punteado: geometría propuesta")
    S.guardar(fig, FIG / "arq_formas")

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.4))
    x = np.arange(len(ids))
    for mv, g in pm.groupby("movie"):
        gg = g.set_index("id").reindex(ids)
        axs[0].plot(x, gg.rapidez, color=S.NEUTRO, lw=0.7, alpha=0.8)
        axs[1].plot(x, gg.cos_giro, color=S.NEUTRO, lw=0.7, alpha=0.8)
    axs[0].plot(x, pm.groupby("id").rapidez.mean().reindex(ids), color=S.CAT[0], lw=2.4, marker="o")
    axs[1].plot(x, pm.groupby("id").cos_giro.mean().reindex(ids), color=S.CAT[0], lw=2.4, marker="o")
    for ax, t, p in ((axs[0], "Rapidez por arquetipo", fr.pvalue), (axs[1], "Persistencia (⟨cos giro⟩) por arquetipo", fr2.pvalue)):
        ax.set_xticks(x, ids)
        ax.set_title(f"{t}\n(gris: películas; Friedman p = {p:.1g})")
    axs[0].set_ylabel("µm/min")
    im = axs[2].imshow(T.reindex(index=ids, columns=ids).values, cmap=S.CMAP_SEQ, vmin=0, vmax=1)
    axs[2].set_xticks(x, ids)
    axs[2].set_yticks(x, ids)
    for i in range(len(ids)):
        for j in range(len(ids)):
            v = T.reindex(index=ids, columns=ids).values[i, j]
            axs[2].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.5 else S.TINTA)
    axs[2].set_xlabel("arquetipo 5 min después")
    axs[2].set_ylabel("arquetipo ahora")
    axs[2].set_title("Transiciones entre arquetipos (probabilidad en 5 min)")
    axs[2].grid(False)
    fig.colorbar(im, ax=axs[2], fraction=0.046)
    S.guardar(fig, FIG / "arq_movimiento")

    # forma alineada con el movimiento: rápidas vs lentas
    RMv = RMbf
    okm = np.isfinite(RMv).all(1) & mbf.rapidez.notna().to_numpy()
    q1, q3 = np.nanquantile(mbf.rapidez[okm], [0.25, 0.75])
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, (nom, sel, col) in zip(axs[:2], (("25% más lentas", okm & (mbf.rapidez.to_numpy() <= q1), S.CAT[1]),
                                             ("25% más rápidas", okm & (mbf.rapidez.to_numpy() >= q3), S.CAT[0]))):
        rm = np.exp(np.log(RMv[sel]).mean(0))
        ax.fill(rm * np.cos(TH), rm * np.sin(TH), color=col, alpha=0.35)
        ax.plot(np.r_[rm * np.cos(TH), rm[0] * np.cos(TH[0])], np.r_[rm * np.sin(TH), rm[0] * np.sin(TH[0])], color=col, lw=2)
        ax.annotate("", xy=(1.25, 0), xytext=(0.6, 0), arrowprops=dict(arrowstyle="->", color=S.TINTA, lw=1.5))
        ax.set_xlim(-1.3, 1.4)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"{nom}: forma media vista\ndesde su dirección de movimiento (→)", fontsize=9)
    for nom, sel, col in (("lentas", okm & (mbf.rapidez.to_numpy() <= q1), S.CAT[1]),
                          ("rápidas", okm & (mbf.rapidez.to_numpy() >= q3), S.CAT[0])):
        rm = np.exp(np.log(RMv[sel]).mean(0))
        axs[2].plot(np.degrees(TH), rm / rm.mean(), color=col, lw=2, label=nom)
    axs[2].axvline(0, color=S.TINTA2, lw=0.8)
    axs[2].set_xlabel("dirección respecto del movimiento (grados; 0 = frente, ±180 = cola)")
    axs[2].set_ylabel("radio relativo")
    axs[2].set_title("Perfil frente–cola")
    axs[2].legend()
    S.guardar(fig, FIG / "arq_frente_cola")

    # CAMAD composición, agrupando los 9 arquetipos en 5 familias (9 colores no se distinguen bien)
    familia = {"F1": "redondas (F1–F2)", "F2": "redondas (F1–F2)", "F3": "elípticas (F3)",
               "F4": "triangulares (F4–F5)", "F5": "triangulares (F4–F5)", "F6": "husos (F6–F8)",
               "F7": "husos (F6–F8)", "F8": "husos (F6–F8)", "F9": "ramificadas (F9)"}
    fams = ["redondas (F1–F2)", "elípticas (F3)", "triangulares (F4–F5)", "husos (F6–F8)", "ramificadas (F9)"]
    mca["familia"] = mca.id.map(familia)
    compf = mca.groupby(["condicion", pd.cut(mca.t_h, [0, 1, 2, 3, 4, 5.1])], observed=True).familia.value_counts(
        normalize=True).rename("fraccion").reset_index()
    compf.to_csv(OUT / "camad_composicion_familias.csv", index=False)
    conds = [x for x in CAMAD_ORDEN_CONDICIONES if x in set(compf.condicion)]
    fig, axs = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.9), sharey=True)
    for ax, cond in zip(axs, conds):
        g = compf[compf.condicion == cond].pivot(index="t_h", columns="familia", values="fraccion").reindex(
            columns=fams).fillna(0)
        base = np.zeros(len(g))
        xs = np.arange(len(g))
        for k, a in enumerate(fams):
            ax.bar(xs, g[a], bottom=base, color=S.CAT[k], width=0.85, edgecolor="white", linewidth=0.8,
                   label=a if cond == conds[0] else None)
            base += g[a].to_numpy()
        ax.set_xticks(xs, ["0–1", "1–2", "2–3", "3–4", "4–5"][:len(g)])
        ax.set_xlabel("tiempo (h)")
        ax.set_title(cond, fontsize=9)
    axs[0].set_ylabel("fracción de células")
    fig.legend(loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("CAMAD: proporción de cada familia de formas a lo largo de la adhesión")
    S.guardar(fig, FIG / "arq_camad")

if __name__ == "__main__":
    main()
