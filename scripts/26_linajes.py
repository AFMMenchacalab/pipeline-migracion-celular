"""
Detección de divisiones celulares y árboles genealógicos (linajes) a partir
del tracking de NÚCLEOS (SiR-DNA) del dataset brightfield.

Por qué con núcleos y no con brightfield: en brightfield una célula que se
divide se redondea y sus hijas quedan pegadas; con solo la distancia no se
distingue una hija de una vecina que pasa cerca (probado: el "splitting" de
laptrack dio 0, 7 o 74 divisiones según el umbral en la misma película). En
el canal de núcleos la división tiene una firma clara: la cromatina se
condensa (el núcleo se ve más chico y brillante) y después aparecen DOS
núcleos hijos juntos, de tamaño parecido entre sí y menor que el de la madre.

Regla (explicable, sin aprendizaje automático). Para cada trayectoria S que
empieza a mitad de la película (frame f > 0):
  1. Madre candidata M: una trayectoria presente en el frame anterior (f-1)
     cuyo núcleo está a menos de R_MADRE um del inicio de S.
  2. Hermana D2: o bien la continuación de M en el frame f (el tracker suele
     dejar que una hija "herede" la trayectoria de la madre), o bien otra
     trayectoria que empieza en f o f+1 cerca de M.
  3. Tamaños: cada hija tiene entre AREA_MIN y AREA_MAX del área de la madre
     (mediana de los 3 frames previos) y las hermanas se parecen entre sí
     (cociente de áreas < 1.6).
  4. Puntaje extra si la madre muestra el pico de brillo de la condensación
     (intensidad media máxima en los 4 frames previos / mediana de su vida).
Cada trayectoria puede ser hija de una sola división. El resultado se valida
(a) con un mosaico de ejemplos para inspección visual y (b) comparando la
tasa de división con el tiempo de duplicación esperado de MDA-MB-231
(~30-40 h en cultivo).

Salidas: resultados/v2/linajes/{divisiones.csv, segmentos.csv, resumen.csv},
figuras linaje_*.png y el video reporte/videos/linajes.mp4
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, res_dir  # noqa: E402
from lib.fuentes import listar_frames, leer_frame, cargar_mascara  # noqa: E402

UM = DATASETS["sirdna"]["um_per_px"]
DT_MIN = DATASETS["sirdna"]["dt_s"] / 60
R_MADRE = 20.0          # um: hija a menos de esto de la madre
R_HERMANAS = 30.0       # um: hermanas a menos de esto entre sí
# Conservación del material nuclear: cada hija ~la mitad de la madre. En la
# inspección visual, las divisiones reales tenían hijas de 0.44-0.59 del área de
# la madre y las falsas al menos una "hija" de 0.8-0.9 (el mismo núcleo con un
# fragmento al lado).
AREA_MIN, AREA_MAX = 0.35, 0.70
SUMA_MAX = 1.30
# Filtros de una división REAL (agregados tras inspeccionar el mosaico de la
# primera versión, donde la mayoría de las "divisiones" eran núcleos lobulados
# que Cellpose partía en dos durante unos cuadros):
PERSISTENCIA = 6          # frames (30 min): ambas hijas siguen existiendo por separado
SEP_MIN_DESPUES = 12.0    # um: separación entre hermanas 25 min después (se alejan)
R_VECINO_PREVIO = 12.0    # um: sin otro objeto pegado a la madre en los 20 min previos
SOLIDEZ_MIN = 0.90        # la madre es compacta (un núcleo lobulado tiene solidez baja)
OUT = res_dir("linajes")


def intensidades(movie):
    items = [i for i in listar_frames("sirdna") if i["movie"] == movie]
    filas = []
    for it in items:
        m = cargar_mascara("sirdna", movie, it["frame"])
        img = leer_frame(it)
        p = regionprops_table(m, intensity_image=img, properties=("label", "intensity_mean"))
        d = pd.DataFrame(p)
        d["frame"] = it["frame"]
        filas.append(d)
    d = pd.concat(filas)
    d["movie"] = movie
    return d


def detectar(tr):
    """tr: tracks nucleares de UNA película con columna intensity_mean."""
    tr = tr.sort_values(["track_id", "frame"])
    info = tr.groupby("track_id").agg(f0=("frame", "min"), f1=("frame", "max"))
    por_frame = {f: g for f, g in tr.groupby("frame")}
    med_int = tr.groupby("track_id")["intensity_mean"].median()
    med_frame = tr.groupby("frame")["intensity_mean"].median()   # brillo típico de un núcleo en ese cuadro
    usadas = set()
    divs = []
    candidatos = info[info.f0 > 0].sort_values("f0")
    for sid, row in candidatos.iterrows():
        f = int(row.f0)
        if sid in usadas or (f - 1) not in por_frame:
            continue
        s0 = tr[(tr.track_id == sid) & (tr.frame == f)].iloc[0]
        prev = por_frame[f - 1]
        prev = prev[prev.track_id != sid]
        if prev.empty:
            continue
        dist = np.hypot(prev.x - s0.x, prev.y - s0.y) * UM
        for mi in np.argsort(dist.values)[:3]:
            if dist.values[mi] > R_MADRE:
                break
            mid = prev.track_id.values[mi]
            m_hist = tr[(tr.track_id == mid) & (tr.frame >= f - 3) & (tr.frame <= f - 1)]
            a_madre = m_hist.area.median()
            # hermana: continuación de M en f, o una trayectoria nueva en f / f+1
            cand = []
            cont = tr[(tr.track_id == mid) & (tr.frame == f)]
            if len(cont):
                cand.append(("continuacion", mid, cont.iloc[0]))
            for ff in (f, f + 1):
                if ff in por_frame:
                    nuevos = por_frame[ff]
                    nuevos = nuevos[(nuevos.track_id != sid) & (nuevos.track_id.map(info.f0) == ff)
                                    & (~nuevos.track_id.isin(usadas))]
                    for _, r2 in nuevos.iterrows():
                        cand.append(("nueva", r2.track_id, r2))
            mejor = None
            for tipo, did, d2 in cand:
                d_h = np.hypot(d2.x - s0.x, d2.y - s0.y) * UM
                d_m = np.hypot(d2.x - m_hist.x.iloc[-1], d2.y - m_hist.y.iloc[-1]) * UM
                r1, r2 = s0.area / a_madre, d2.area / a_madre
                if d_h > R_HERMANAS or d_m > R_MADRE:
                    continue
                if not (AREA_MIN <= r1 <= AREA_MAX and AREA_MIN <= r2 <= AREA_MAX) or r1 + r2 > SUMA_MAX:
                    continue
                if max(s0.area, d2.area) / min(s0.area, d2.area) > 1.6:
                    continue
                costo = d_h / R_HERMANAS + abs(np.log(s0.area / d2.area))
                if mejor is None or costo < mejor[0]:
                    mejor = (costo, tipo, did, d2, d_h, r1, r2)
            if mejor is None:
                continue
            _, tipo, did, d2, d_h, r1, r2 = mejor
            ventana = tr[(tr.track_id == mid) & (tr.frame >= f - 4) & (tr.frame <= f - 1)]
            brillo = ventana.intensity_mean.max() / med_int[mid] if len(ventana) else np.nan
            # --- filtros de división real
            post1 = tr[(tr.track_id == sid) & (tr.frame >= f)]
            post2 = tr[(tr.track_id == did) & (tr.frame >= f)]
            if len(post1) < PERSISTENCIA or len(post2) < PERSISTENCIA:
                continue
            ff = f + 5
            a1, a2 = post1[post1.frame <= ff].iloc[-1], post2[post2.frame <= ff].iloc[-1]
            sep_despues = np.hypot(a1.x - a2.x, a1.y - a2.y) * UM
            if sep_despues < SEP_MIN_DESPUES:
                continue
            vecino_previo = False
            for fp_ in range(f - 4, f):
                if fp_ not in por_frame:
                    continue
                mp = tr[(tr.track_id == mid) & (tr.frame == fp_)]
                if mp.empty:
                    continue
                otros = por_frame[fp_]
                otros = otros[otros.track_id != mid]
                if len(otros) and (np.hypot(otros.x - mp.x.iloc[0], otros.y - mp.y.iloc[0]) * UM < R_VECINO_PREVIO).any():
                    vecino_previo = True
                    break
            if vecino_previo:
                continue
            if ventana.solidity.median() < SOLIDEZ_MIN:
                continue
            # brillo relativo al núcleo típico del mismo cuadro (cromatina condensada)
            b_madre = (ventana.intensity_mean / ventana.frame.map(med_frame)).max()
            h = pd.concat([post1[post1.frame <= f + 1], post2[post2.frame <= f + 1]])
            b_hijas = (h.intensity_mean / h.frame.map(med_frame)).mean()
            divs.append({"movie": int(s0.movie), "frame": f, "madre": mid, "hija1": sid, "hija2": did,
                         "hija2_tipo": tipo, "dist_hermanas_um": d_h, "area_hija1_rel": r1,
                         "area_hija2_rel": r2, "pico_brillo_madre": brillo, "sep_25min_um": sep_despues,
                         "solidez_madre": ventana.solidity.median(), "brillo_rel_madre": b_madre,
                         "brillo_rel_hijas": b_hijas,
                         "x_um": s0.x * UM, "y_um": s0.y * UM})
            usadas.add(sid)
            if tipo == "nueva":
                usadas.add(did)
            break
    return pd.DataFrame(divs)


def construir_linajes(tr, divs):
    """Parte cada trayectoria madre en el frame de la división y arma el árbol.
    Devuelve (tr con columna 'segmento', segmentos con padre/etiqueta)."""
    tr = tr.copy()
    tr["segmento"] = tr.track_id.astype(np.int64) * 10
    padre = {}
    for _, d in divs.sort_values("frame").iterrows():
        m, f = d.madre, d.frame
        seg_m = tr.loc[(tr.track_id == m) & (tr.frame == f - 1), "segmento"]
        if seg_m.empty:
            continue
        seg_m = int(seg_m.iloc[0])
        # hija que continúa la trayectoria de la madre -> nuevo segmento desde f
        if d.hija2_tipo == "continuacion":
            nuevo = seg_m + 1
            while nuevo in set(tr.segmento):
                nuevo += 1
            sel = (tr.track_id == m) & (tr.frame >= f) & (tr.segmento == seg_m)
            tr.loc[sel, "segmento"] = nuevo
            h2 = nuevo
        else:
            h2 = int(tr.loc[tr.track_id == d.hija2, "segmento"].iloc[0])
        h1 = int(tr.loc[tr.track_id == d.hija1, "segmento"].iloc[0])
        padre[h1] = seg_m
        padre[h2] = seg_m
    segs = tr.groupby("segmento").agg(movie=("movie", "first"), f0=("frame", "min"), f1=("frame", "max"),
                                      x0=("x", "first"), y0=("y", "first")).reset_index()
    segs["padre"] = segs.segmento.map(padre).fillna(-1).astype(np.int64)
    # etiquetas jerárquicas por película: raíces 1..N (orden de aparición), hijas n.1 / n.2
    etiqueta, generacion, arbol = {}, {}, {}
    hijos = segs[segs.padre >= 0].groupby("padre").segmento.apply(lambda s: sorted(s)).to_dict()
    for mv, g in segs.groupby("movie"):
        raices = g[g.padre < 0].sort_values(["f0", "y0", "x0"])
        for k, r in enumerate(raices.segmento, start=1):
            pila = [(r, str(k), 0, r)]
            while pila:
                sgm, lab, gen, raiz = pila.pop()
                etiqueta[sgm], generacion[sgm], arbol[sgm] = lab, gen, raiz
                for j, h in enumerate(hijos.get(sgm, []), start=1):
                    pila.append((h, f"{lab}.{j}", gen + 1, raiz))
    segs["etiqueta"] = segs.segmento.map(etiqueta)
    segs["generacion"] = segs.segmento.map(generacion)
    segs["arbol"] = segs.segmento.map(arbol)
    return tr, segs


CICLO_MIN_H = 15.0   # una célula recién nacida no puede volver a dividirse antes (ciclo de MDA-MB-231 > ~20 h)


def _pelicula(args):
    mv, trm = args
    it = intensidades(mv)
    trm = trm.merge(it, on=["movie", "frame", "label"], how="left")
    divs = detectar(trm)
    vacio = pd.DataFrame(columns=["frame", "madre", "hija1", "hija2", "hija2_tipo"])
    # Restricción biológica: si la "madre" de una división nació en otra división
    # hace menos de CICLO_MIN_H, esa segunda división es imposible y se descarta.
    # Con películas de 8.3 h esto deja a lo sumo una división por linaje.
    descartadas = 0
    while True:
        trs, segs = construir_linajes(trm, divs) if len(divs) else construir_linajes(trm, vacio)
        if not len(divs):
            break
        seg_m = trs.set_index(["track_id", "frame"]).segmento
        info = segs.set_index("segmento")
        malas = []
        for i, d in divs.iterrows():
            sm = seg_m.get((d.madre, d.frame - 1))
            if sm is None or sm not in info.index or info.loc[sm, "padre"] < 0:
                continue
            edad_h = (d.frame - info.loc[sm, "f0"]) * DT_MIN / 60
            if edad_h < CICLO_MIN_H:
                malas.append(i)
        if not malas:
            break
        descartadas += len(malas)
        divs = divs.drop(index=malas).reset_index(drop=True)
    divs["descartadas_por_ciclo_en_pelicula"] = descartadas
    return divs, trs, segs


def figura_mosaico(divs, n=12, seed=1):
    from lib import estilo as S
    from lib.estilo import plt
    if divs.empty:
        return
    ej = divs.sample(min(n, len(divs)), random_state=seed).sort_values(["movie", "frame"])
    offs = [-2, -1, 0, 1, 2]
    fig, axs = plt.subplots(len(ej), len(offs), figsize=(len(offs) * 1.6, len(ej) * 1.6))
    L = int(25 / UM)
    for i, (_, d) in enumerate(ej.iterrows()):
        items = {x["frame"]: x for x in listar_frames("sirdna") if x["movie"] == d.movie}
        cx, cy = int(d.x_um / UM), int(d.y_um / UM)
        for j, o in enumerate(offs):
            ax = axs[i, j]
            ax.axis("off")
            f = int(d.frame + o)
            if f not in items:
                continue
            im = leer_frame(items[f]).astype(float)
            H, W = im.shape
            y0, x0 = max(0, cy - L), max(0, cx - L)
            crop = im[y0:min(H, cy + L), x0:min(W, cx + L)]
            ax.imshow(crop, cmap="gray", vmin=np.percentile(im, 1), vmax=np.percentile(im, 99.9))
            if i == 0:
                ax.set_title(f"{o * DT_MIN:+.0f} min" if o else "división", fontsize=8)
        axs[i, 0].text(-0.1, 0.5, f"película {d.movie}\nframe {d.frame}", transform=axs[i, 0].transAxes,
                       ha="right", va="center", fontsize=7)
    fig.suptitle("Divisiones detectadas (núcleos SiR-DNA): 10 min antes a 10 min después")
    S.guardar(fig, res_dir("figuras") / "linaje_mosaico_divisiones")


def figura_arboles(segs, divs, n=12):
    from lib import estilo as S
    from lib.estilo import plt
    cuenta = segs[segs.padre >= 0].groupby("arbol").size().sort_values(ascending=False)
    top = cuenta.index[:n]
    if len(top) == 0:
        return
    cols = 4
    filas = int(np.ceil(len(top) / cols))
    fig, axs = plt.subplots(filas, cols, figsize=(cols * 3.6, filas * 3.0), squeeze=False)
    for ax, raiz in zip(axs.ravel(), top):
        t = segs[segs.arbol == raiz].set_index("segmento")
        # posición horizontal: hojas en orden, nodos internos en el promedio de sus hijos
        hijos = t[t.padre >= 0].groupby("padre").apply(lambda g: sorted(g.index)).to_dict()
        xpos = {}
        cont = [0]

        def ubicar(s):
            if s not in hijos:
                xpos[s] = cont[0]
                cont[0] += 1
            else:
                for h in hijos[s]:
                    ubicar(h)
                xpos[s] = np.mean([xpos[h] for h in hijos[s]])
        ubicar(raiz)
        for s, r in t.iterrows():
            x = xpos[s]
            ax.plot([x, x], [r.f0 * DT_MIN / 60, r.f1 * DT_MIN / 60], color=S.CAT[int(r.generacion) % 3], lw=2.2)
            ax.text(x + 0.06, r.f1 * DT_MIN / 60, r.etiqueta, fontsize=6.5, color=S.TINTA2, va="top")
            if s in hijos:
                y = t.loc[hijos[s][0], "f0"] * DT_MIN / 60
                ax.plot([xpos[h] for h in hijos[s]], [y, y], color=S.TINTA2, lw=1)
                ax.plot([x, x], [r.f1 * DT_MIN / 60, y], color=S.TINTA2, lw=1)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_ylabel("tiempo (h)")
        mv = int(t.movie.iloc[0])
        ax.set_title(f"película {mv}, célula {t.loc[raiz, 'etiqueta']}", fontsize=9)
        ax.set_xlim(-0.5, max(xpos.values()) + 0.9)
        ax.grid(False)
    for ax in axs.ravel()[len(top):]:
        ax.axis("off")
    fig.suptitle("Ejemplos de árboles genealógicos (en 8.3 h una célula se divide a lo sumo una vez)")
    S.guardar(fig, res_dir("figuras") / "linaje_arboles")


def video_linajes(trs, segs, movie):
    import cv2
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib.util
    spec = importlib.util.spec_from_file_location("anim", Path(__file__).resolve().parent / "22_animaciones.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    t = trs[trs.movie == movie].merge(segs[["segmento", "etiqueta", "arbol"]], on="segmento")
    arboles_div = set(segs[(segs.movie == movie) & (segs.padre >= 0)].arbol)
    items_n = [i for i in listar_frames("sirdna") if i["movie"] == movie]
    items_b = [i for i in listar_frames("bf") if i["movie"] == movie]
    lo_n, hi_n = np.percentile(leer_frame(items_n[0]), (1, 99.9))
    lo_b, hi_b = np.percentile(leer_frame(items_b[0]), (1, 99.7))
    colores = [(214, 120, 42), (52, 104, 235), (122, 175, 27), (0, 161, 237), (164, 123, 232), (72, 73, 227)]
    frames = []
    ancho = 640
    for inu, ib in zip(items_n, items_b):
        f = inu["frame"]
        n = cv2.applyColorMap(A.a_gris8(leer_frame(inu), lo_n, hi_n), cv2.COLORMAP_BONE)
        b = cv2.cvtColor(A.a_gris8(leer_frame(ib), lo_b, hi_b), cv2.COLOR_GRAY2BGR)
        esc = ancho / n.shape[1]
        n = cv2.resize(n, (ancho, int(n.shape[0] * esc)), interpolation=cv2.INTER_AREA)
        b = cv2.resize(b, (ancho, int(b.shape[0] * esc)), interpolation=cv2.INTER_AREA)
        g = t[t.frame == f]
        for _, r in g.iterrows():
            p = (int(r.x * esc), int(r.y * esc))
            en_linaje = r.arbol in arboles_div
            col = colores[int(r.arbol) % len(colores)] if en_linaje else (170, 170, 170)
            for panel in (n, b):
                cv2.circle(panel, p, 3 if en_linaje else 2, col, -1, cv2.LINE_AA)
                if en_linaje:
                    cv2.putText(panel, str(r.etiqueta), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1,
                                cv2.LINE_AA)
        cab = np.hstack([A.titulo("Núcleos (SiR-DNA) con etiqueta de linaje", ancho, 30, 14),
                         A.titulo("Brightfield, mismas etiquetas", ancho, 30, 14)])
        pie = A.titulo(f"Película {movie} · t = {f * DT_MIN / 60:.1f} h · en color: células con al menos una división "
                       f"en su linaje (la hija de 12 es 12.1 / 12.2)", 2 * ancho, 26, 12)
        frames.append(np.vstack([cab, np.hstack([n, b]), pie]))
    A.escribir_mp4(frames, res_dir("reporte", "videos") / "linajes.mp4", 8)


def main():
    tr = pd.read_csv(res_dir("sirdna", "tracking", "dist") / "tracks.csv.gz",
                     usecols=["movie", "track_id", "frame", "x", "y", "area", "label", "solidity"])
    with ProcessPoolExecutor(16) as ex:
        res = list(ex.map(_pelicula, [(mv, g) for mv, g in tr.groupby("movie")]))
    divs = pd.concat([r[0] for r in res if len(r[0])], ignore_index=True)
    trs = pd.concat([r[1] for r in res], ignore_index=True)
    segs = pd.concat([r[2] for r in res], ignore_index=True)
    divs.to_csv(OUT / "divisiones.csv", index=False)
    segs.to_csv(OUT / "segmentos.csv", index=False)
    trs[["movie", "frame", "label", "track_id", "segmento", "x", "y"]].to_csv(OUT / "tracks_con_segmento.csv.gz", index=False)
    horas = tr.groupby("movie").size() * DT_MIN / 60
    res_m = pd.DataFrame({"divisiones": divs.groupby("movie").size(), "horas_celula": horas}).fillna(0)
    res_m["tasa_por_celula_h"] = res_m.divisiones / res_m.horas_celula
    res_m["t_duplicacion_h"] = np.log(2) / res_m.tasa_por_celula_h
    res_m.to_csv(OUT / "resumen.csv")
    tot = res_m.divisiones.sum() / res_m.horas_celula.sum()
    arboles = segs[segs.padre >= 0].groupby("arbol").size()
    gens = segs.groupby("arbol").generacion.max()
    print(res_m.round(3).to_string())
    print(f"TOTAL: {int(res_m.divisiones.sum())} divisiones en {res_m.horas_celula.sum():.0f} horas-célula -> "
          f"tiempo de duplicación {np.log(2) / tot:.1f} h; árboles con división: {len(arboles)}; "
          f"generaciones máx: {int(gens.max())}; con 2 generaciones: {(gens >= 2).sum()}")
    print("hija2:", divs.hija2_tipo.value_counts().to_dict(), " pico de brillo madre (mediana):",
          round(divs.pico_brillo_madre.median(), 2))
    # control interno: en 8.3 h una hija no puede volver a dividirse (ciclo > ~18 h)
    gen_m = segs.set_index("segmento").generacion
    seg_m = trs.set_index(["movie", "track_id", "frame"]).segmento
    divs["gen_madre"] = [gen_m.get(seg_m.get((r.movie, r.madre, r.frame - 1)), np.nan) for r in divs.itertuples()]
    n_imp = int(divs.groupby("movie").descartadas_por_ciclo_en_pelicula.first().sum())
    print(f"control: divisiones descartadas por ocurrir < {CICLO_MIN_H:.0f} h después de nacer la madre "
          f"(imposibles): {n_imp} ({100 * n_imp / max(len(divs) + n_imp, 1):.1f}% de las detectadas)")
    divs.to_csv(OUT / "divisiones.csv", index=False)
    pd.DataFrame({"divisiones": [len(divs)], "segunda_generacion_imposibles": [n_imp],
                  "t_duplicacion_h": [np.log(2) / tot]}).to_csv(OUT / "control.csv", index=False)
    figura_mosaico(divs)
    figura_arboles(segs, divs)
    mv = int(res_m.divisiones.idxmax())
    video_linajes(trs, segs, mv)
    print("video: película", mv)


if __name__ == "__main__":
    main()
