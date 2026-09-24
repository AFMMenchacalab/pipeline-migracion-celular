"""
PASO 6 (v2): reporte HTML autocontenido (carpeta resultados/v2/reporte_html/)
con figuras, fotos, videos y un simulador interactivo de la EAD.

Lee todos los números de los CSV/JSON de resultados/v2 (nada escrito a mano
salvo la plantilla y los textos de este archivo), de modo que si se vuelve a
correr el pipeline el reporte se regenera con los números nuevos.

Uso: ../venv/bin/python 23_reporte_html.py
Salida: resultados/v2/reporte_html/index.html (+ figuras/, videos/, fotos/)
"""
import datetime
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import CAMAD_CONDICION, CAMAD_ORDEN_CONDICIONES, DATASETS, RES, res_dir  # noqa: E402

OUT = res_dir("reporte_html")
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
         "noviembre", "diciembre"]


# ------------------------------------------------------------------ utilidades
def f(x, d=2):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(x):
        return "–"
    s = f"{x:.{d}f}"
    return s.replace("-", "−")


def fp(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "–"
    if not np.isfinite(p):
        return "–"
    return "&lt; 0.001" if p < 0.001 else f"{p:.3f}"


def peq(p):
    """'= 0.012' o '&lt; 0.001' (para escribir «p Holm = 0.012» / «p Holm < 0.001»)."""
    v = fp(p)
    return v if v.startswith("&lt;") else f"= {v}"


def js(*partes):
    p = RES.joinpath(*partes)
    return json.loads(p.read_text()) if p.exists() else {}


def csv(*partes):
    p = RES.joinpath(*partes)
    return pd.read_csv(p) if p.exists() else None


def var(ds):
    p = RES / f"variante_{ds}.txt"
    return p.read_text().strip() if p.exists() else None


def fila_resumen(ds, metrica):
    t = csv("inferencia", ds, "resumen_bootstrap.csv")
    if t is None or not (t.metrica == metrica).any():
        return None
    return t[t.metrica == metrica].iloc[0]


def mic(ds, metrica, d=2, unidad=""):
    r = fila_resumen(ds, metrica)
    if r is None:
        return "–"
    u = f" {unidad}" if unidad else ""
    return f"{f(r.media, d)}{u} (IC95% {f(r.ic95_inf, d)}–{f(r.ic95_sup, d)})"


def tabla(df, cols, heads, num=()):
    h = ["<div class=\"tabla\"><table><thead><tr>"]
    for c, t in zip(cols, heads):
        h.append(f"<th class=\"{'n' if c in num else ''}\">{t}</th>")
    h.append("</tr></thead><tbody>")
    for _, r in df.iterrows():
        h.append("<tr>" + "".join(f"<td class=\"{'n' if c in num else ''}\">{r[c]}</td>" for c in cols) + "</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def figura(ruta, caption, alt=""):
    return (f'<figure><div class="placa"><img src="{ruta}" alt="{html.escape(alt or caption[:90])}" loading="lazy">'
            f'</div><figcaption>{caption}</figcaption></figure>\n')


def video(ruta, caption):
    poster = ruta.rsplit(".", 1)[0] + ".jpg"
    return (f'<figure><div class="placa"><video src="{ruta}" poster="{poster}" controls muted loop playsinline '
            f'preload="metadata"></video></div><figcaption>{caption}</figcaption></figure>\n')


def explica(titulo, texto, clase="explica"):
    return f'<div class="{clase} texto"><strong>{titulo}</strong>{texto}</div>\n'


# ------------------------------------------------------------------ recursos
def copiar_recursos():
    (OUT / "figuras").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    (OUT / "fotos").mkdir(exist_ok=True)
    for p in (RES / "figuras").glob("*.png"):
        im = Image.open(p)
        if im.width > 2000:
            im = im.resize((2000, int(im.height * 2000 / im.width)), Image.LANCZOS)
        im.save(OUT / "figuras" / p.name, optimize=True)
    for p in (RES / "validacion_seg_bf").glob("ejemplo_*.png"):
        im = Image.open(p).convert("RGB")
        im.thumbnail((1800, 1800))
        im.save(OUT / "figuras" / (p.stem + ".jpg"), quality=85)
    for p in (RES / "reporte" / "videos").glob("*"):
        shutil.copy(p, OUT / "videos" / p.name)


def fotos_datos():
    import tifffile
    from lib.fuentes import listar_frames, leer_frame

    def guardar(arr, nombre, lo=1, hi=99.7):
        a = arr.astype(float)
        l, h = np.percentile(a, (lo, hi))
        a = np.clip((a - l) / max(h - l, 1e-6) * 255, 0, 255).astype(np.uint8)
        im = Image.fromarray(a)
        im.thumbnail((900, 900))
        im.save(OUT / "fotos" / nombre, quality=85)

    it = [i for i in listar_frames("bf") if i["movie"] == 3][0]
    guardar(leer_frame(it)[200:800, 150:950], "bf.jpg")
    it = [i for i in listar_frames("sirdna") if i["movie"] == 3][0]
    guardar(leer_frame(it)[200:800, 150:950], "sirdna.jpg", 1, 99.9)
    from lib.config import CACHE, DATASETS_DIR
    s = np.load(CACHE / "camad_frames" / "exp07.npy", mmap_mode="r")
    guardar(np.asarray(s[300]), "camad.jpg")
    w = sorted((DATASETS_DIR / "camad-mdamb231" / "whad" / "images" / "MCF7").rglob("*_t10_*.tif"))
    if w:
        guardar(tifffile.imread(w[0]), "whad.jpg")


def tarjetas_datos():
    det_bf = csv("bf", "detecciones.csv.gz")
    n_tr_bf = csv("bf", "tracking", var("bf") or "dist_tam", "resumen.csv")
    n_tr_c = csv("camad", "tracking", var("camad") or "dist_tam", "resumen.csv")
    tr_bf = int(n_tr_bf.tracks.sum()) if n_tr_bf is not None else None
    tr_c = int(n_tr_c.tracks.sum()) if n_tr_c is not None else None
    wh = csv("whad", "por_posicion.csv")
    cards = [
        ("bf.jpg", "Brightfield MDA-MB-231",
         "Células sin ninguna tinción, filmadas con objetivo 20×. Es el dataset principal: 16 videos independientes de 100 imágenes (8.3 h).",
         f"16 videos · 1 imagen / 5 min · 0.65 µm/píxel · {tr_bf:,} trayectorias".replace(",", ".") if tr_bf else "16 videos · 5 min · 0.65 µm/píxel"),
        ("sirdna.jpg", "Núcleos fluorescentes (SiR-DNA)",
         "Los mismos campos y en el mismo instante que el dataset anterior, pero mostrando solo los núcleos teñidos. <b>Nuevo:</b> nos permite comprobar la segmentación y el seguimiento.",
         "1600 imágenes emparejadas 1 a 1 con brightfield"),
        ("camad.jpg", "CAMAD: células sobre distintos sustratos",
         "MDA-MB-231 filmadas en contraste de fase (40×) durante las 5 horas posteriores a sembrarlas sobre vidrio, Matrigel, colágeno o matrices producidas por otras células. <b>Nuevo.</b>",
         f"16 experimentos · 1 imagen / 30 s · 600 imágenes cada uno" + (f" · {tr_c:,} trayectorias".replace(",", ".") if tr_c else "")),
        ("whad.jpg", "WHAD: cierre de herida",
         "Capas de células MCF10A y MCF7 a las que se les raya una franja vacía; se mide cuánto tardan en cerrarla. La herida viene delineada a mano. <b>Nuevo.</b>",
         f"{len(wh) if wh is not None else '–'} posiciones · 1 imagen / hora"),
    ]
    out = []
    for foto, tit, txt, meta in cards:
        out.append(f'<article class="dato"><img src="fotos/{foto}" alt="{html.escape(tit)}" loading="lazy"><div>'
                   f'<h4>{tit}</h4>{txt}<p class="meta">{meta}</p></div></article>')
    return "\n".join(out)


# ------------------------------------------------------------------ secciones
def seccion_cellpose():
    """Diferencias entre Cellpose 3 y Cellpose-SAM (arquitectura y por qué se usa SAM)."""
    c = js("cellpose3_vs_sam", "comparacion.json")
    s = ['<div class="texto"><h3>¿Por qué Cellpose-SAM y no Cellpose 3?</h3>',
         "<p>Las dos versiones de Cellpose predicen lo mismo para cada píxel: la probabilidad de que pertenezca a una célula y un «flujo» que apunta hacia el centro de su célula. Después, un mismo post-proceso sigue esos flujos y agrupa los píxeles que llegan al mismo centro, y así separa células que se tocan. Lo que cambia es la red neuronal que hace esa predicción.</p></div>"]
    filas = [
        ("Arquitectura", "U-Net residual: red convolucional que mira vecindarios locales de la imagen y los combina a varias escalas",
         "Codificador de <i>Segment Anything</i> (SAM, Meta): un <i>vision transformer</i> (ViT-L) que divide la imagen en parches y, mediante atención, relaciona cada parche con todos los demás del bloque"),
        ("Tamaño", "6.6 millones de parámetros", "304.6 millones de parámetros (24 bloques de transformer de dimensión 1024)"),
        ("Preentrenamiento", "Solo imágenes de microscopía de células (varios conjuntos públicos combinados)",
         "SAM se preentrenó con 11 millones de imágenes y más de mil millones de contornos de objetos. Después se reentrenó entero con imágenes de células"),
        ("Resolución de trabajo", "Reescala la imagen para que las células midan ~30 píxeles de diámetro",
         "Parches de 8 × 8 píxeles (SAM original usa 16 × 16) para ver estructuras pequeñas; bloques de 256 × 256"),
        ("Tamaño de las células", "Hay que indicar el diámetro o estimarlo con un segundo modelo; si se estima mal, la segmentación empeora",
         "No necesita diámetro: se entrenó con células de tamaños muy distintos"),
        ("Canales e imagen", "Hay que indicar qué canal es citoplasma y cuál núcleo",
         "No necesita indicarlo; se entrenó con canales en cualquier orden y con imágenes ruidosas, borrosas o de baja resolución"),
        ("Costo de cálculo", f"{f(c.get('cellpose3', {}).get('gflop_por_bloque_224', 31), 0)} GFLOP por bloque de 224 × 224; funciona razonablemente en CPU",
         "727 GFLOP por bloque de 256 × 256 (unas 20 veces más); en la práctica necesita GPU"),
    ]
    s.append(tabla(pd.DataFrame(filas, columns=["a", "b", "c"]), ["a", "b", "c"], ["", "Cellpose 3 (cyto3)", "Cellpose-SAM (cpsam, Cellpose 4)"]))
    s.append('<div class="texto">')
    s.append("<p><b>Por qué es mejor Cellpose-SAM para este trabajo.</b> Primero, la atención le permite usar el contexto de todo el bloque y no solo el vecindario inmediato. Eso ayuda a decidir dónde termina una célula y empieza otra cuando se tocan o cuando el borde es tenue, como en brightfield. Segundo, el preentrenamiento de SAM le da una noción general de qué es un «objeto». Por eso, según sus autores, generaliza a tipos de imagen que no vio al entrenarse con una precisión comparable a la de anotadores humanos (Pachitariu, Rariden y Stringer, 2025). Tercero, al no depender del diámetro ni del orden de los canales, el mismo modelo sirve sin ajustes para brightfield, fluorescencia de núcleos y contraste de fase, que son las tres modalidades de este trabajo.</p>")
    if c:
        a, b = c["cellpose3"], c["cellpose_sam"]
        s.append(f"<p>En nuestras imágenes brightfield, evaluadas contra los núcleos (una imagen por video, {c['n_imagenes']} en total), Cellpose-SAM obtuvo F1 = {f(b['f1'], 2)} contra {f(a['f1'], 2)} de Cellpose 3 (p {peq(c['p_wilcoxon_pareado_f1'])}, prueba pareada). También produjo menos células fusionadas ({b['fusiones']} contra {a['fusiones']}) y menos detecciones sin núcleo ({b['celulas_sin_nucleo']} contra {a['celulas_sin_nucleo']}). La contrapartida es el costo: Cellpose 3 es unas 5 veces más rápido en CPU. Esto es relevante para equipos sin GPU, como una Raspberry Pi (sección 15).</p>")
    s.append("</div>")
    return "\n".join(s)


def camad_aplicada():
    """Configuración aplicada a CAMAD (eleccion_aplicada.json); si todavía no
    existe, la mejor de la escala de extracción según la grilla."""
    ea = js("validacion_seg_camad", "eleccion_aplicada.json")
    if ea:
        return ea
    rk = csv("validacion_seg_camad", "ranking_parametros.csv")
    if rk is None:
        return {}
    from lib.config import CAMAD_DOWNSAMPLE
    rk["pf"] = rk.puntaje + 0.5 * rk.precision_local.fillna(0)
    r = rk[rk.escala == CAMAD_DOWNSAMPLE].sort_values("pf", ascending=False).iloc[0]
    return {"recall": r.recall, "iou_medio": r.iou_medio, "cellprob_threshold": r.cellprob, "flow_threshold": r.flow}


def seccion_validacion():
    s = []
    el = js("validacion_seg_bf", "eleccion.json")
    s.append('<h3>Segmentación: cada célula debe contener un núcleo</h3><div class="texto">')
    if el:
        v1 = el.get("metricas_v1_default(cp0,flow0.4)", {})
        lo = el.get("metricas_loeo_media", {})
        el_m = el.get("metricas_elegida_por_pelicula_media", {})
        mx = el.get("maximo_f1", {})
        s.append(f"<p>Como tenemos la imagen de núcleos del mismo instante, una célula bien detectada en brightfield debería contener exactamente un núcleo. Contamos células con un núcleo (aciertos), sin núcleo (detecciones falsas), con dos o más núcleos (células fusionadas) y núcleos sin célula (células perdidas). Con los umbrales elegidos la segmentación alcanza un <b>F1 de {f(el_m.get('f1'), 3)}</b>, con precisión {f(el_m.get('precision'), 3)} y recall {f(el_m.get('recall'), 3)} (promedio de los 16 videos, con desvío de ±{f(el.get('metricas_loeo_sd', {}).get('f1'), 3)} entre videos).</p>")
        if mx and (mx.get("cellprob"), mx.get("flow")) != (el["cellprob_threshold"], el["flow_threshold"]):
            s.append(f"<p>La combinación con el F1 más alto (<span class=\"mono\">cellprob = {mx['cellprob']:g}, flow = {mx['flow']:g}</span>, F1 = {f(mx['f1'], 4)}) quedó empatada con los umbrales por defecto (F1 = {f(el_m.get('f1'), 4)}): la diferencia es menor que una décima de la variación entre videos. Entre opciones empatadas elegimos la de <b>mayor precisión</b> ({f(el_m.get('precision'), 2)} contra {f(mx['precision'], 2)}), porque produce la mitad de detecciones falsas. Para medir movimiento, un objeto falso es peor que una célula no detectada: genera una trayectoria inventada, mientras que la célula perdida solo reduce un poco la muestra. Resultado: se mantienen los umbrales por defecto de Cellpose (<span class=\"mono\">cellprob = {el['cellprob_threshold']:g}, flow = {el['flow_threshold']:g}</span>), ahora con evidencia de que son los adecuados.</p>")
    s.append("</div>")
    s.append(figura("figuras/val_segmentacion_bf.png",
                    "<b>Segmentación brightfield comparada con los núcleos.</b> Izquierda: F1 promedio de los 16 videos para cada combinación de umbrales. Centro: cada punto es un video, con los umbrales anteriores (naranja) y los elegidos (azul). Derecha: cantidad de cada tipo de error."))
    s.append(figura("figuras/ejemplo_pelicula15_frame050.jpg",
                     "<b>Ejemplo del campo más denso.</b> Izquierda: brightfield con el contorno de cada célula detectada y el centro de cada núcleo (puntos celestes). Derecha: la imagen de núcleos. Aun con células que se tocan, casi todos los contornos contienen un único núcleo."))
    # CAMAD
    ea = camad_aplicada()
    g = csv("validacion_seg_camad", "grilla.csv")
    s.append('<h3>Segmentación de CAMAD comparada con dibujos hechos a mano</h3><div class="texto">')
    if g is not None:
        full = g[(g.escala == 1)].groupby("exp").recall.mean().mean()
        s.append(f"<p>CAMAD trae 96 imágenes en las que un experto dibujó el contorno de las células aisladas. Con las imágenes a resolución completa (como hacía la versión anterior), Cellpose encontraba el <b>{f(100 * full, 1)}%</b> de esas células. La causa es la escala: a 0.117 µm por píxel una célula mide 150–400 píxeles, mucho más de lo que el modelo espera. Reduciendo cada imagen 4 veces por lado, el recall sube al <b>{f(100 * ea.get('recall', np.nan), 1)}%</b>, con una superposición media (IoU) de {f(ea.get('iou_medio'), 2)} entre el contorno detectado y el dibujado. Además cada video se procesa unas 10 veces más rápido. Si la configuración se elige dejando afuera cada experimento y se evalúa en ese experimento (una estimación más exigente), el recall es {f(100 * js('validacion_seg_camad', 'eleccion.json').get('metricas_loeo_media', {}).get('recall', np.nan), 0)}%. Los experimentos más difíciles son el 11 (Matrigel) y el 12 (colágeno), con muchas células fuera de foco.</p>")
    s.append("</div>")
    s.append(figura("figuras/val_segmentacion_camad.png",
                    "<b>CAMAD: calidad de la segmentación.</b> Izquierda: recall y superposición según cuánto se reduce la imagen. Centro: recall por experimento con la configuración elegida, agrupado por sustrato. Derecha: cada punto es un experimento, versión anterior contra versión nueva."))
    s.append(explica("¿Por qué no se mide la precisión en CAMAD?",
                     "<p>Los autores solo dibujaron las células aisladas, no todas las del campo. Una detección fuera de esos dibujos puede ser una célula real no dibujada, así que no se la puede contar como error. Por eso evaluamos la segmentación desde el lado de las células dibujadas: cuántas se encontraron, qué tan bien coincide el contorno y si alguna quedó partida o fusionada con otra.</p>"))
    # tracking
    et = js("validacion_tracking", "eleccion.json")
    r = csv("validacion_tracking", "por_pelicula.csv")
    s.append('<h3>Seguimiento: ¿la trayectoria sigue siempre a la misma célula?</h3><div class="texto">')
    if r is not None and et:
        m = r.groupby("variante")[["precision_eslabon", "recall_eslabon", "pureza", "completitud"]].mean()
        e = et["elegida"]
        s.append(f"<p>Seguimos por separado los núcleos (que son pequeños, brillantes y rara vez se tocan, así que su seguimiento es muy confiable) y comparamos, paso por paso, si el seguimiento de las células en brightfield une las mismas células que el de los núcleos. Con el método elegido (<span class=\"mono\">{e}</span>), el <b>{f(100 * m.loc[e, 'precision_eslabon'], 1)}%</b> de las uniones entre imágenes consecutivas son correctas y se reproduce el {f(100 * m.loc[e, 'recall_eslabon'], 1)}% de las uniones de los núcleos. El método anterior (<span class=\"mono\">v1</span>) daba {f(100 * m.loc['v1', 'precision_eslabon'], 1)}% y {f(100 * m.loc['v1', 'recall_eslabon'], 1)}%. Sobre trayectorias completas, en promedio el {f(100 * m.loc[e, 'pureza'], 0)}% de cada trayectoria sigue a un único núcleo; los errores se concentran en células que se cruzan o se dividen.</p>")
        s.append(f"<p>También probamos detectar divisiones celulares (mitosis) con el seguimiento, pero el resultado dependía demasiado del umbral (entre 0 y 74 divisiones en el mismo video, cuando lo esperable es alrededor de 50). Sin una marca específica de mitosis no es confiable, así que quedó como trabajo futuro.</p>")
    s.append("</div>")
    s.append(figura("figuras/val_tracking.png",
                    "<b>Seguimiento en brightfield comparado con el seguimiento de núcleos.</b> Cada punto es un video. Precisión: fracción de uniones correctas. Recall: fracción de uniones de los núcleos que el método reproduce. Pureza: qué parte de cada trayectoria sigue a una única célula. Completitud: qué parte de la vida de cada célula queda en una sola trayectoria."))
    return "\n".join(s)


def seccion_simulaciones():
    a = csv("simulaciones", "a_reproduccion_liu2021.csv")
    d = csv("simulaciones", "d_regimen_real.csv")
    s = ['<h3>Comprobación con células simuladas</h3><div class="texto">']
    if a is not None:
        s.append("<p>Para verificar que los cálculos son correctos, simulamos células con persistencia conocida usando exactamente los parámetros de Liu et al. (2021). Nuestra SE da " +
                 ", ".join(f"{f(x, 2)}" for x in a.se_media) + " para persistencias de 0.5, 2, 8 y 32 minutos; el artículo reporta aproximadamente 0.90, 0.75, 0.60 y 0.45.</p>")
    if d is not None:
        r0 = d[(d.sigma_um == 0)].set_index("P_min")
        r2 = d[(d.sigma_um == 2)].set_index("P_min")
        s.append(f"<p>Después simulamos células como las nuestras (una posición cada 5 minutos, 60 pasos). Sin error de posición, la EAD₁ corregida distingue bien la persistencia: {f(r0.loc[5, 'ead1_corr'])} para P = 5 min, {f(r0.loc[20, 'ead1_corr'])} para 20 min y {f(r0.loc[60, 'ead1_corr'])} para 60 min. Pero con un error de posición de 2 µm, similar al que medimos en brightfield, los mismos casos dan {f(r2.loc[5, 'ead1_corr'])}, {f(r2.loc[20, 'ead1_corr'])} y {f(r2.loc[60, 'ead1_corr'])}: el temblor de la posición tapa casi toda la persistencia. Esto marca un límite de lo que se puede medir con estos datos y explica por qué conviene también analizar pasos más largos o usar los núcleos, cuya posición tiembla menos.</p>")
    s.append("</div>")
    s.append(figura("figuras/sim_validacion_metodos.png",
                    "<b>Validación con simulaciones.</b> (a) SE calculada contra la publicada. (b) EAD según el intervalo de tiempo para cuatro persistencias. (c) La EAD cruda (líneas punteadas) depende de la cantidad de pasos; la corregida (llenas) no. (d) En las condiciones de nuestros datos, EAD₁ (llena) y SE (punteada) corregidas según la persistencia real, sin error de posición (azul) y con 1 µm (naranja) o 2 µm (verde)."))
    return "\n".join(s)


def seccion_bf(vbf):
    s = ['<div class="texto">']
    peli = csv("bf", "estadisticas", vbf, "por_pelicula.csv")
    pr = csv("inferencia", "bf", "pruebas_nulo.csv")
    rel = csv("inferencia", "bf", "relacion_densidad.csv")
    het = csv("inferencia", "bf", "heterogeneidad_peliculas.csv")
    ncel = int(peli.n_segmentos.sum()) if peli is not None else 0
    s.append(f"<p>Analizamos {ncel:,} trayectorias de al menos una hora en los 16 videos".replace(",", ".") +
             ". Todas las cifras son el promedio entre videos, con su intervalo de confianza del 95%.</p>")
    s.append("<h3>Rapidez y persistencia clásicas</h3>")
    s.append(f"<p>Las células avanzan a <b>{mic('bf', 'rapidez_um_min', 2, 'µm/min')}</b>. Según el modelo de caminata persistente, mantienen su dirección durante unos <b>{mic('bf', 'prw_P_min', 1, 'min')}</b>, con un error de posición de {mic('bf', 'prw_sigma_um', 2, 'µm')}. Como las imágenes están separadas 5 minutos, la dirección se conserva apenas durante 2 o 3 imágenes. A la escala de horas el movimiento es casi el de una caminata al azar: el exponente del MSD a tiempos largos es {mic('bf', 'alfa_largo', 2)} (1 sería completamente al azar).</p>")
    s.append("</div>")
    s.append(tabla_resumen("bf", ["rapidez_um_min", "prw_P_min", "prw_S_um_min", "prw_sigma_um", "direccionalidad_1h",
                                  "alfa_corto", "alfa_largo", "cos_giro_medio", "ead1_ens", "ead1_ens_corr", "ead1_cel",
                                  "ead1_cel_corr", "tl1_ens_min", "tl1_cel_mediana_min", "se_fps_completa", "se_ventana_corr",
                                  "se_wavelet_corr", "ead_t_pct_06_1", "corr_dir_0_50um", "area_um2", "aspect_ratio",
                                  "densidad_cel_mm2"]))
    s.append(seccion_alfa("bf", vbf))
    s.append(figura("figuras/bf_msd_vacf_fps.png",
                    "<b>Cómo se alejan las células de su punto de partida.</b> Izquierda: MSD de cada video (gris) y promedio (azul), con las rectas de movimiento al azar (α = 1) y en línea recta (α = 2), y el ajuste del modelo persistente (naranja). Centro: cuánto se parece la velocidad a la de un tiempo atrás; cae a casi cero en 10–20 minutos. Derecha: espectro de la velocidad."))
    s.append('<h3>Entropías</h3><div class="texto">')
    s.append(f"<p>La <b>EAD₁ corregida</b> del conjunto de células de cada video es {mic('bf', 'ead1_ens_corr', 3)} y la de cada célula, promediada, {mic('bf', 'ead1_cel_corr', 3)}. Ambas son menores que 1, lo que confirma una persistencia real, aunque débil. La dirección se pierde por completo (TL₁) en {mic('bf', 'tl1_ens_min', 0, 'min')}. La <b>SE corregida</b> es {mic('bf', 'se_ventana_corr', 3)}. Para comparar con los artículos, los valores sin corregir son EAD₁ = {mic('bf', 'ead1_cel', 3)} por célula y SE = {mic('bf', 'se_fps_completa', 3)}. Parecen indicar más persistencia, pero la diferencia se explica sobre todo por el sesgo de las trayectorias cortas.</p>")
    s.append("</div>")
    s.append(figura("figuras/bf_ead.png",
                    "<b>Entropía de los ángulos de giro.</b> Izquierda: con qué frecuencia aparece cada ángulo entre la dirección actual y la de τ minutos después; el pico en 0° (misma dirección) se desvanece en menos de una hora. Centro: EAD corregida según τ; se acerca a 1 (sin dirección preferida) en 15–30 minutos. Derecha: cada célula ubicada según su EAD₁ y su TL₁ (la «distribución conjunta» de Liu et al. 2024)."))
    s.append(figura("figuras/bf_mapas_temporales.png",
                    "<b>Persistencia en el tiempo, célula por célula.</b> Cada fila es una célula y el color indica su SE (arriba) o EAD (abajo) en cada momento: más oscuro significa más aleatorio. A la derecha, el promedio de todas las células de cada video en cada instante."))
    s.append(figura("figuras/bf_pasos_rapidez_giro.png",
                    "<b>Las células rápidas giran menos.</b> Izquierda: distribución de la velocidad al cuadrado (escala logarítmica). Centro: rapidez de cada paso contra el giro que hace. Derecha: el coseno medio del giro (1 = sigue derecho, 0 = gira al azar) aumenta con la rapidez."))
    s.append(figura("figuras/bf_sensibilidad_paso.png",
                    "<b>¿Cambia la persistencia si miramos cada 10, 15 o 20 minutos en vez de cada 5?</b> Cada línea gris es un video."))
    s.append('<h3>Pruebas estadísticas</h3><div class="texto"><p>Cada prueba usa los 16 videos como réplicas. La columna «p Holm» ya está corregida por hacer varias pruebas a la vez; un valor menor que 0.05 se considera significativo.</p></div>')
    if pr is not None:
        t = pr.copy()
        t["med"] = t.mediana.map(lambda x: f(x, 3))
        t["p_"] = t.p.map(fp)
        t["ph"] = [f'<span class="{"sig" if p < 0.05 else "nosig"}">{fp(p)}</span>' for p in t.p_holm]
        s.append(tabla(t, ["prueba", "hipotesis", "med", "p_", "ph"], ["Prueba", "Qué significaría", "Mediana", "p", "p Holm"],
                       num=("med", "p_", "ph")))
    s.append('<h3>Movimiento coordinado entre vecinas y efecto de la densidad</h3><div class="texto">')
    s.append(f"<p>Dos células que están a menos de 25 µm (en la práctica, que se tocan) se mueven en direcciones más parecidas de lo que dicta el azar: el alineamiento de direcciones hasta 50 µm es {mic('bf', 'corr_dir_0_50um', 3)}, mientras que entre 100 y 200 µm es prácticamente cero. Siguiendo a Liu et al. (2021), también comparamos la evolución en el tiempo de la SE de pares de células vecinas y lejanas.</p>")
    for ds_, nom in (("bf", "brightfield"), ("sirdna", "núcleos")):
        prx = csv("inferencia", ds_, "pruebas_nulo.csv")
        if prx is not None and prx.prueba.str.startswith("corr SE(t)").any():
            rr = prx[prx.prueba.str.startswith("corr SE(t)")].iloc[0]
            pe = csv(ds_, "estadisticas", "dist" if ds_ == "sirdna" else vbf, "por_pelicula.csv")
            s.append(f"<p>Con {nom}, la correlación media entre las series de SE de vecinas es {f(pe.corr_se_vecinos.mean(), 3)} y la de pares lejanos {f(pe.corr_se_lejanos.mean(), 3)} (p Holm {peq(rr.p_holm)}). " +
                     ("La diferencia es significativa, aunque muy pequeña: las vecinas cambian de persistencia de forma apenas coordinada.</p>" if rr.p_holm < 0.05 else "No hay diferencia: en estos cultivos la persistencia de cada célula evoluciona de forma independiente de la de sus vecinas.</p>"))
    if rel is not None:
        sig = rel[rel.p_holm < 0.05]
        if len(sig):
            s.append("<p>Entre videos, la densidad de células se relaciona significativamente con: " +
                     "; ".join(f"{NOMBRE.get(r.metrica, r.metrica)} (ρ = {f(r.rho_vs_densidad, 2)}, p Holm {peq(r.p_holm)})" for _, r in sig.iterrows()) + ".")
            if "se_ventana_corr" in set(sig.metrica) and sig.set_index("metrica").loc["se_ventana_corr", "rho_vs_densidad"] > 0:
                s.append(" En los videos más densos el movimiento es más aleatorio: las células chocan más a menudo con sus vecinas y cambian de rumbo con más frecuencia. La película 15, la más densa, es a la vez la más rápida y la de SE más alta (el movimiento más aleatorio de todas).")
            s.append("</p>")
        else:
            s.append("<p>Con 16 videos, ninguna medida se relaciona significativamente con la densidad de células después de corregir por comparaciones múltiples (la tabla completa está en el anexo).</p>")
    if het is not None:
        h = het.set_index("metrica")
        s.append(f"<p>Los videos difieren bastante entre sí: el video al que pertenece una célula explica el {f(100 * h.loc['rapidez_um_min', 'icc_pelicula'], 0)}% de la variación en rapidez entre células y el {f(100 * h.loc['ead1_corr', 'icc_pelicula'], 0)}% de la variación en EAD₁. Esto confirma que tratar a cada célula como una réplica independiente exageraría la certeza de las conclusiones.</p>")
    s.append("</div>")
    s.append(figura("figuras/bf_colectivo_acoplamiento.png",
                    "<b>Coordinación entre células.</b> Izquierda: cuánto se parecen las direcciones de dos células según la distancia que las separa. Centro: correlación entre las series de SE de pares vecinos y lejanos; cada línea es un video. Derecha: acoplamiento entre rapidez y persistencia en cada video, coloreado por densidad."))
    s.append(figura("figuras/bf_vs_densidad.png", "<b>Medidas de cada video según su densidad de células.</b> El número junto a cada punto identifica el video."))
    rel = csv("inferencia", "bf", "relacion_densidad.csv")
    if rel is not None:
        t = rel.copy()
        t["m"] = t.metrica.map(NOMBRE)
        t["r"] = t.rho_vs_densidad.map(lambda x: f(x, 2))
        t["ph"] = [f'<span class="{"sig" if p < 0.05 else "nosig"}">{fp(p)}</span>' for p in t.p_holm]
        s.append(tabla(t, ["m", "r", "ph"], ["Medida", "ρ con la densidad (entre películas)", "p Holm"], num=("r", "ph")))
    s.append(seccion_forma("bf"))
    return "\n".join(s)


NOMBRE = {
    "rapidez_um_min": "Rapidez (µm/min)", "direccionalidad_1h": "Direccionalidad en 1 h", "prw_P_min": "Persistencia P (min)",
    "prw_D_um2_min": "Coef. de difusión D (µm²/min)", "prw_S_um_min": "Rapidez del modelo S (µm/min)",
    "prw_sigma_um": "Error de posición σ (µm)", "alfa_corto": "α a tiempos cortos", "alfa_largo": "α a tiempos largos",
    "vacf_lag1": "Autocorrelación de velocidad a 1 paso", "cos_giro_medio": "Coseno medio del giro",
    "rho_rapidez_cosgiro": "Acoplamiento rapidez–persistencia (ρ)", "se_fps_completa": "SE (trayectoria completa, cruda)",
    "se_ventana": "SE (ventana 80 min, cruda)", "se_ventana_corr": "SE corregida", "se_wavelet": "SE en el tiempo (cruda)",
    "se_wavelet_corr": "SE en el tiempo (corregida)", "ead1_cel": "EAD₁ por célula (cruda)", "ead1_cel_corr": "EAD₁ por célula (corregida)",
    "tl1_cel_mediana_min": "TL₁ por célula, mediana (min)", "ead1_ens": "EAD₁ del conjunto (cruda)",
    "ead1_ens_corr": "EAD₁ del conjunto (corregida)", "tl1_ens_min": "TL₁ del conjunto (min)",
    "ead_t_pct_0_03": "% del tiempo con EAD &lt; 0.3", "ead_t_pct_03_06": "% del tiempo con EAD 0.3–0.6",
    "ead_t_pct_06_1": "% del tiempo con EAD ≥ 0.6", "corr_dir_0_50um": "Alineamiento con vecinas (0–50 µm)",
    "corr_dir_100_200um": "Alineamiento a 100–200 µm", "corr_se_vecinos": "Correlación de SE(t), vecinas",
    "corr_se_lejanos": "Correlación de SE(t), lejanas", "area_um2": "Área celular (µm²)", "aspect_ratio": "Alargamiento (largo/ancho)",
    "circularidad": "Circularidad", "solidez": "Solidez", "densidad_cel_mm2": "Densidad (células/mm²)",
    "n_segmentos": "Trayectorias analizadas",
}


def tabla_resumen(ds, metricas):
    t = csv("inferencia", ds, "resumen_bootstrap.csv")
    if t is None:
        return ""
    t = t.set_index("metrica")
    filas = []
    for m in metricas:
        if m not in t.index:
            continue
        r = t.loc[m]
        d = 3 if abs(r.media) < 10 else 1
        filas.append({"m": NOMBRE.get(m, m), "v": f(r.media, d), "ic": f"{f(r.ic95_inf, d)} – {f(r.ic95_sup, d)}",
                      "sd": f(r.sd, d), "n": int(r.n)})
    return tabla(pd.DataFrame(filas), ["m", "v", "ic", "sd", "n"],
                 ["Medida", "Promedio entre videos", "IC95%", "Desvío entre videos", "Videos"], num=("v", "ic", "sd", "n"))


def seccion_bf_vs_nuc():
    t = csv("inferencia", "bf_vs_nucleos.csv")
    s = ['<div class="texto"><p>Las mismas medidas se pueden calcular siguiendo los núcleos fluorescentes en lugar de las células en brightfield. Si ambas vías dan resultados parecidos, el análisis sin marcador (que no requiere teñir las células y es el único posible en CAMAD) es confiable para esas medidas.</p>']
    if t is not None:
        keep = ["rapidez_um_min", "prw_P_min", "prw_sigma_um", "direccionalidad_1h", "cos_giro_medio", "ead1_ens_corr",
                "ead1_cel_corr", "tl1_ens_min", "se_ventana_corr", "se_wavelet_corr", "corr_dir_0_50um", "alfa_largo"]
        t = t[t.metrica.isin(keep)].copy()
        t["orden"] = t.metrica.map({m: i for i, m in enumerate(keep)})
        t = t.sort_values("orden")
        buenas = t[t.ccc_lin >= 0.8].metrica.map(NOMBRE).tolist()
        malas = t[t.ccc_lin < 0.5].metrica.map(NOMBRE).tolist()
        sig = t[t.p_holm < 0.05]
        s.append(f"<p>Coinciden bien (concordancia de Lin ≥ 0.8) en: {', '.join(buenas) if buenas else 'ninguna'}. " +
                 (f"Coinciden poco (concordancia &lt; 0.5) en: {', '.join(malas)}. " if malas else "") +
                 (f"Hay diferencias sistemáticas significativas en: " + "; ".join(f"{NOMBRE[r.metrica]} ({f(r['diferencia_relativa_%'], 0)}%)" for _, r in sig.iterrows()) + "." if len(sig) else "No hay diferencias sistemáticas significativas.") + "</p>")
        s.append("<p>La diferencia más importante es el <b>error de posición</b>: el centro del núcleo es un punto mucho más estable que el centro del contorno de toda la célula, que cambia de forma al moverse. Por eso las medidas de persistencia a pasos cortos salen algo más persistentes con los núcleos.</p>")
        t["bf"] = t.media_bf.map(lambda x: f(x, 3))
        t["nu"] = t.media_nucleos.map(lambda x: f(x, 3))
        t["d"] = t["diferencia_relativa_%"].map(lambda x: f(x, 1) + "%")
        t["c"] = t.ccc_lin.map(lambda x: f(x, 2))
        t["ph"] = [f'<span class="{"sig" if p < 0.05 else "nosig"}">{fp(p)}</span>' for p in t.p_holm]
        t["m"] = t.metrica.map(NOMBRE)
        s.append("</div>")
        s.append(tabla(t, ["m", "bf", "nu", "d", "c", "ph"], ["Medida", "Células (brightfield)", "Núcleos", "Diferencia",
                                                               "Concordancia", "p Holm"], num=("bf", "nu", "d", "c", "ph")))
    else:
        s.append("</div>")
    s.append(figura("figuras/bf_vs_nucleos.png",
                    "<b>La misma medida calculada con células y con núcleos.</b> Cada punto es un video; si ambos métodos coincidieran exactamente, los puntos estarían sobre la línea punteada."))
    s.append(seccion_nucleos_detalle())
    return "\n".join(s)


def seccion_camad(vca):
    s = ['<div class="texto">']
    peli = csv("camad", "estadisticas", vca, "por_pelicula.csv")
    omni = csv("inferencia", "camad", "omnibus_experimentos.csv")
    mix = csv("inferencia", "camad", "modelo_mixto_vs_vidrio.csv")
    if peli is None or not (RES / "figuras" / "camad_condiciones.png").exists():
        return ('<div class="aviso texto"><strong>En proceso</strong><p>La segmentación de los 9438 cuadros de CAMAD '
                'todavía está corriendo. Esta sección se completa automáticamente cuando termine.</p></div>')
    s.append("<p>En CAMAD las células se filmaron durante las primeras 5 horas después de sembrarlas en un medio sin suero sobre cinco superficies distintas. Es la etapa en que se adhieren, se aplanan y empiezan a moverse. Para que los números sean comparables con el dataset brightfield, las medidas de persistencia usan también un paso de 5 minutos (una de cada diez imágenes).</p>")
    s.append("</div>")
    vids = [("camad_exp14.mp4", "Vidrio (exp14)"), ("camad_exp01.mp4", "Matrigel (exp1)"), ("camad_exp12.mp4", "Colágeno I (exp12)"),
            ("camad_exp04.mp4", "Matriz de macrófagos dispersa (exp4)"), ("camad_exp07.mp4", "Matriz de macrófagos confluente (exp7)"),
            ("camad_exp09.mp4", "Matriz 231, identidad dudosa (exp9)")]
    s.append('<div class="rejilla-videos">')
    for v, c in vids:
        s.append(video(f"videos/{v}", f"<b>{c}.</b> Una imagen cada 2 minutos durante 5 horas."))
    s.append("</div>")
    if peli is not None:
        g = peli[peli.linea == "MDA-MB-231"].groupby("condicion")
        filas = []
        for m in ["rapidez_um_min", "direccionalidad_1h", "prw_P_min", "ead1_ens_corr", "se_ventana_corr", "tl1_cel_mediana_min",
                  "area_um2", "circularidad", "n_segmentos"]:
            fila = {"m": NOMBRE.get(m, m)}
            for c in CAMAD_ORDEN_CONDICIONES:
                x = peli[peli.condicion == c][m].dropna()
                d = 2 if (len(x) and abs(x.mean()) < 10) else 0
                fila[c] = f"{f(x.mean(), d)} ± {f(x.std(), d)}" if len(x) > 1 else (f(x.mean(), d) if len(x) else "–")
            if omni is not None and (omni.metrica == m).any():
                o = omni[omni.metrica == m].iloc[0]
                fila["p"] = f'<span class="{"sig" if o.anova_perm_p_holm < 0.05 else "nosig"}">{fp(o.anova_perm_p)} ({fp(o.anova_perm_p_holm)})</span>'
            else:
                fila["p"] = "–"
            filas.append(fila)
        cols = ["m"] + CAMAD_ORDEN_CONDICIONES + ["p"]
        heads = ["Medida"] + [c.replace("Matriz 231 (exp8-9)", "Matriz 231*") for c in CAMAD_ORDEN_CONDICIONES] + ["p (Holm)"]
        s.append('<div class="texto"><p>Promedio ± desvío entre experimentos. La última columna es la prueba de si los cinco sustratos de MDA-MB-231 difieren entre sí (ANOVA por permutación a nivel de experimento; entre paréntesis, corregida por Holm). La columna «Matriz 231*» no entra en la prueba.</p></div>')
        s.append(tabla(pd.DataFrame(filas), cols, heads, num=tuple(cols[1:])))
    s.append(figura("figuras/camad_condiciones.png",
                    "<b>Cada punto es un experimento.</b> La barra negra es el promedio del sustrato y la línea vertical su intervalo de confianza del 95%. Con dos a cuatro experimentos por sustrato, los intervalos son amplios."))
    s.append('<div class="texto">')
    # --- interpretación (números calculados de la tabla por experimento)
    import itertools
    def rango(cond, m, d=0):
        x = peli[peli.condicion == cond][m]
        a, b = f(x.min(), d), f(x.max(), d)
        return f"{a}–{b}" if len(x) > 1 and a != b else a
    def perm_exacta(m, menor=True):
        q = peli[peli.condicion.isin(["Vidrio", "Matrigel", "Colágeno I"])]
        v, g = q[m].to_numpy(), (q.condicion == "Vidrio").to_numpy()
        obs = v[g].mean()
        combs = list(itertools.combinations(range(len(v)), int(g.sum())))
        c = sum((v[list(k)].mean() <= obs) if menor else (v[list(k)].mean() >= obs) for k in combs)
        return c / len(combs)
    s.append("<h3>Lo que se ve con claridad: las células se extienden sobre Matrigel y colágeno, no sobre vidrio</h3>")
    s.append(f"<p>Sobre Matrigel y colágeno las células se aplanan y se extienden: su área media es de {rango('Matrigel', 'area_um2')} µm² y {rango('Colágeno I', 'area_um2')} µm², con circularidad de {rango('Matrigel', 'circularidad', 2)} y {rango('Colágeno I', 'circularidad', 2)} (formas alargadas e irregulares). Sobre vidrio sin recubrimiento, en un medio sin suero, siguen redondas: {rango('Vidrio', 'area_um2')} µm² y circularidad {rango('Vidrio', 'circularidad', 2)}. En los seis experimentos con recubrimiento el área es mayor y la circularidad menor que en los dos de vidrio. Es el resultado más extremo posible, con una prueba exacta de permutación a nivel de experimento de p = {f(perm_exacta('area_um2', True), 3)}, que además es el valor más chico que permiten 2 contra 6 experimentos. Es lo esperable: las proteínas de la matriz (laminina y colágeno) ofrecen puntos de anclaje a las integrinas, y sin ellas ni suero la célula casi no puede adherirse. El panel central de la figura temporal muestra cómo el área crece durante la primera hora sobre Matrigel y colágeno.</p>")
    s.append(f"<h3>Movimiento y persistencia: tendencias, no diferencias firmes</h3>")
    s.append(f"<p>La rapidez media fue de {rango('Matrigel', 'rapidez_um_min', 2)} µm/min sobre Matrigel, {rango('Colágeno I', 'rapidez_um_min', 2)} sobre colágeno, {rango('Matriz RAW confluente', 'rapidez_um_min', 2)} sobre la matriz confluente de macrófagos y {rango('Matriz RAW dispersa', 'rapidez_um_min', 2)} sobre la matriz dispersa. Sobre vidrio los dos experimentos no coinciden ({rango('Vidrio', 'rapidez_um_min', 2)}) y solo aportan 6 y 7 trayectorias. Además, en exp14 el coseno medio del giro es negativo: las células redondas y mal adheridas parecen «bambolearse» más que migrar.</p>")
    s.append("<p>La persistencia medida con entropía (EAD₁ y SE corregidas, de 0.95 a 1.00) está cerca del movimiento al azar en todos los sustratos, sin diferencias detectables. Es coherente con el momento del experimento: durante las primeras horas tras la siembra y sin suero, las células exploran y se extienden más de lo que migran en una dirección.</p>")
    s.append("<p>Con 2 a 4 experimentos por sustrato, las diferencias de movimiento entre sustratos no resultan significativas a nivel de experimento después de corregir por comparaciones múltiples (tabla anterior). El análisis por célula, que respeta que las células de un mismo experimento no son independientes, apunta en la misma dirección que la figura. Pero con solo 2 experimentos de vidrio como referencia sus valores de p son optimistas (tabla desplegable), así que conviene leerlo como una <b>tendencia a confirmar</b>.</p>")
    s.append("<div class=\"aviso\"><strong>Población mezclada en la matriz dispersa (exp4 y exp5)</strong><p>En esos dos experimentos hay entre 1000 y 1600 objetos por mm², diez veces más que en el resto y muchos más de los que corresponden a las células sembradas. La mayoría son células pequeñas y redondas (~15 µm) que casi no se mueven, probablemente macrófagos que quedaron de la preparación de la matriz. Cellpose los segmenta bien, pero no hay forma de separarlos de las MDA-MB-231 sin un marcador. Los valores de esa condición describen a la mezcla, no a las MDA-MB-231.</p></div>")
    s.append("<div class=\"explica\"><strong>Una lección de método: la EAD mide concentración, no dirección</strong><p>La figura de sensibilidad muestra algo que conviene tener presente al usar la EAD. Con pasos de 30 segundos la EAD₁ por célula baja, lo que en principio indicaría más persistencia, pero el coseno medio del giro se vuelve cero o negativo. El error de posición hace que dos pasos cortos consecutivos tiendan a apuntar en sentidos opuestos, y esos retrocesos (giros cercanos a 180°) también concentran la distribución de ángulos y bajan la entropía. Por eso la EAD siempre debe leerse junto con el coseno medio del giro, que distingue «sigue derecho» de «va y viene», y con un paso de análisis lo bastante largo.</p></div>")
    s.append("</div>")
    if mix is not None and len(mix):
        t = mix.copy()
        t["m"] = t.metrica.map({"rapidez_um_min": "Rapidez (µm/min)", "direccionalidad_1h": "Direccionalidad 1 h",
                                "se_ventana_corr": "SE corregida", "se_wavelet_media_corr": "SE en el tiempo corregida",
                                "ead1_corr": "EAD₁ corregida", "tl1_min": "TL₁ (min)", "area_um2": "Área (µm²)",
                                "aspect_ratio": "Alargamiento", "circularidad": "Circularidad"}).fillna(t.metrica)
        t["d"] = [f"{f(r.diferencia, 3)} [{f(r.ic95_inf, 3)}, {f(r.ic95_sup, 3)}]" for _, r in t.iterrows()]
        t["ph"] = [f'<span class="{"sig" if p < 0.05 else "nosig"}">{fp(p)}</span>' for p in t.p_holm]
        s.append('<details><summary>Análisis por célula: diferencia de cada sustrato respecto de vidrio (modelo mixto u OLS con errores agrupados por experimento; con solo 2 experimentos de vidrio, las p son optimistas)</summary>')
        s.append(tabla(t, ["m", "condicion", "d", "ph"], ["Medida", "Sustrato", "Diferencia [IC95%]", "p Holm"], num=("d", "ph")))
        s.append("</details>")
    s.append(seccion_alfa("camad", vca))
    s.append(seccion_forma("camad"))
    s.append(figura("figuras/camad_tiempo.png",
                    "<b>Qué pasa durante las 5 horas.</b> Cada línea es un experimento. Fila superior: rapidez. Centro: área de las células, que crece a medida que se aplanan sobre el sustrato. Abajo: EAD corregida en ventanas de una hora."))
    s.append(figura("figuras/camad_sensibilidad_paso.png",
                    "<b>El intervalo de tiempo importa.</b> Con 30 segundos entre imágenes, el desplazamiento de una célula es comparable al error de posición y el movimiento parece al azar, o incluso con tendencia a «volver atrás» (coseno negativo), que es la huella del ruido. Recién con pasos de varios minutos aparece la persistencia real."))
    s.append(figura("figuras/camad_ead_tau.png", "<b>EAD según el intervalo τ en cada sustrato.</b> Cada línea es un experimento."))
    return "\n".join(s)


def seccion_whad():
    p = csv("whad", "por_posicion.csv")
    s = ['<div class="texto">']
    if p is not None:
        g = p.groupby(["linea", "condicion"]).agg(v=("velocidad_frente_um_h", "mean"), c=("cierre_12h_pct", "mean"),
                                                  d=("desprendidas_media", "mean"), n=("id", "size")).reset_index()
        s.append("<p>En este ensayo se raya una franja sin células en una capa confluente y se mide cuánto tarda en cerrarse. Usamos directamente los contornos de la herida dibujados por los autores del dataset, así que aquí no interviene nuestra segmentación. Las células se trataron con mitomicina C, que frena la división celular, de modo que el cierre se debe a migración.</p>")
        s.append("<p>" + " ".join(
            f"{r.linea} {r.condicion}: frente a {f(r.v, 1)} µm/h, {f(r.c, 0)}% cerrado a las 12 h, {f(r.d, 1)} grupos desprendidos por imagen (n = {r.n} {'posición' if r.n == 1 else 'posiciones'})."
            for _, r in g.iterrows()) + "</p>")
        s.append("<p>Según lo publicado con estos mismos datos (Gunyuz et al. 2022; resumido en Iheme et al. 2024), la sobreexpresión de SEMA6D induce la migración de las MCF7 y las lleva a un fenotipo más «desprendido». Aquí se ve algo compatible: con SEMA6D hay muchas más células o grupos sueltos por imagen, y el cierre de la herida como lámina es más lento, como si las células migraran por separado en lugar de avanzar juntas. Sin embargo, hay una sola posición con SEMA6D y las posiciones de una misma condición provienen del mismo pocillo, así que este resultado es solo descriptivo.</p>")
    s.append("</div>")
    s.append(figura("figuras/whad_cierre.png",
                    "<b>Cierre de herida.</b> Izquierda: área de la herida relativa a la inicial; las líneas finas son posiciones y las gruesas el promedio por condición. Derecha: células o grupos desprendidos por imagen. MCF10A: CC control, NC Notch1 activo, C61 sin CYR61, N61 ambos. MCF7: LacZ control, SEMA6D sobreexpresado."))
    return "\n".join(s)

def seccion_alfa(ds, var):
    """Régimen de movimiento por réplica: MSD de cada una y sus alfa."""
    pe = csv(ds, "estadisticas", var, "por_pelicula.csv")
    ce = csv(ds, "estadisticas", var, "por_celula.csv")
    if pe is None:
        return ""
    s = ['<h3>Régimen de movimiento en cada réplica: el exponente α del MSD</h3><div class="texto">',
         "<p>El exponente α describe cómo crece la distancia recorrida con el tiempo: en un gráfico log-log del MSD es la pendiente. α = 1 corresponde a una caminata al azar (difusiva); α = 2, a un movimiento en línea recta (balístico); entre 1 y 2 el movimiento es <b>superdifusivo</b> (persistente) y por debajo de 1 es <b>subdifusivo</b> (confinado o frenado). Lo medimos de tres formas: en cada réplica a escala corta y larga, célula por célula, y como «α local» a cada escala de tiempo.</p>"]
    if ds == "bf":
        m13 = pe.set_index("movie").loc[13]
        s.append(f"<p>En las 16 películas el α a escala corta (5–30 min) está entre {f(pe.alfa_corto.min())} y {f(pe.alfa_corto.max())}: todas son superdifusivas, como corresponde a células que mantienen su dirección durante varios minutos. A escala larga (1–4 h) el α baja, entre {f(pe.alfa_largo.min())} y {f(pe.alfa_largo.max())}, y el α local muestra el cruce de régimen: sube hasta ~1.3 alrededor de 20–40 min y cae hacia 1 después de ~2 h, que es exactamente lo que predice el modelo de caminata persistente. Célula por célula, la mediana de α por película va de {f(pe.alfa_celula_mediana.min())} a {f(pe.alfa_celula_mediana.max())}; en promedio el {f(100 * pe.frac_celulas_superdifusivas.mean(), 0)}% de las células es claramente superdifusiva (α > 1.2) y el {f(100 * pe.frac_celulas_subdifusivas.mean(), 0)}% subdifusiva (α < 0.8). La película 13, la de menor densidad y menor rapidez, se aparta del resto: α largo {f(m13.alfa_largo)} y {f(100 * m13.frac_celulas_subdifusivas, 0)}% de células subdifusivas (su MSD se curva hacia abajo a tiempos largos).</p>")
        s.append("<p>Un α largo algo mayor que 1 (1.1–1.3) en casi todas las películas puede reflejar persistencia de largo plazo, pero también un sesgo conocido: a tiempos largos solo aportan las células que permanecen muchas horas en el campo y se siguieron sin cortes, que tienden a ser las más persistentes. Por eso el α largo se interpreta con cautela.</p>")
    elif ds == "camad":
        s.append("<p>En CAMAD el α es más informativo que el modelo de caminata persistente, porque varias condiciones no se comportan como una caminata persistente. <b>Sobre vidrio</b> el MSD se aplana a los ~30–60 minutos: las células se bambolean en su lugar dentro de un radio de pocos micrómetros, sin migrar (α largo ≤ 0.7 y entre 67% y 100% de células subdifusivas). <b>Sobre la matriz dispersa</b> el movimiento es subdifusivo (α ≈ 0.6), coherente con una población mayormente inmóvil. <b>Sobre Matrigel, colágeno y la matriz confluente</b> el α largo está entre 1.0 y 1.8: las células migran, y en algunos experimentos (exp6, exp11, exp3, exp7) de forma claramente dirigida. En los seis experimentos con Matrigel o colágeno el α largo es mayor que en los dos de vidrio (prueba exacta a nivel de experimento, p = 0.036). A escala corta (0.5–5 min) el α es menor que 1 en casi todos: a esa escala domina el error de posición.</p>")
    else:
        s.append(f"<p>Con los núcleos el patrón es el mismo que con las células completas: α corto entre {f(pe.alfa_corto.min())} y {f(pe.alfa_corto.max())} y α largo entre {f(pe.alfa_largo.min())} y {f(pe.alfa_largo.max())}.</p>")
    s.append("</div>")
    nombre = {"bf": "Brightfield", "sirdna": "Núcleos", "camad": "CAMAD"}[ds]
    s.append(figura(f"figuras/{ds}_msd_por_replica.png", f"<b>{nombre}: el MSD de cada réplica por separado</b> (escala log-log). Puntos: datos; naranja: ajuste del modelo de caminata persistente con error de posición; líneas punteadas: ajuste de α a escala corta (negro) y larga (verde). En el título de cada panel, los valores de esa réplica."))
    s.append(figura(f"figuras/{ds}_alfa.png", f"<b>{nombre}: α por réplica, por célula y a cada escala de tiempo.</b> Izquierda: α corto y largo de cada réplica. Centro: distribución del α de cada célula en cada réplica (caja: cuartiles; bigotes: 1.5 veces el rango intercuartil). Derecha: α local según τ para cada réplica (gris) y el promedio (azul); las líneas punteadas marcan α = 1 y α = 2."))
    return "\n".join(s)


def seccion_forma(ds):
    t = csv("inferencia", ds, "forma_movimiento.csv")
    if t is None or not len(t):
        return ""
    t = t.set_index(["forma", "movimiento"])
    g = lambda a, b: t.loc[(a, b)]
    s = ['<h3>¿La forma de la célula predice cómo se mueve?</h3><div class="texto">']
    if ds in ("bf", "sirdna"):
        r1, r2, r3 = g("aspect_ratio", "rapidez_um_min"), g("aspect_ratio", "ead1_corr"), g("circularidad", "rapidez_um_min")
        s.append(f"<p>Para cada célula comparamos su forma media con su movimiento, calculando la correlación <i>dentro</i> de cada película y usando las películas como réplicas; así una diferencia entre campos no se confunde con una relación real. Resultado: <b>las células más alargadas se mueven más rápido y con más persistencia</b>. La correlación de Spearman entre alargamiento y rapidez es {f(r1.rho_mediana_dentro)} (mediana entre películas; p Holm {peq(r1.p_holm)}), y entre alargamiento y EAD₁ es {f(r2.rho_mediana_dentro)} (EAD más baja significa más persistencia). Las células redondas son más lentas (circularidad vs rapidez: {f(r3.rho_mediana_dentro)}). Es la firma de la polaridad frente–atrás: una célula que migra se alarga, con un frente que avanza y una cola que se retrae.</p>")
        s.append("<p>Las correlaciones son moderadas (|ρ| ≈ 0.1–0.3): la forma explica una parte, no todo, del movimiento de cada célula. La versión anterior del análisis había encontrado correlaciones más débiles (≤ 0.16); con la segmentación y el seguimiento validados la señal es más clara. Con los núcleos (su forma también cambia al migrar) se obtiene el mismo patrón, lo que descarta que sea un artefacto de la segmentación del contorno celular.</p>")
    else:
        r = g("circularidad", "rapidez_um_min")
        s.append(f"<p>En CAMAD, dentro de cada experimento, las células más redondas también son más lentas (circularidad vs rapidez, ρ = {f(r.rho_mediana_dentro)}), pero con solo 10 experimentos con suficientes células y unas pocas trayectorias por experimento ninguna correlación sobrevive a la corrección por comparaciones múltiples (p Holm ≥ {fp(t.p_holm.min())}).</p>")
    s.append("</div>")
    nombre = {"bf": "Brightfield", "sirdna": "Núcleos", "camad": "CAMAD"}[ds]
    s.append(figura(f"figuras/{ds}_forma_movimiento.png", f"<b>{nombre}: correlación entre la forma y el movimiento de cada célula</b> (mediana de las correlaciones calculadas dentro de cada réplica). Rojo: relación positiva; azul: negativa. Las estrellas indican significancia con las réplicas como unidad y corrección de Holm."))
    return "\n".join(s)


def tabla_replicas(ds, var):
    pe = csv(ds, "estadisticas", var, "por_pelicula.csv")
    if pe is None:
        return ""
    cols = [("movie", "Réplica", 0), ("n_segmentos", "Trayect.", 0), ("densidad_cel_mm2", "Densidad /mm²", 0),
            ("rapidez_um_min", "Rapidez µm/min", 2), ("direccionalidad_1h", "Direcc. 1 h", 2),
            ("alfa_corto", "α corto", 2), ("alfa_largo", "α largo", 2), ("alfa_celula_mediana", "α célula (med.)", 2),
            ("prw_P_min", "P min", 1), ("prw_S_um_min", "S µm/min", 2), ("prw_sigma_um", "σ µm", 2), ("prw_r2", "R² PRW", 3),
            ("vacf_lag1", "VACF(1)", 2), ("cos_giro_medio", "⟨cos giro⟩", 2), ("rho_rapidez_cosgiro", "ρ rap.–giro", 2),
            ("ead1_ens_corr", "EAD₁ ens.", 3), ("ead1_cel_corr", "EAD₁ cél.", 3), ("tl1_ens_min", "TL₁ min", 0),
            ("se_ventana_corr", "SE", 3), ("se_wavelet_corr", "SE(t)", 3), ("corr_dir_0_50um", "Alin. 0–50 µm", 3),
            ("area_um2", "Área µm²", 0), ("aspect_ratio", "Alarg.", 2), ("circularidad", "Circ.", 2)]
    t = pe.copy()
    if ds == "camad":
        t["movie"] = [f"exp{m} ({c})" for m, c in zip(t.movie, t.condicion)]
        orden = {c: i for i, c in enumerate(CAMAD_ORDEN_CONDICIONES)}
        t = t.assign(o=pe.condicion.map(orden)).sort_values(["o", "movie"])
    for c, _, d in cols[1:]:
        t[c] = t[c].map(lambda x, d=d: f(x, d))
    return tabla(t, [c for c, _, _ in cols], [h for _, h, _ in cols], num=tuple(c for c, _, _ in cols[1:]))


def seccion_tablas_replica(vbf, vca):
    s = ['<div class="texto"><p>Todas las medidas de cada película o experimento, para quien quiera revisar réplica por réplica. Abreviaturas: P, S y σ son la persistencia, la rapidez y el error de posición del modelo de caminata persistente; VACF(1) es la autocorrelación de la velocidad a un paso; ρ rap.–giro es el acoplamiento rapidez–persistencia; EAD₁ ens. y cél. son la EAD corregida del conjunto de células y el promedio por célula; SE es la entropía corregida del espectro; «Alin.» es el alineamiento de direcciones entre vecinas. Los mismos datos, con más columnas, están en <span class="mono">resultados/v2/&lt;dataset&gt;/estadisticas/&lt;variante&gt;/por_pelicula.csv</span>.</p></div>']
    for ds, var, tit in (("bf", vbf, "Brightfield (16 películas)"), ("sirdna", "dist", "Núcleos SiR-DNA (16 películas)"),
                         ("camad", vca, "CAMAD (16 experimentos)")):
        s.append(f"<details open><summary><b>{tit}</b></summary>{tabla_replicas(ds, var)}</details>")
    for ds, tit in (("bf", "Brightfield"), ("sirdna", "Núcleos")):
        h = csv("inferencia", ds, "heterogeneidad_peliculas.csv")
        if h is not None:
            h = h.copy()
            h["m"] = h.metrica.map({**NOMBRE, "alfa_celula": "α célula", "ead1_corr": "EAD₁ corregida", "tl1_min": "TL₁ (min)",
                                    "se_wavelet_media_corr": "SE en el tiempo corregida"}).fillna(h.metrica)
            h["i"] = h.icc_pelicula.map(lambda x: f"{f(100 * x, 0)}%")
            h["k"] = h.kruskal_p_entre_peliculas.map(fp)
            s.append(f"<details><summary><b>{tit}: cuánto de la variación entre células se debe a la película</b></summary>")
            s.append(tabla(h, ["m", "i", "k"], ["Medida", "Varianza explicada por la película (ICC)", "p Kruskal-Wallis entre películas"], num=("i", "k")))
            s.append("</details>")
    pr = csv("inferencia", "sirdna", "pruebas_nulo.csv")
    if pr is not None:
        t = pr.copy()
        t["med"] = t.mediana.map(lambda x: f(x, 3))
        t["ph"] = [f'<span class="{"sig" if p < 0.05 else "nosig"}">{fp(p)}</span>' for p in t.p_holm]
        s.append("<details><summary><b>Núcleos: pruebas contra el azar</b></summary>")
        s.append(tabla(t, ["prueba", "hipotesis", "med", "ph"], ["Prueba", "Qué significaría", "Mediana", "p Holm"], num=("med", "ph")))
        s.append("</details>")
    o = csv("inferencia", "camad", "omnibus_experimentos.csv")
    if o is not None:
        t = o.copy()
        t["m"] = t.metrica.map(NOMBRE).fillna(t.metrica)
        for c in ["kruskal_p", "anova_perm_p", "anova_perm_p_holm"]:
            t[c + "_"] = t[c].map(fp)
        s.append("<details><summary><b>CAMAD: comparación entre los cinco sustratos, todas las medidas</b></summary>")
        s.append(tabla(t, ["m", "kruskal_p_", "anova_perm_p_", "anova_perm_p_holm_"],
                       ["Medida", "p Kruskal-Wallis", "p ANOVA por permutación", "p permutación (Holm)"],
                       num=("kruskal_p_", "anova_perm_p_", "anova_perm_p_holm_")))
        s.append("</details>")
    wh = csv("whad", "por_posicion.csv")
    if wh is not None:
        t = wh.copy()
        t["pos"] = t.grupo + " · " + t.posicion.astype(str)
        for c, d in (("velocidad_frente_um_h", 1), ("cierre_12h_pct", 0), ("t50_h", 1), ("desprendidas_media", 1)):
            t[c + "_"] = t[c].map(lambda x, d=d: f(x, d))
        s.append("<details><summary><b>WHAD: cada posición</b></summary>")
        s.append(tabla(t, ["linea", "condicion", "pos", "velocidad_frente_um_h_", "cierre_12h_pct_", "t50_h_", "desprendidas_media_"],
                       ["Línea", "Condición", "Experimento · posición", "Frente µm/h", "% cierre a 12 h", "t 50% (h)", "Desprendidas/imagen"],
                       num=("velocidad_frente_um_h_", "cierre_12h_pct_", "t50_h_", "desprendidas_media_")))
        s.append("</details>")
    return "\n".join(s)


def seccion_nucleos_detalle():
    s = ['<details><summary><b>Resultados completos con los núcleos (mismas figuras que para las células)</b></summary>']
    for fig_, cap in (("sirdna_msd_vacf_fps", "MSD, autocorrelación y espectro de la velocidad de los núcleos."),
                      ("sirdna_ead", "EAD de los núcleos: PDF de ángulos, EAD(τ) y distribución conjunta (TL₁, EAD₁)."),
                      ("sirdna_pasos_rapidez_giro", "Distribución de v², rapidez contra giro y acoplamiento rapidez–persistencia con núcleos."),
                      ("sirdna_colectivo_acoplamiento", "Coordinación entre núcleos vecinos."),
                      ("sirdna_sensibilidad_paso", "Sensibilidad al paso de análisis con núcleos."),
                      ("sirdna_mapas_temporales", "SE(t) y EAD(t) de los núcleos.")):
        s.append(figura(f"figuras/{fig_}.png", f"<b>Núcleos.</b> {cap}"))
    s.append(seccion_alfa("sirdna", "dist"))
    s.append(seccion_forma("sirdna"))
    s.append("</details>")
    return "\n".join(s)


def seccion_usos(vbf):
    s = ['<div class="texto">',
         "<p>Además del reporte, el pipeline deja tablas listas para análisis propios. Para cada video o experimento entrega:</p>",
         "<ul>",
         "<li><b>Una tabla por célula</b> (<span class=\"mono\">por_celula.csv</span>): rapidez media y mediana, direccionalidad, desplazamiento neto, α propio, persistencia por entropía (EAD₁, TL₁, SE, SE(t) promedio), área, alargamiento, circularidad, solidez, posición y momento de inicio. Es la base para comparar poblaciones de células o buscar subpoblaciones.</li>",
         "<li><b>Una tabla por réplica</b> (<span class=\"mono\">por_pelicula.csv</span>): todas las medidas poblacionales (MSD y sus α, modelo de caminata persistente, autocorrelación, entropías del conjunto, coordinación entre vecinas, densidad, morfología). Es la tabla sobre la que se hacen las comparaciones entre condiciones.</li>",
         "<li><b>Curvas</b>: MSD, autocorrelación de la velocidad, espectro de potencias, EAD(τ) y correlación espacial, por réplica.</li>",
         "<li><b>Series en el tiempo</b> (<span class=\"mono\">se_wavelet_tiempo.csv.gz</span>, <span class=\"mono\">ead_tiempo.csv.gz</span>): la persistencia de cada célula minuto a minuto, para detectar cambios de comportamiento (por ejemplo, tras agregar un fármaco).</li>",
         "<li><b>Trayectorias</b> (<span class=\"mono\">tracks.csv.gz</span>): posición, forma y etiqueta de cada célula en cada imagen; y con marcador nuclear, <b>linajes</b> (qué célula es hija de cuál).</li>",
         "<li><b>Control de calidad</b>: validación de la segmentación y del seguimiento, y videos para revisar a ojo.</li>",
         "</ul>",
         "<p><b>Preguntas de investigación que esto permite responder</b>, con los mismos scripts:</p>",
         "<ul>",
         "<li><b>¿Una condición cambia la migración?</b> (fármaco, sustrato, silenciamiento de un gen, línea celular). Se compara la tabla por réplica entre condiciones, como se hizo con los sustratos de CAMAD. Las medidas más estables entre réplicas (EAD₁, SE) necesitan menos réplicas para detectar un efecto (sección 14).</li>",
         "<li><b>¿Cuándo cambia?</b> Las series en el tiempo permiten ubicar el momento en que la persistencia o la rapidez cambian, por ejemplo tras un estímulo, como en el análisis de Liu et al. (2021) de células que pasan por canales.</li>",
         "<li><b>¿Hay subpoblaciones?</b> Con la tabla por célula se pueden agrupar células por su firma de movimiento y forma (por ejemplo, rápidas y alargadas contra lentas y redondas).</li>",
         "<li><b>¿Las células se coordinan?</b> Alineamiento entre vecinas y correlación de persistencia entre pares, según densidad o condición.</li>",
         "<li><b>¿Qué formas adoptan y cuándo cambian?</b> Mapa de formas, arquetipos geométricos y transiciones entre ellos (sección 9): por ejemplo, si un tratamiento empuja a las células hacia el estado «sólido» o cambia la proporción de husos y formas ramificadas.</li>",
         "<li><b>¿Qué papel juegan el ciclo celular y el núcleo?</b> Con un marcador nuclear, el contenido de ADN y la posición del núcleo se cruzan con el movimiento (sección 10).</li>",
         "<li><b>¿El movimiento distingue la agresividad?</b> Ver la sección siguiente.</li>",
         "</ul>",
         "<p><b>Recomendaciones para adquirir datos propios</b> (surgen de lo aprendido aquí): una imagen cada 1–2 minutos, para que el error de posición no oculte la persistencia; campos con densidad moderada; al menos 5 réplicas biológicas (días o placas distintos) por condición; si es posible, un marcador nuclear, que habilita la validación automática y los linajes; y anotar siempre la calibración (µm por píxel e intervalo). Con esas condiciones el pipeline se aplica con cambios mínimos de configuración.</p>",
         "</div>"]
    return "\n".join(s)


def seccion_formas():
    pr = csv("morfoespacio", "pruebas_bf.csv")
    cc = csv("morfoespacio", "correlacion_cruzada_bf.csv")
    ct = csv("morfoespacio", "camad_por_tiempo.csv")
    arq = csv("arquetipos", "arquetipos.csv")
    fr = csv("arquetipos", "friedman.csv")
    T = csv("arquetipos", "transiciones.csv")
    if pr is None or arq is None:
        return ""
    P = pr.set_index("prueba")
    s = ['<div class="texto">',
         "<p>Las máscaras guardan el contorno de cada célula en cada imagen, así que se puede estudiar la forma con tanto detalle como el movimiento. Hicimos tres cosas: un <b>mapa de formas</b> con la frontera de la transición «sólido–fluido» de la literatura, una <b>clasificación en arquetipos geométricos</b> y la relación de ambos con el movimiento.</p>",
         "<h3>Índice de forma y transición sólido–fluido</h3>",
         "<p>El índice de forma q = perímetro / √área vale 3.54 para un círculo y crece cuanto más alargada o irregular es la célula. En tejidos epiteliales se observó que q ≈ 3.81 separa un estado «sólido» o atascado (células compactas, casi quietas) de uno «fluido» (células alargadas que migran) (Bi et al., <i>Nat Phys</i> 2015; Park et al., <i>Nat Mater</i> 2015). Esa frontera se dedujo para monocapas confluentes, y además el perímetro digital puede inflar q un 2–4%, así que aquí se usa como referencia orientativa.</p>",
         f"<p><b>Resultado:</b> la rapidez muestra un salto justo alrededor de esa frontera. Las células con q ≤ 3.8 (casi redondas) se mueven a ~0.5 µm/min y al pasar a q ≈ 4 la rapidez sube a ~0.8 µm/min; más allá sigue creciendo, pero despacio. Dentro de cada película, las observaciones «fluidas» (q > 3.81) son en mediana {f(P.loc['rapidez(q>3.81) - rapidez(q<=3.81)', 'mediana'], 2)} µm/min más rápidas que las «sólidas», en todas las películas (p Holm {peq(P.loc['rapidez(q>3.81) - rapidez(q<=3.81)', 'p_holm'])}). En cambio, la persistencia paso a paso <b>no</b> depende de q (ρ = {f(P.loc['rho_q_cosgiro', 'mediana'])}). La forma indica <i>cuánto</i> se mueve una célula, pero no <i>qué tan derecho</i> va.</p>",
         f"<p><b>Las células avanzan a lo largo de su eje mayor.</b> El ángulo entre la dirección de movimiento y el eje mayor es menor que al azar (⟨|cos|⟩ supera el valor de azar en {f(P.loc['<|cos(mov, eje)|> - 2/pi', 'mediana'], 2)}; p Holm {peq(P.loc['<|cos(mov, eje)|> - 2/pi', 'p_holm'])}), y tanto más cuanto más alargadas son (ρ = {f(P.loc['rho_alarg_coseje', 'mediana'])}).</p>"]
    if cc is not None:
        m = cc.groupby("lag")["corr"].mean()
        s.append(f"<p><b>La forma se adelanta al movimiento.</b> La correlación entre el alargamiento de una célula en un instante y su rapidez es {f(m.loc[0])} en el mismo paso, se mantiene en {f(m.loc[1])} cinco minutos después y todavía es {f(m.loc[6])} a los 30 minutos. En cambio, la rapidez actual no predice el alargamiento posterior (correlaciones ≈ 0 a desfases negativos). Primero la célula se estira y después avanza, lo que es coherente con que la protrusión del frente preceda al desplazamiento.</p>")
    s.append("</div>")
    s.append(figura("figuras/morfo_bf_mapa.png", "<b>Mapa de formas (brightfield).</b> Cada hexágono agrupa observaciones de células con forma parecida (índice de forma en el eje horizontal, solidez en el vertical: 1 = contorno sin entrantes). De izquierda a derecha: cuántas observaciones hay, rapidez mediana, persistencia (rojo = sigue derecho, azul = gira) y cuánto se mueve la célula a lo largo de su eje mayor. La línea punteada es q = 3.81."))
    s.append(figura("figuras/morfo_bf_fase.png", "<b>Izquierda:</b> «diagrama de fase» densidad local contra forma, coloreado por rapidez. <b>Centro:</b> rapidez media según el índice de forma (gris: cada película; azul: todas); el salto está cerca de q ≈ 3.8. <b>Derecha:</b> correlación entre el alargamiento en un instante y la rapidez en instantes posteriores (desfase positivo) o anteriores (negativo)."))
    s.append(figura("figuras/morfo_bf_eje.png", "<b>Izquierda:</b> ángulo entre la dirección de movimiento y el eje mayor según el alargamiento (la línea punteada es lo esperado al azar). <b>Derecha:</b> por película, fracción de observaciones «fluidas» contra rapidez media."))
    if ct is not None:
        g = ct.groupby(["condicion", "t_h"]).agg(q=("q_mediana", "mean"), fl=("frac_fluida", "mean")).reset_index()
        def val(c, i, col):
            x = g[g.condicion == c]
            return x.iloc[i][col] if len(x) > i else np.nan
        s.append(f'<div class="texto"><p><b>En CAMAD la transición se ve en el tiempo, y su ritmo depende del sustrato.</b> Sobre Matrigel y colágeno la mayoría de las células ya es «fluida» en la primera hora ({f(100 * val("Matrigel", 0, "fl"), 0)}% y {f(100 * val("Colágeno I", 0, "fl"), 0)}%) y la fracción sigue subiendo en la segunda ({f(100 * val("Matrigel", 1, "fl"), 0)}% y {f(100 * val("Colágeno I", 1, "fl"), 0)}%). Sobre la matriz confluente de macrófagos la transición es gradual ({f(100 * val("Matriz RAW confluente", 0, "fl"), 0)}% → {f(100 * val("Matriz RAW confluente", 4, "fl"), 0)}% en 5 h). Sobre vidrio casi no ocurre: {f(100 * val("Vidrio", 0, "fl"), 0)}% en la primera hora y ~{f(100 * val("Vidrio", 4, "fl"), 0)}% al final, con un índice de forma mediano de ~3.7 todo el tiempo. Es decir, el sustrato determina si las células cruzan al estado «fluido» y qué tan rápido lo hacen.</p></div>')
        s.append(figura("figuras/morfo_camad_tiempo.png", "<b>CAMAD:</b> distribución de formas de la población en tres momentos (filas) para cada sustrato (columnas). La línea punteada es q = 3.81; abajo a la derecha, la mediana de q."))
    # arquetipos
    s.append('<h3>Arquetipos de forma: agrupar morfologías parecidas y describirlas con una geometría</h3><div class="texto">')
    s.append(f"<p>Para cada célula tomamos su contorno, lo centramos, lo llevamos a un mismo tamaño y lo giramos para que su eje mayor quede horizontal (y su lado más «cargado» siempre del mismo lado). Así, dos células con la misma forma dan el mismo contorno aunque tengan distinto tamaño u orientación. Cada contorno se describe con su radio en 64 direcciones y se descompone en <b>armónicos</b>, que son los números que definen la geometría: el primero mide si un extremo es más ancho (gota), el segundo el alargamiento (elipse o huso), el tercero la triangularidad, y los siguientes las protuberancias. Con esos descriptores agrupamos {len(arq)} arquetipos (mezcla de gaussianas sobre {f(arq.n.sum() / 1000, 0)} mil contornos) y a cada uno le asignamos la geometría simple que mejor lo describe, con reglas explícitas sobre los armónicos.</p>")
    t = arq.copy()
    t["porc"] = t.frac.map(lambda x: f"{f(100 * x, 0)}%")
    t["al"] = t.alargamiento_med.map(lambda x: f(x, 1))
    t["so"] = t.solidez_med.map(lambda x: f(x, 2))
    t["qq"] = t.q_med.map(lambda x: f(x, 2))
    t["v"] = t.rapidez_med.map(lambda x: f(x, 2))
    t["c"] = t.cos_giro_med.map(lambda x: f(x, 2))
    if T is not None:
        TT = T.set_index(T.columns[0])
        t["perm"] = [f(5 / (1 - TT.loc[i, i]), 0) if i in TT.index else "–" for i in t.id]
    s.append("</div>")
    s.append(tabla(t, ["id", "nombre", "porc", "al", "so", "qq", "v", "c", "perm"],
                   ["", "Geometría", "Frecuencia", "Alargamiento", "Solidez", "Índice q", "Rapidez (µm/min)", "⟨cos giro⟩", "Permanencia (min)"],
                   num=("porc", "al", "so", "qq", "v", "c", "perm")))
    s.append(figura("figuras/arq_formas.png", "<b>Los arquetipos de forma</b>, ordenados de más lentos a más rápidos. En azul, la forma media del grupo; en gris, 25 células del grupo; en naranja punteado, la geometría simple propuesta (superelipse, gota o triángulo redondeado ajustados a la forma media)."))
    pf = fr.set_index("medida").p if fr is not None else None
    s.append('<div class="texto">')
    s.append(f"<p><b>Qué muestran.</b> La forma se relaciona con el movimiento de manera consistente en las 16 películas (prueba de Friedman con las películas como réplicas: rapidez p {peq(pf.loc['rapidez'])}, persistencia p {peq(pf.loc['cos_giro'])}). La rapidez crece de las formas redondas a las alargadas: el círculo (F1) es el más lento y los husos muy alargados y las formas ramificadas, los más rápidos. <b>La persistencia, en cambio, no es monótona.</b> Es máxima para las elipses de alargamiento moderado (≈ 2:1) y cae tanto en las redondas como en las muy alargadas o ramificadas. Una interpretación posible es que una célula con varias prolongaciones tiene varios «frentes» que compiten, avanza rápido pero cambia de rumbo a menudo. No encontramos este patrón no monótono descrito explícitamente para MDA-MB-231, aunque la competencia entre protrusiones está descrita en otros tipos celulares; conviene confirmarlo.</p>")
    if T is not None:
        s.append(f"<p><b>Las células cambian de forma, pero entre formas vecinas.</b> La matriz de transiciones muestra que en 5 minutos una célula casi siempre permanece en su arquetipo o pasa a uno parecido (círculo ↔ óvalo ↔ elipse; entre los husos; entre los triángulos). El círculo es el estado más estable: una célula redonda permanece así en promedio {f(5 / (1 - TT.loc['F1', 'F1']), 0)} minutos, contra 7–11 minutos para las demás formas.</p>")
    s.append("<p><b>Vista desde la dirección de movimiento.</b> Si se promedian las formas girándolas según hacia dónde avanza cada célula, las rápidas son claramente más largas en la dirección de avance que las lentas, y el perfil es simétrico entre frente y cola. En promedio no aparece un frente ancho en abanico, sino una forma de huso, que es lo típico de la migración mesenquimal de MDA-MB-231.</p></div>")
    s.append(figura("figuras/arq_movimiento.png", "<b>Izquierda y centro:</b> rapidez y persistencia de cada arquetipo (gris: películas; azul: promedio). <b>Derecha:</b> probabilidad de pasar de un arquetipo (fila) a otro (columna) en 5 minutos."))
    s.append(figura("figuras/arq_frente_cola.png", "<b>Forma media vista desde la dirección de movimiento</b> (la flecha indica hacia dónde avanza) para el 25% más lento y el 25% más rápido de las observaciones, y el perfil del radio desde el frente (0°) hasta la cola (±180°)."))
    s.append(figura("figuras/arq_camad.png", "<b>CAMAD:</b> proporción de cada familia de formas (los 9 arquetipos agrupados en 5) a lo largo de la adhesión. Sobre vidrio y sobre la matriz dispersa dominan las formas redondas; sobre Matrigel y colágeno aparecen en la primera hora las elípticas, triangulares y en huso."))
    return "\n".join(s)


def seccion_relaciones():
    c = csv("descubrimiento", "cribado_correlaciones.csv")
    h = csv("descubrimiento", "hallazgos_pruebas.csv")
    r2 = csv("descubrimiento", "r2_modelos.csv")
    if c is None or h is None:
        return ""
    H = h.set_index("hallazgo")
    rep = c[c.replica & ~c.trivial]
    s = ['<div class="texto">',
         "<p>Además de las preguntas planteadas de antemano, buscamos de forma sistemática relaciones que no estábamos mirando. La ventaja de este dataset es que para cada célula, en cada instante, tenemos a la vez su contorno (brightfield) y su núcleo (fluorescencia). Eso permite cruzar movimiento, forma celular, forma del núcleo, <b>contenido de ADN</b> (que indica en qué fase del ciclo celular está la célula: ~1× en G1, ~2× en G2), la <b>posición del núcleo dentro de la célula</b> y el contacto con vecinas.</p>",
         f"<p><b>Cómo evitamos hallazgos por azar.</b> Con muchas pruebas, alguna sale «significativa» por casualidad. Por eso: (1) buscamos en las 8 películas impares y exigimos que la relación se repita, con el mismo signo, en las 8 pares; (2) usamos las películas como réplicas y controlamos la tasa de falsos descubrimientos; (3) descartamos las relaciones triviales (una variable calculada a partir de la otra). De {len(c)} pares de variables probados, {int(c.replica.sum())} replicaron; {len(rep)} no son triviales, y la mayoría son esperables (por ejemplo, núcleos más grandes en células más grandes). Las que siguen son las que nos parecen más interesantes.</p>",
         "</div>"]
    s.append(figura("figuras/desc_hallazgos.png", "<b>Hallazgos de la búsqueda exploratoria.</b> Cada línea gris une los dos valores de una misma película. H1: probabilidad de dar media vuelta en el paso siguiente según el núcleo esté delante o detrás respecto de la dirección de avance. H2: rapidez de células con 1× y 2× de ADN, sin células redondeadas ni con cromatina muy condensada. H3: rapidez medida con el núcleo, para células libres y en contacto. A la derecha, el brillo del colorante de núcleos a lo largo de la película."))
    s.append('<div class="texto">')
    s.append(f"<p><b>H1. La posición del núcleo anticipa las reversas.</b> Cuando el núcleo está <i>delante</i> del centro de la célula respecto de la dirección en que avanza, la probabilidad de que la célula dé media vuelta en los 5 minutos siguientes es {f(H.loc['H1 núcleo delante -> más reversas', 'mediana_diferencia'] * 100, 0)} puntos porcentuales mayor que cuando está <i>detrás</i> (~38% contra ~26%). Ocurre en las {int(H.loc['H1 núcleo delante -> más reversas', 'peliculas_a_favor'])} películas (p {peq(H.loc['H1 núcleo delante -> más reversas', 'p_wilcoxon'])}). En promedio el núcleo va detrás en el 54–58% de los pasos. Que el núcleo se ubique en la parte trasera de una célula que migra es conocido en fibroblastos (revisión: <a href=\"https://pmc.ncbi.nlm.nih.gov/articles/PMC5995615/\">Nuclear positioning in migrating fibroblasts</a>). Lo que agregamos es su uso <b>predictivo</b>: medido solo con el contorno y un colorante nuclear, sin marcadores de polaridad, anticipa las reversas en células de cáncer de mama. En el modelo que intenta predecir el giro siguiente en películas que no vio, esta variable está entre las más útiles, después de la rapidez y el giro actuales.</p>")
    s.append(f"<p><b>H2. Las células en fase G2 se mueven más despacio.</b> Las células con el doble de ADN son {f(-H.loc['H2 G2 más lentas que G1', 'mediana_diferencia'], 3)} µm/min más lentas que las de G1 (~8%) en {int(H.loc['H2 G2 más lentas que G1', 'peliculas_a_favor'])} de 16 películas (p {peq(H.loc['H2 G2 más lentas que G1', 'p_wilcoxon'])}), y ~{f(H.loc['H2b G2 más grandes que G1', 'mediana_diferencia'], 0)} µm² más grandes. Excluimos las células redondeadas y las de cromatina muy condensada para que el efecto no se deba a células a punto de dividirse, que se redondean y se detienen. <b>Esto contrasta con lo publicado:</b> un estudio con el reportero FUCCI en MDA-MB-231 encontró que G1 es más rápida solo en migración dirigida y no en migración al azar (<a href=\"https://www.biorxiv.org/content/10.1101/288183.full.pdf\">bioRxiv 288183</a>). Nuestros datos son de migración al azar, con 16 réplicas y decenas de miles de observaciones, y sí muestran el efecto. Es el candidato más claro a hallazgo nuevo, aunque nuestra clasificación del ciclo se basa en el contenido de ADN y no en un reportero de ciclo. Habría que confirmarlo con FUCCI.</p>")
    s.append(f"<p><b>H3. El contacto con otra célula no las frena.</b> Las células que están tocando a una vecina se mueven {f(H.loc['H3 en contacto más rápidas (núcleo)', 'mediana_diferencia'], 3)} µm/min <i>más rápido</i> que las libres en {int(H.loc['H3 en contacto más rápidas (núcleo)', 'peliculas_a_favor'])} de 16 películas (p {peq(H.loc['H3 en contacto más rápidas (núcleo)', 'p_wilcoxon'])}), midiendo la rapidez con el núcleo para que la deformación del contorno al tocarse no la infle. Las células normales suelen frenar y cambiar de rumbo al chocar («inhibición de la locomoción por contacto»); que las MDA-MB-231 la tienen debilitada ya está reportado. Es una asociación: también puede ser que las células más rápidas simplemente se topen con más vecinas.</p>")
    s.append(f"<p><b>Un hallazgo técnico útil.</b> El brillo del colorante de núcleos baja ~{f(-100 * H.loc['T1 brillo SiR-DNA a las 6-8 h / 0-2 h (fotoblanqueo)', 'mediana_diferencia'], 0)}% en 8 horas en todas las películas (fotoblanqueo). Cualquier medida basada en intensidad, como el contenido de ADN, debe normalizarse imagen por imagen, como hicimos aquí. La rapidez de las células, en cambio, no disminuye a lo largo de la película, así que no hay signos de daño por la iluminación.</p>")
    if r2 is not None:
        rr = r2.set_index(r2.columns[0]).iloc[:, 0]
        s.append(f"<p><b>¿Cuánto se puede predecir?</b> Un modelo de aprendizaje automático (gradient boosting) entrenado con todas estas variables y evaluado en películas que no vio explica el {f(100 * rr.loc['rapidez_sig'], 0)}% de la variación en la rapidez del paso siguiente y el {f(100 * rr.loc['cos_giro_sig'], 0)}% del giro. Es poco, y eso también es un resultado: a escala de 5 minutos el movimiento de cada célula es en gran parte impredecible (estocástico), y las relaciones anteriores son tendencias estadísticas robustas, no reglas deterministas.</p>")
    s.append("</div>")
    s.append(figura("figuras/desc_importancia.png", "<b>Variables que más ayudan a predecir el giro del paso siguiente</b>, en películas que el modelo no vio durante el entrenamiento."))
    if len(rep):
        t = rep.head(25).copy()
        nombres = {**NOMBRE, "cos_eje": "movimiento a lo largo del eje mayor", "q": "índice de forma", "nuc_area": "área del núcleo",
                   "nuc_alargamiento": "alargamiento del núcleo", "nuc_solidez": "solidez del núcleo", "nuc_intensidad": "brillo del núcleo",
                   "adn_rel": "contenido de ADN", "off_rel": "descentrado del núcleo", "nuc_frac_area": "núcleo / célula (área)",
                   "cos_nucleo_mov": "núcleo respecto del avance", "t_h": "tiempo en la película", "dist_vecina_um": "distancia a la vecina",
                   "densidad_local": "densidad local", "rapidez": "rapidez", "cos_giro": "⟨cos giro⟩", "alargamiento": "alargamiento",
                   "solidez": "solidez", "area_um2": "área"}
        t["A"] = t.a.map(nombres).fillna(t.a)
        t["B"] = t.b.map(nombres).fillna(t.b)
        t["r1"] = t.rho_desc.map(lambda x: f(x, 2))
        t["r2"] = t.rho_val.map(lambda x: f(x, 2))
        s.append("<details><summary><b>Las 25 relaciones no triviales más fuertes que replicaron</b> (correlación de Spearman mediana dentro de las películas de descubrimiento y de validación)</summary>")
        s.append(tabla(t, ["A", "B", "r1", "r2"], ["Variable", "Variable", "ρ descubrimiento", "ρ validación"], num=("r1", "r2")))
        s.append("</details>")
    return "\n".join(s)


def seccion_linajes():
    r = csv("linajes", "resumen.csv")
    c = csv("linajes", "control.csv")
    if r is None:
        return ""
    ndiv = int(r.divisiones.sum())
    horas = r.horas_celula.sum()
    esperadas = horas * np.log(2) / 30
    s = ['<div class="texto">',
         "<p>El programa puede ponerle a cada célula una <b>etiqueta</b> que se conserva mientras se la sigue y, si se divide, pasarles a sus hijas una etiqueta derivada: las hijas de la célula 12 son la 12.1 y la 12.2, las de la 12.1 serían 12.1.1 y 12.1.2, y así se arma el <b>árbol genealógico</b>. Lo difícil es detectar con certeza <i>cuándo</i> una célula se divide.</p>",
         "<p>Para eso usamos el canal de núcleos. En brightfield una célula que se divide se redondea y sus hijas quedan pegadas, así que no se distingue de dos vecinas. Regla usada: aparecen dos núcleos junto a donde estaba el núcleo madre; cada uno tiene aproximadamente la mitad de su área, porque el material genético se reparte; ambos siguen existiendo por separado al menos 30 minutos y se alejan entre sí; y la madre era un núcleo compacto, no uno lobulado. Además, una célula recién nacida no puede volver a dividirse en menos de 15 horas, porque el ciclo celular de MDA-MB-231 dura más de 20 h.</p>",
         "</div>"]
    s.append(video("videos/linajes.mp4", "<b>Etiquetas y linajes en el video con más divisiones.</b> Izquierda, núcleos; derecha, brightfield. Cada punto es una célula; en color, las que pertenecen a un linaje con una división detectada, con su etiqueta. Las dos hijas comparten el color de su madre."))
    s.append(figura("figuras/linaje_mosaico_divisiones.png", "<b>Doce divisiones detectadas, elegidas al azar</b>, de 10 minutos antes a 10 minutos después. Las divisiones reales se reconocen por el núcleo muy brillante y compacto (cromatina condensada) que se separa en dos núcleos pequeños. Las detecciones falsas suelen ser núcleos tenues o fuera de foco."))
    s.append(figura("figuras/linaje_arboles.png", "<b>Ejemplos de árboles genealógicos.</b> El eje vertical es el tiempo; cada línea es una célula y cada bifurcación una división. En 8.3 horas cada linaje tiene como máximo una división; con videos más largos los árboles tendrían varios niveles."))
    s.append('<div class="texto">')
    s.append(f"<p><b>Qué tan bien funciona.</b> Se detectaron {ndiv} divisiones en {f(horas / 1000, 1)} mil horas-célula de seguimiento. Si las MDA-MB-231 se duplican cada ~30 horas, se esperarían unas {f(esperadas, 0)}, así que el detector encuentra solo una fracción de las divisiones (es deliberadamente conservador). En la inspección visual de ejemplos al azar, aproximadamente la mitad de las detecciones son divisiones claras.")
    if c is not None and len(c):
        s.append(f" Antes de aplicar la restricción del ciclo celular aparecían {int(c.segunda_generacion_imposibles.iloc[0])} «divisiones» de células recién nacidas, que son imposibles y confirman que parte de las detecciones son errores.")
    s.append("</p><p><b>Conclusión:</b> el etiquetado y la construcción de árboles funcionan; la detección automática de mitosis todavía es un prototipo. Para que sea confiable habría que entrenar un clasificador de mitosis con ejemplos etiquetados a mano (unas pocas centenas bastan), filmar con más frecuencia (la división dura 30–60 minutos) o usar un marcador de ciclo celular. Con eso se podrían medir, por ejemplo, si las células hermanas se mueven de forma parecida o si la división cambia la persistencia.</p></div>")
    return "\n".join(s)


def seccion_agresividad(vbf):
    """Qué dicen (y qué no) estos datos sobre la hipótesis de que el movimiento
    distingue células más y menos agresivas, y cuántas réplicas harían falta."""
    from statsmodels.stats.power import TTestIndPower
    pe = csv("bf", "estadisticas", vbf, "por_pelicula.csv")
    s = ['<div class="texto">',
         "<p>La hipótesis de fondo del proyecto es que una célula de cáncer más agresiva se puede distinguir de una menos agresiva, o de una no tumoral, analizando cómo se mueve. Los resultados son <b>compatibles con esa idea y muestran que es factible medirla</b>, pero todavía no la demuestran. Para demostrarla hay que comparar líneas de distinta agresividad filmadas en las <i>mismas</i> condiciones: mismo sustrato, medio, microscopio e intervalo entre fotos. Por ejemplo MDA-MB-231 (muy invasiva), MCF7 (tumoral, poco invasiva) y MCF10A (epitelio mamario no tumoral).</p>",
         "<p><b>Qué no permiten estos datos.</b> En brightfield y en CAMAD solo hay MDA-MB-231. WHAD incluye MCF10A y MCF7, pero es un ensayo de cierre de herida colectivo, con otro microscopio y muy pocas posiciones. Comparar entre datasets mezclaría el efecto de la línea celular con todo lo demás: las mismas MDA-MB-231 se mueven más rápido en plástico con suero que en las primeras horas sin suero de CAMAD.</p>",
         "<p><b>Qué sí muestran.</b> (1) El análisis mide con precisión conocida, validada contra núcleos y contra anotación manual. (2) Algunas medidas varían muy poco entre campos de la misma condición, y eso las hace buenas candidatas para distinguir condiciones. En particular la EAD₁ corregida, que varía apenas unas milésimas entre videos (tabla). (3) Cuando en un mismo campo conviven tipos celulares distintos, sus firmas de movimiento difieren mucho: en CAMAD exp4 hay muchas células pequeñas y redondas (probablemente macrófagos) que casi no se desplazan, mientras las MDA-MB-231 avanzan varias veces más rápido. (4) En WHAD, las MCF7 con SEMA6D sobreexpresado, un fenotipo más migratorio, desprenden más células; va en la dirección esperada, aunque es una sola posición.</p>",
         "</div>"]
    if pe is not None:
        pw = TTestIndPower()
        filas = []
        for m, nombre, difs, rel in (("rapidez_um_min", "Rapidez", (0.3, 0.5), True),
                                     ("ead1_ens_corr", "EAD₁ corregida", (0.01, 0.02), False),
                                     ("cos_giro_medio", "Coseno medio del giro", (0.05, 0.10), False),
                                     ("prw_P_min", "Persistencia P", (0.3, 0.5), True),
                                     ("aspect_ratio", "Alargamiento", (0.1, 0.2), True)):
            mu, sd = pe[m].mean(), pe[m].std()
            celdas = []
            for d in difs:
                delta = d * mu if rel else d
                try:
                    n = pw.solve_power(effect_size=delta / sd, alpha=0.05, power=0.8)
                    n = 2 if not np.isfinite(n) else int(np.ceil(max(n, 2)))
                except Exception:
                    n = 2
                celdas.append((f"{int(d * 100)}%" if rel else f"{d:g}", n))
            filas.append({"m": nombre, "v": f"{f(mu, 3)} ± {f(sd, 3)}",
                          "a": f"{celdas[0][0]}: {celdas[0][1]}", "b": f"{celdas[1][0]}: {celdas[1][1]}"})
        s.append('<div class="texto"><p>¿Cuántas réplicas biológicas (videos o experimentos independientes) por línea celular harían falta para detectar una diferencia, con 80% de probabilidad y α = 0.05? Cálculo con la variación entre videos medida en brightfield:</p></div>')
        s.append(tabla(pd.DataFrame(filas), ["m", "v", "a", "b"], ["Medida", "Media ± desvío entre videos",
                                                               "Diferencia: réplicas por grupo", "Diferencia: réplicas por grupo"],
                       num=("v", "a", "b")))
    s.append('<div class="texto"><p><b>Diseño propuesto para probar la hipótesis.</b> Filmar MDA-MB-231, MCF7 y MCF10A en el mismo experimento, sobre el mismo sustrato, con una imagen cada 1–2 minutos (para que el error de posición no tape la persistencia), un marcador nuclear y al menos 5 réplicas biológicas por línea. Después, combinar varias medidas de cada célula (rapidez, EAD, SE, acoplamiento rapidez–persistencia, forma) en un clasificador y evaluarlo dejando afuera réplicas completas. Así el acierto no se infla por aprender las particularidades de un video. Un enfoque similar, basado en la forma de células individuales, ya permitió predecir el potencial metastásico de líneas de cáncer (Wu et al., <i>Science Advances</i> 2020).</p></div>')
    return "\n".join(s)


def seccion_rendimiento():
    r = js("rendimiento", "rendimiento.json")
    if not r:
        return ""
    t = {x["imagen"]: x for x in r["tabla"]}
    bf, sm = t["1024x1022 (BF)"], t["512x511"]
    ca = t["642x478 (CAMAD 4x)"]
    M = r["medido"]
    seg = lambda x: f"{f(x, 1)} s" if x < 90 else (f"{f(x / 60, 1)} min" if x < 5400 else f"{f(x / 3600, 1)} h")
    s = ['<div class="texto">']
    s.append(f"<p>Casi todo el tiempo se va en la segmentación con Cellpose-SAM, que es un modelo grande (un <i>vision transformer</i> de unos 300 millones de parámetros). Contamos las operaciones que realiza: {f(M['gflop_por_tile_256'], 0)} mil millones de operaciones (GFLOP) por cada bloque de 256 × 256 píxeles, es decir, unos <b>{f(bf['tflop_por_imagen'], 1)} billones de operaciones (TFLOP) por imagen</b> de 1024 × 1022 píxeles. En comparación, detectar el contorno de cada célula, unirla con la imagen anterior y medirla tarda {f(bf['otros_pasos_pc_s'], 2)} s por imagen, y las estadísticas de un video completo unos {f(M['estadistica_s_por_pelicula'], 0)} s.</p>")
    s.append("</div>")
    filas = []
    for nombre, x, n in (("Imagen brightfield (1024 × 1022)", bf, 1600), ("Imagen CAMAD reducida (642 × 478)", ca, 9438),
                         ("Imagen reducida a 512 × 511", sm, None)):
        filas.append({"a": nombre, "g": seg(x["pc_gpu_fp16_s"] + x["otros_pasos_pc_s"]),
                      "c": seg(x["pc_cpu_4hilos_s"] + x["otros_pasos_pc_s"]),
                      "p": f"{seg(x['pi5_s'])} ({seg(x['pi5_s_min'])} – {seg(x['pi5_s_max'])})",
                      "tot": (f"{seg(n * (x['pc_gpu_fp16_s'] + x['otros_pasos_pc_s']))} / {seg(n * (x['pi5_s']))}" if n else "–")})
    s.append(tabla(pd.DataFrame(filas), ["a", "g", "c", "p", "tot"],
                   ["Por imagen", "PC con GPU (RX 6800 XT)", "PC solo CPU (4 hilos)", "Raspberry Pi 5, 8 GB (estimado)",
                    "Dataset completo: PC GPU / Pi 5"], num=("g", "c", "p", "tot")))
    s.append(figura("figuras/rendimiento.png",
                    "<b>Tiempo por imagen contra el intervalo entre fotos</b> (escala logarítmica). Para que el análisis pueda hacerse durante la adquisición, la barra tiene que quedar a la izquierda de la línea del intervalo correspondiente. Las barras de la Pi 5 muestran el rango de la estimación."))
    s.append('<div class="texto">')
    s.append(f"<p><b>Cómo se estimó la Raspberry Pi 5.</b> No la medimos directamente. Su procesador (4 núcleos Cortex-A76 a 2.4 GHz) rinde 30.2 GFLOPS en el benchmark Linpack (medición publicada por J. Geerling). En nuestra PC, Cellpose aprovecha el {f(100 * r['eficiencia_cellpose_vs_matmul'], 0)}% de la capacidad de cálculo pura del procesador; aplicando la misma proporción a la Pi quedan unos {f(r['pi5_gflops_efectivos']['central'], 0)} GFLOPS efectivos (rango 12–25). El modelo cabe en los 8 GB de memoria (ocupa 1.2 GB), pero la Pi necesita un disipador activo para sostener esa carga sin bajar su frecuencia.</p>")
    s.append(f"<p><b>¿Se puede analizar en la Pi entre foto y foto?</b> Con Cellpose-SAM, <b>no</b>. Una imagen de 1 megapíxel tardaría unos {seg(bf['pi5_s'])}, más que el intervalo de 5 minutos del dataset brightfield y mucho más que los 30 segundos de CAMAD. Aun reduciendo la imagen a 512 × 512, harían falta unos {seg(sm['pi5_s'])} por imagen. Los pasos posteriores (contornos, seguimiento, estadística) sí serían viables en la Pi, porque tardan menos de un segundo por imagen.</p>")
    s.append(f"<p><b>Alternativas factibles:</b></p><ul>"
             f"<li><b>Que la Pi solo adquiera y envíe cada imagen a la PC por la red.</b> Una imagen de 2 MB tarda una fracción de segundo por cable, y la PC la procesa en {seg(bf['pc_gpu_fp16_s'] + bf['otros_pasos_pc_s'])}, así que el análisis podría ir en tiempo real incluso con intervalos de 30 s. Es la opción recomendada.</li>"
             f"<li><b>Procesar todo al terminar el experimento</b> en la PC: los 1600 cuadros de brightfield tardan unos {seg(1600 * (bf['pc_gpu_fp16_s'] + bf['otros_pasos_pc_s']))} con la GPU. En la Pi tardarían unos {f(1600 * bf['pi5_s'] / 86400, 0)} días.</li>"
             f"<li><b>Usar un modelo más liviano en la Pi</b>, como Cellpose 3, que hace unas 15–20 veces menos operaciones por píxel (sección 3). En esta PC es unas 5 veces más rápido en CPU, así que en la Pi 5 tardaría del orden de 3–5 minutos por imagen: justo en el límite para fotos cada 5 minutos e inviable cada 30 s. Además segmenta peor en nuestras imágenes (F1 0.81 contra 0.86). Un acelerador de IA para la Pi 5 (AI HAT+) podría bajar ese tiempo a segundos, pero habría que convertir el modelo a su formato y volver a validarlo.</li></ul>")
    s.append("</div>")
    return "\n".join(s)


LIMITACIONES = """
<ul>
<li><b>Una sola condición en el dataset principal.</b> Los 16 videos brightfield son de la misma línea celular y la misma condición, así que no permiten comparar tratamientos. Podrían provenir del mismo día o de la misma placa. Verificamos que no son continuaciones unos de otros.</li>
<li><b>Pocos experimentos por sustrato en CAMAD</b> (2 a 4). Alcanzan para describir tendencias, no para afirmar diferencias con seguridad.</li>
<li><b>Error de posición.</b> El centro calculado desde el contorno de toda la célula tiembla alrededor de 2 µm entre imágenes. Con pasos de 5 minutos eso oculta buena parte de la persistencia real. Las medidas con núcleos, o con pasos más largos, son más sensibles.</li>
<li><b>Muestreo de 5 minutos en brightfield.</b> La persistencia de estas células dura unos pocos pasos. Para medirla bien harían falta imágenes cada 1 o 2 minutos.</li>
<li><b>Referencias imperfectas.</b> La segmentación de núcleos también comete errores (núcleos en división o poco teñidos), y en CAMAD solo están dibujadas las células aisladas. Las métricas de validación son estimaciones, no verdades absolutas.</li>
<li><b>Detección de mitosis todavía en prototipo</b> (sección 8): detecta una fracción de las divisiones y cerca de la mitad de sus detecciones son reales. Cuando una división no se detecta, la trayectoria se corta o una de las hijas continúa la de la madre.</li>
<li><b>Identidad de exp8 y exp9 de CAMAD</b> sin confirmar, por eso se excluyeron de la comparación.</li>
</ul>
"""

PROXIMOS = """
<ul>
<li>Confirmar con el reportero de ciclo celular FUCCI que las células en G2 migran más despacio también en migración al azar (hallazgo H2), y probar si la posición del núcleo anticipa las reversas en otras líneas (H1).</li>
<li>Convertir el detector de divisiones en un clasificador de mitosis entrenado con unas pocas centenas de ejemplos etiquetados a mano sobre el canal de núcleos, para tener árboles genealógicos confiables y estudiar si las células hermanas se mueven parecido.</li>
<li>Reducir el error de posición con un suavizado de trayectorias basado en el propio modelo de caminata persistente (filtro de Kalman), y medir cuánto mejora la sensibilidad de la EAD.</li>
<li>Pedir a los autores de CAMAD la planilla de experimentos para confirmar qué células se filmaron en exp8 y exp9 y el orden real de los frames con relleno negro.</li>
<li>Para experimentos propios: filmar cada 1–2 minutos, con al menos 5 réplicas biológicas por condición, y agregar un marcador nuclear. Con eso tanto la entropía como las comparaciones entre condiciones ganarían mucha potencia.</li>
<li>Probar el efecto de la densidad de forma controlada, sembrando a distintas densidades, ya que entre videos parece influir en la coordinación entre vecinas.</li>
</ul>
"""


def seccion_tecnico(vbf, vca):
    s = ['<div class="texto">']
    s.append("<p>Todo el análisis se regenera con <span class=\"mono\">bash scripts/correr_v2.sh</span> (unas 3–4 horas con la GPU RX 6800 XT). El detalle de cada cambio y su justificación está en <span class=\"mono\">CAMBIOS_v2.md</span> y en el encabezado de cada script. Los scripts de la versión anterior quedaron sin tocar.</p>")
    s.append(f"<p>Variantes de seguimiento usadas: brightfield <span class=\"mono\">{vbf}</span>, núcleos <span class=\"mono\">dist</span>, CAMAD <span class=\"mono\">{vca}</span>.</p></div>")
    cambios = [
        ("Segmentación", "Cellpose-SAM en precisión fp16 y guardando los flujos", "La GPU no acelera bf16: 4.1 → 1.7 s por imagen, con las mismas máscaras (AP ≥ 0.995). Con los flujos guardados los umbrales se ajustan sin recalcular la red."),
        ("Segmentación BF", "Umbrales elegidos contra los núcleos", "Antes no había ninguna referencia para brightfield."),
        ("Segmentación CAMAD", "Imagen reducida 4× y referencia desde los polígonos .roi", "A resolución completa se perdía la mitad de las células; la versión anterior fusionaba células vecinas en la referencia."),
        ("Detecciones", "Área mínima (50 µm² BF, 20 µm² núcleos, 60 µm² CAMAD)", "Objetos más chicos no pueden ser células y solo agregan trayectorias de ruido."),
        ("Seguimiento", "Distancias en µm, penalización por cambio de tamaño, variantes comparadas contra núcleos", "Elegir el método con una referencia y no a ojo."),
        ("Modelo persistente", "Término de error de posición 4σ²", "Sin él, la persistencia sale artificialmente corta. Explica los 4 ajustes degenerados de la versión anterior."),
        ("Direccionalidad", "Ventana fija de 1 h", "El cociente distancia neta / recorrido baja solo con la duración."),
        ("Entropía", "SE y EAD (Liu 2021, 2024) con corrección por tamaño de muestra", "Sin corregir, la EAD de una célula depende de la longitud de su trayectoria."),
        ("Paso de análisis", "5 min en ambos datasets, más sensibilidad al paso", "La EAD depende del intervalo; usar el mismo hace comparables los datasets."),
        ("Estadística", "Videos como réplicas, bootstrap, ANOVA por permutación, modelo mixto y Holm", "Evitar tratar miles de células como si fueran independientes."),
    ]
    s.append(tabla(pd.DataFrame(cambios, columns=["a", "b", "c"]), ["a", "b", "c"], ["Paso", "Cambio", "Por qué"]))
    rel = csv("inferencia", "bf", "relacion_densidad.csv")
    if rel is not None:
        t = rel.copy()
        t["m"] = t.metrica.map(NOMBRE)
        t["r"] = t.rho_vs_densidad.map(lambda x: f(x, 2))
        t["p_"] = t.p.map(fp)
        t["ph"] = t.p_holm.map(fp)
        s.append('<h3>Relación con la densidad (brightfield, 16 videos)</h3>')
        s.append(tabla(t, ["m", "r", "p_", "ph"], ["Medida", "ρ de Spearman", "p", "p Holm"], num=("r", "p_", "ph")))
    s.append('<div class="texto"><h3>Archivos</h3><ul>'
             '<li><span class="mono">resultados/v2/&lt;dataset&gt;/estadisticas/&lt;variante&gt;/por_pelicula.csv</span>: una fila por video o experimento, con todas las medidas.</li>'
             '<li><span class="mono">por_celula.csv</span>: una fila por trayectoria.</li>'
             '<li><span class="mono">resultados/v2/inferencia/</span>: todas las pruebas estadísticas.</li>'
             '<li><span class="mono">resultados/v2/validacion_*</span>: validación de segmentación y seguimiento.</li>'
             '<li><span class="mono">~/microscopio_cache/</span>: máscaras, flujos y cuadros de CAMAD (pesados, fuera de Syncthing, regenerables).</li>'
             '</ul></div>')
    return "\n".join(s)


def hallazgos(vbf, vca):
    """Lista de lo más importante, armada con los números."""
    h = []
    el = js("validacion_seg_bf", "eleccion.json")
    ea = camad_aplicada()
    g = csv("validacion_seg_camad", "grilla.csv")
    r = csv("validacion_tracking", "por_pelicula.csv")
    et = js("validacion_tracking", "eleccion.json")
    if el and r is not None and et:
        m = r.groupby("variante").precision_eslabon.mean()
        h.append(f"<b>Las mediciones ahora están validadas.</b> Comparando con los núcleos fluorescentes del mismo campo, la detección de células en brightfield acierta con F1 = {f(el['metricas_loeo_media']['f1'], 2)} y el {f(100 * m[et['elegida']], 1)}% de las uniones entre imágenes del seguimiento son correctas.")
    if g is not None and ea:
        full = g[g.escala == 1].groupby("exp").recall.mean().mean()
        h.append(f"<b>La segmentación de CAMAD mejoró de {f(100 * full, 1)}% a {f(100 * ea['recall'], 1)}%</b> de células encontradas, al trabajar las imágenes a una escala adecuada para el modelo.")
    rb = fila_resumen("bf", "prw_P_min")
    if rb is not None:
        pe_ = csv("bf", "estadisticas", vbf, "por_pelicula.csv")
        h.append(f"<b>Las MDA-MB-231 son rápidas pero poco persistentes:</b> avanzan a {mic('bf', 'rapidez_um_min', 2, 'µm/min')} y mantienen la dirección unos {f(rb.media, 0)} minutos. Son superdifusivas a escala corta en las 16 películas (α de {f(pe_.alfa_corto.min())} a {f(pe_.alfa_corto.max())}) y tienden a una caminata al azar a escala de horas.")
    e1 = fila_resumen("bf", "ead1_ens_corr")
    se = fila_resumen("bf", "se_ventana_corr")
    if e1 is not None and se is not None:
        h.append(f"<b>Las dos entropías de Shannon confirman una persistencia débil pero real:</b> EAD₁ corregida = {f(e1.media, 3)} y SE corregida = {f(se.media, 3)} (1 sería movimiento completamente al azar), ambas significativamente menores que 1.")
    h.append("<b>Aporte metodológico:</b> sin corrección, la entropía de una célula depende de la longitud de su trayectoria. Lo corregimos, y las simulaciones muestran que el error de posición limita cuánta persistencia se puede detectar con imágenes cada 5 minutos.")
    t = csv("inferencia", "bf_vs_nucleos.csv")
    if t is not None:
        t = t.set_index("metrica")
        cos_b, cos_n = t.loc["cos_giro_medio", "media_bf"], t.loc["cos_giro_medio", "media_nucleos"]
        h.append(f"<b>Medir células o núcleos da la misma imagen general, con una diferencia instructiva.</b> La rapidez coincide bien entre ambos métodos (concordancia {f(t.loc['rapidez_um_min', 'ccc_lin'], 2)}). En cambio, con los núcleos, cuya posición «tiembla» menos, se detecta casi el doble de persistencia entre pasos (coseno medio del giro {f(cos_n, 2)} contra {f(cos_b, 2)}), y aparece una correlación débil pero estadísticamente significativa entre la persistencia de células vecinas, que con brightfield no llega a verse.")
    fm = csv("inferencia", "bf", "forma_movimiento.csv")
    if fm is not None:
        r = fm.set_index(["forma", "movimiento"]).loc[("aspect_ratio", "rapidez_um_min")]
        h.append(f"<b>La forma predice parte del movimiento:</b> dentro de cada película, las células más alargadas se mueven más rápido y con más persistencia (ρ = {f(r.rho_mediana_dentro)} entre alargamiento y rapidez, p Holm {peq(r.p_holm)}), la firma de la polaridad frente–atrás. Se confirma con los núcleos.")
    arq_ = csv("arquetipos", "arquetipos.csv")
    if arq_ is not None:
        h.append(f"<b>Mapa de formas y arquetipos.</b> La rapidez salta cuando la célula pasa de compacta a alargada, cerca del índice de forma 3.8 que marca la transición «sólido–fluido» en la literatura. La forma se adelanta al movimiento hasta 30 min. Las células se agrupan en {len(arq_)} arquetipos geométricos (círculo, óvalo, elipse, triángulos, husos, ramificada), y la persistencia es máxima en las elipses moderadas, no en las más alargadas.")
    hp = csv("descubrimiento", "hallazgos_pruebas.csv")
    if hp is not None:
        h.append("<b>Relaciones nuevas en los datos:</b> el núcleo ubicado delante anticipa que la célula dará media vuelta (16 de 16 películas); las células en G2 (el doble de ADN) son ~8% más lentas, lo que contrasta con un estudio previo, y es el candidato más claro a hallazgo nuevo; y el contacto con vecinas no las frena.")
    pc = csv("camad", "estadisticas", vca, "por_pelicula.csv")
    if pc is not None:
        a = pc.groupby("condicion").area_um2.mean()
        h.append(f"<b>El sustrato cambia la forma de la célula en las primeras horas.</b> Sobre Matrigel y colágeno las MDA-MB-231 se extienden (área media ~{f(a.get('Matrigel'), 0)} y ~{f(a.get('Colágeno I'), 0)} µm²), mientras que sobre vidrio sin recubrimiento siguen redondas (~{f(a.get('Vidrio'), 0)} µm²), en todos los experimentos (p exacta = 0.036). Sobre vidrio, además, las células no migran: su MSD se aplana (confinamiento), mientras que sobre Matrigel y colágeno el α a escala larga es 1.0–1.8. Las diferencias de rapidez y de persistencia por entropía entre sustratos son tendencias que requieren más experimentos.")
    r_l = csv("linajes", "resumen.csv")
    if r_l is not None:
        h.append(f"<b>Etiquetas y árboles genealógicos (prototipo).</b> Cada célula recibe una etiqueta que pasa a sus hijas (12 → 12.1 y 12.2). Se detectaron {int(r_l.divisiones.sum())} divisiones con el canal de núcleos; la mitad de las revisadas a ojo son reales, así que para usarlo en serio falta un clasificador de mitosis.")
    rr = js("rendimiento", "rendimiento.json")
    if rr:
        bf = [x for x in rr["tabla"] if x["imagen"].startswith("1024")][0]
        h.append(f"<b>Tiempo de cálculo:</b> ~{f(bf['pc_gpu_fp16_s'] + bf['otros_pasos_pc_s'], 1)} s por imagen en la PC con GPU; en una Raspberry Pi 5 serían ~{f(bf['pi5_s'] / 60, 0)} min. No alcanza para analizar entre foto y foto, así que la Pi debería solo adquirir y enviar las imágenes a la PC.")
    return "\n".join(f"    <li>{x}</li>" for x in h)


def main():
    vbf, vca = var("bf") or "dist_tam", var("camad") or "dist_tam"
    copiar_recursos()
    fotos_datos()
    hoy = datetime.date.today()
    fecha = f"{hoy.day} de {MESES[hoy.month - 1]} de {hoy.year}"
    tpl = (Path(__file__).resolve().parent / "plantillas" / "reporte.html").read_text()
    rep = {
        "FECHA": fecha, "HALLAZGOS": hallazgos(vbf, vca), "TARJETAS_DATOS": tarjetas_datos(),
        "CELLPOSE": seccion_cellpose(),
        "VALIDACION": seccion_validacion(), "SIMULACIONES": seccion_simulaciones(),
        "RESULTADOS_BF": seccion_bf(vbf), "BF_VS_NUC": seccion_bf_vs_nuc(), "LINAJES": seccion_linajes(),
        "FORMAS": seccion_formas(), "RELACIONES": seccion_relaciones(), "RESULTADOS_CAMAD": seccion_camad(vca),
        "RESULTADOS_WHAD": seccion_whad(), "AGRESIVIDAD": seccion_agresividad(vbf), "RENDIMIENTO": seccion_rendimiento(), "LIMITACIONES": LIMITACIONES, "PROXIMOS": PROXIMOS,
        "TECNICO": seccion_tecnico(vbf, vca), "REPLICAS": seccion_tablas_replica(vbf, vca), "USOS": seccion_usos(vbf),
    }
    for k, v in rep.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    faltan = [x for x in rep if "{{" + x + "}}" in tpl]
    (OUT / "index.html").write_text(tpl)
    tam = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"index.html listo ({tam:.1f} MB en total con recursos). Marcadores sin reemplazar: {faltan}")


if __name__ == "__main__":
    main()
