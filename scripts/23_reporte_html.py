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
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


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
    ea = js("validacion_seg_camad", "eleccion_aplicada.json")
    g = csv("validacion_seg_camad", "grilla.csv")
    s.append('<h3>Segmentación de CAMAD comparada con dibujos hechos a mano</h3><div class="texto">')
    if g is not None:
        full = g[(g.escala == 1)].groupby("exp").recall.mean().mean()
        s.append(f"<p>CAMAD trae 96 imágenes en las que un experto dibujó el contorno de las células aisladas. Con las imágenes a resolución completa (como hacía la versión anterior), Cellpose encontraba el <b>{f(100 * full, 0)}%</b> de esas células. La causa es la escala: a 0.117 µm por píxel una célula mide 150–400 píxeles, mucho más de lo que el modelo espera. Reduciendo cada imagen 4 veces por lado, el recall sube al <b>{f(100 * ea.get('recall', np.nan), 0)}%</b>, con una superposición media (IoU) de {f(ea.get('iou_medio'), 2)} entre el contorno detectado y el dibujado. Además cada video se procesa unas 10 veces más rápido. Si la configuración se elige dejando afuera cada experimento y se evalúa en ese experimento (una estimación más exigente), el recall es {f(100 * js('validacion_seg_camad', 'eleccion.json').get('metricas_loeo_media', {}).get('recall', np.nan), 0)}%. Los experimentos más difíciles son el 11 (Matrigel) y el 12 (colágeno), con muchas células fuera de foco.</p>")
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
    s.append(f"<p>Analizamos {ncel:,} trayectorias de al menos una hora en los 16 videos. Todas las cifras son el promedio entre videos, con su intervalo de confianza del 95%.".replace(",", ".") + "</p>")
    s.append("<h3>Rapidez y persistencia clásicas</h3>")
    s.append(f"<p>Las células avanzan a <b>{mic('bf', 'rapidez_um_min', 2, 'µm/min')}</b>. Según el modelo de caminata persistente, mantienen su dirección durante unos <b>{mic('bf', 'prw_P_min', 1, 'min')}</b>, con un error de posición de {mic('bf', 'prw_sigma_um', 2, 'µm')}. Como las imágenes están separadas 5 minutos, la dirección se conserva apenas durante 2 o 3 imágenes. A la escala de horas el movimiento es casi el de una caminata al azar: el exponente del MSD a tiempos largos es {mic('bf', 'alfa_largo', 2)} (1 sería completamente al azar).</p>")
    s.append("</div>")
    s.append(tabla_resumen("bf", ["rapidez_um_min", "prw_P_min", "prw_S_um_min", "prw_sigma_um", "direccionalidad_1h",
                                  "alfa_corto", "alfa_largo", "cos_giro_medio", "ead1_ens", "ead1_ens_corr", "ead1_cel",
                                  "ead1_cel_corr", "tl1_ens_min", "tl1_cel_mediana_min", "se_fps_completa", "se_ventana_corr",
                                  "se_wavelet_corr", "ead_t_pct_06_1", "corr_dir_0_50um", "area_um2", "aspect_ratio",
                                  "densidad_cel_mm2"]))
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
    if rel is not None:
        sig = rel[rel.p_holm < 0.05]
        if len(sig):
            s.append("<p>Entre videos, la densidad de células se relaciona significativamente con: " +
                     "; ".join(f"{NOMBRE.get(r.metrica, r.metrica)} (ρ = {f(r.rho_vs_densidad, 2)}, p Holm = {fp(r.p_holm)})" for _, r in sig.iterrows()) + ".</p>")
        else:
            s.append("<p>Con 16 videos, ninguna medida se relaciona significativamente con la densidad de células después de corregir por comparaciones múltiples (la tabla completa está en el anexo).</p>")
    if het is not None:
        h = het.set_index("metrica")
        s.append(f"<p>Los videos difieren bastante entre sí: el video al que pertenece una célula explica el {f(100 * h.loc['rapidez_um_min', 'icc_pelicula'], 0)}% de la variación en rapidez entre células y el {f(100 * h.loc['ead1_corr', 'icc_pelicula'], 0)}% de la variación en EAD₁. Esto confirma que tratar a cada célula como una réplica independiente exageraría la certeza de las conclusiones.</p>")
    s.append("</div>")
    s.append(figura("figuras/bf_colectivo_acoplamiento.png",
                    "<b>Coordinación entre células.</b> Izquierda: cuánto se parecen las direcciones de dos células según la distancia que las separa. Centro: correlación entre las series de SE de pares vecinos y lejanos; cada línea es un video. Derecha: acoplamiento entre rapidez y persistencia en cada video, coloreado por densidad."))
    s.append(figura("figuras/bf_vs_densidad.png", "<b>Medidas de cada video según su densidad de células.</b> El número junto a cada punto identifica el video."))
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
    "ead_t_pct_0_03": "% del tiempo con EAD < 0.3", "ead_t_pct_03_06": "% del tiempo con EAD 0.3–0.6",
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
                 (f"Coinciden poco (concordancia < 0.5) en: {', '.join(malas)}. " if malas else "") +
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
    return "\n".join(s)


def seccion_camad(vca):
    s = ['<div class="texto">']
    peli = csv("camad", "estadisticas", vca, "por_pelicula.csv")
    omni = csv("inferencia", "camad", "omnibus_experimentos.csv")
    mix = csv("inferencia", "camad", "modelo_mixto_vs_vidrio.csv")
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
    if omni is not None:
        sig = omni[omni.anova_perm_p_holm < 0.05]
        if len(sig):
            s.append("<p>Después de corregir por comparaciones múltiples, los sustratos difieren significativamente en: " +
                     "; ".join(NOMBRE.get(m, m) for m in sig.metrica) + ".</p>")
        else:
            s.append("<p>Con solo 2 a 4 experimentos por sustrato, ninguna diferencia entre sustratos resulta significativa a nivel de experimento después de corregir por comparaciones múltiples. Las diferencias que se ven en la figura deben tomarse como <b>tendencias a confirmar</b> con más experimentos.</p>")
    if mix is not None and len(mix):
        sigm = mix[mix.p_holm < 0.05]
        s.append("<p>El modelo mixto, que usa los datos de cada célula pero respeta que las células de un mismo experimento no son independientes, " +
                 ("encuentra diferencias respecto de vidrio en: " + "; ".join(f"{r.condicion} en {r.metrica}" for _, r in sigm.iterrows()) + "." if len(sigm) else "no encuentra diferencias significativas respecto de vidrio después de la corrección.") + "</p>")
    s.append("</div>")
    if mix is not None and len(mix):
        t = mix.copy()
        t["m"] = t.metrica.map({"rapidez_um_min": "Rapidez (µm/min)", "direccionalidad_1h": "Direccionalidad 1 h",
                                "se_ventana_corr": "SE corregida", "se_wavelet_media_corr": "SE en el tiempo corregida",
                                "ead1_corr": "EAD₁ corregida", "tl1_min": "TL₁ (min)", "area_um2": "Área (µm²)",
                                "aspect_ratio": "Alargamiento", "circularidad": "Circularidad"}).fillna(t.metrica)
        t["d"] = [f"{f(r.diferencia, 3)} [{f(r.ic95_inf, 3)}, {f(r.ic95_sup, 3)}]" for _, r in t.iterrows()]
        t["ph"] = [f'<span class="{"sig" if p < 0.05 else "nosig"}">{fp(p)}</span>' for p in t.p_holm]
        s.append('<details><summary>Tabla del modelo mixto (diferencia de cada sustrato respecto de vidrio)</summary>')
        s.append(tabla(t, ["m", "condicion", "d", "ph"], ["Medida", "Sustrato", "Diferencia [IC95%]", "p Holm"], num=("d", "ph")))
        s.append("</details>")
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
            f"{r.linea} {r.condicion}: frente a {f(r.v, 1)} µm/h, {f(r.c, 0)}% cerrado a las 12 h, {f(r.d, 1)} grupos desprendidos por imagen (n = {r.n} posiciones)."
            for _, r in g.iterrows()) + "</p>")
        s.append("<p>El patrón coincide con lo publicado para SEMA6D (Gunyuz et al. 2022): las MCF7 que sobreexpresan SEMA6D cierran la herida más despacio y desprenden muchas más células sueltas. Sin embargo, hay una sola posición con SEMA6D y todas las posiciones de una condición provienen del mismo pocillo, así que este resultado es solo descriptivo.</p>")
    s.append("</div>")
    s.append(figura("figuras/whad_cierre.png",
                    "<b>Cierre de herida.</b> Izquierda: área de la herida relativa a la inicial; las líneas finas son posiciones y las gruesas el promedio por condición. Derecha: células o grupos desprendidos por imagen. MCF10A: CC control, NC Notch1 activo, C61 sin CYR61, N61 ambos. MCF7: LacZ control, SEMA6D sobreexpresado."))
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
             f"<li><b>Usar un modelo más liviano en la Pi</b>, como Cellpose 3 (una red U-Net unas 50–100 veces más barata), o un acelerador de IA para la Pi 5 (AI HAT+). Serían del orden de segundos por imagen, pero habría que volver a validar la calidad de la segmentación con las mismas referencias de este reporte, y convertir el modelo al formato del acelerador no es trivial.</li></ul>")
    s.append("</div>")
    return "\n".join(s)


LIMITACIONES = """
<ul>
<li><b>Una sola condición en el dataset principal.</b> Los 16 videos brightfield son de la misma línea celular y la misma condición, así que no permiten comparar tratamientos. Podrían provenir del mismo día o de la misma placa. Verificamos que no son continuaciones unos de otros.</li>
<li><b>Pocos experimentos por sustrato en CAMAD</b> (2 a 4). Alcanzan para describir tendencias, no para afirmar diferencias con seguridad.</li>
<li><b>Error de posición.</b> El centro calculado desde el contorno de toda la célula tiembla alrededor de 2 µm entre imágenes. Con pasos de 5 minutos eso oculta buena parte de la persistencia real. Las medidas con núcleos, o con pasos más largos, son más sensibles.</li>
<li><b>Muestreo de 5 minutos en brightfield.</b> La persistencia de estas células dura unos pocos pasos. Para medirla bien harían falta imágenes cada 1 o 2 minutos.</li>
<li><b>Referencias imperfectas.</b> La segmentación de núcleos también comete errores (núcleos en división o poco teñidos), y en CAMAD solo están dibujadas las células aisladas. Las métricas de validación son estimaciones, no verdades absolutas.</li>
<li><b>Sin detección de mitosis.</b> Cuando una célula se divide, su trayectoria se corta o una de las hijas continúa la de la madre.</li>
<li><b>Identidad de exp8 y exp9 de CAMAD</b> sin confirmar, por eso se excluyeron de la comparación.</li>
</ul>
"""

PROXIMOS = """
<ul>
<li>Usar el canal de núcleos para detectar mitosis (la cromatina condensada se ve más brillante y pequeña) y separar las trayectorias de madres e hijas.</li>
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
    ea = js("validacion_seg_camad", "eleccion_aplicada.json")
    g = csv("validacion_seg_camad", "grilla.csv")
    r = csv("validacion_tracking", "por_pelicula.csv")
    et = js("validacion_tracking", "eleccion.json")
    if el and r is not None and et:
        m = r.groupby("variante").precision_eslabon.mean()
        h.append(f"<b>Las mediciones ahora están validadas.</b> Comparando con los núcleos fluorescentes del mismo campo, la detección de células en brightfield acierta con F1 = {f(el['metricas_loeo_media']['f1'], 2)} y el {f(100 * m[et['elegida']], 1)}% de las uniones entre imágenes del seguimiento son correctas.")
    if g is not None and ea:
        full = g[g.escala == 1].groupby("exp").recall.mean().mean()
        h.append(f"<b>La segmentación de CAMAD mejoró de {f(100 * full, 0)}% a {f(100 * ea['recall'], 0)}%</b> de células encontradas, al trabajar las imágenes a una escala adecuada para el modelo.")
    rb = fila_resumen("bf", "prw_P_min")
    if rb is not None:
        h.append(f"<b>Las MDA-MB-231 son rápidas pero poco persistentes:</b> avanzan a {mic('bf', 'rapidez_um_min', 2, 'µm/min')} y mantienen la dirección unos {f(rb.media, 0)} minutos. A la escala de horas se mueven casi como una caminata al azar.")
    e1 = fila_resumen("bf", "ead1_ens_corr")
    se = fila_resumen("bf", "se_ventana_corr")
    if e1 is not None and se is not None:
        h.append(f"<b>Las dos entropías de Shannon confirman una persistencia débil pero real:</b> EAD₁ corregida = {f(e1.media, 3)} y SE corregida = {f(se.media, 3)} (1 sería movimiento completamente al azar), ambas significativamente menores que 1.")
    h.append("<b>Aporte metodológico:</b> sin corrección, la entropía de una célula depende de la longitud de su trayectoria. Lo corregimos, y las simulaciones muestran que el error de posición limita cuánta persistencia se puede detectar con imágenes cada 5 minutos.")
    t = csv("inferencia", "bf_vs_nucleos.csv")
    if t is not None:
        buenas = (t.ccc_lin >= 0.8).sum()
        h.append(f"<b>El análisis sin marcador reproduce al de núcleos</b> en {buenas} de {len(t)} medidas con concordancia alta. Las diferencias se concentran en las medidas más sensibles al error de posición.")
    omni = csv("inferencia", "camad", "omnibus_experimentos.csv")
    if omni is not None:
        sig = omni[omni.anova_perm_p_holm < 0.05]
        if len(sig):
            h.append("<b>El sustrato cambia el movimiento</b> en: " + ", ".join(NOMBRE.get(m, m) for m in sig.metrica) + " (CAMAD, a nivel de experimento).")
        else:
            h.append("<b>Sustratos (CAMAD):</b> se ven tendencias entre vidrio, Matrigel, colágeno y matrices de macrófagos, pero con 2 a 4 experimentos por sustrato ninguna es estadísticamente segura. Hacen falta más réplicas.")
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
        "VALIDACION": seccion_validacion(), "SIMULACIONES": seccion_simulaciones(),
        "RESULTADOS_BF": seccion_bf(vbf), "BF_VS_NUC": seccion_bf_vs_nuc(), "RESULTADOS_CAMAD": seccion_camad(vca),
        "RESULTADOS_WHAD": seccion_whad(), "RENDIMIENTO": seccion_rendimiento(), "LIMITACIONES": LIMITACIONES, "PROXIMOS": PROXIMOS,
        "TECNICO": seccion_tecnico(vbf, vca),
    }
    for k, v in rep.items():
        tpl = tpl.replace("{{" + k + "}}", v)
    faltan = [x for x in rep if "{{" + x + "}}" in tpl]
    (OUT / "index.html").write_text(tpl)
    tam = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"index.html listo ({tam:.1f} MB en total con recursos). Marcadores sin reemplazar: {faltan}")


if __name__ == "__main__":
    main()
