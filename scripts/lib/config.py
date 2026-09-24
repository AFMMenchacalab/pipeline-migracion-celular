"""
Configuración central del pipeline v2 (2026-09-23).

Un solo lugar con la definición de cada dataset (rutas, calibración espacial
y temporal, cómo se agrupan los frames en películas y, para CAMAD, qué
condición experimental es cada experimento). Antes cada script tenía su
propia copia de estas constantes (FRAMES_PER_MOVIE, rutas, calibración) y
había que mantenerlas sincronizadas a mano; con tres datasets eso ya no
escala.

Dónde va cada cosa:
  - ROOT/resultados/v2/...   tablas y figuras (livianas, se sincronizan por
                             Syncthing a la otra PC).
  - CACHE (~/microscopio_cache, FUERA de la carpeta sincronizada): máscaras
    por frame, flows de Cellpose y frames extraídos de los AVI de CAMAD.
    Son decenas de GB regenerables; sincronizarlos solo llenaría el disco
    de la otra PC.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
RES = ROOT / "resultados" / "v2"
CACHE = Path.home() / "microscopio_cache"

# CAMAD se reduce 4x en cada eje al extraer los frames (ver
# 10_preparar_camad.py): 0.117 um/px (objetivo 40x) -> 0.468 um/px, del mismo
# orden que el dataset brightfield (0.65 um/px), para el que Cellpose-SAM ya
# está validado. A resolución completa una célula mide 150-400 px de
# diámetro y cada frame (2568x1912) tarda ~5x más en GPU sin ganar precisión
# útil en el centroide.
CAMAD_DOWNSAMPLE = 4

# area_min_um2 (en cada dataset): objetos más chicos se descartan antes del
# tracking. Cellpose acepta objetos desde 15 px (~3-7 um2), que no pueden ser
# una célula (célula BF mediana ~370 um2, núcleo ~100 um2, célula CAMAD recién
# sembrada >= ~150 um2, macrófago RAW ~100 um2): solo agregarían trayectorias
# de basura/fragmentos con rapidez y ángulos de ruido puro.

DATASETS = {
    "bf": {
        "descripcion": "Brightfield MDA-MB-231 (Zenodo 10074471, Training-source-BF)",
        "src": DATASETS_DIR / "brightfield-mdamb231" / "1.1-training-source-BF-1600",
        "pattern": "*.tif",
        "um_per_px": 1 / 1.539376,          # XResolution del TIFF (ImageJ)
        "dt_s": 300.016,                    # finterval del TIFF (igual en las 16 películas)
        "frames_per_movie": 100,
        "cellpose_diameter": None,          # cpsam no necesita diámetro
        "area_min_um2": 50.0,
    },
    "sirdna": {
        "descripcion": "Núcleos SiR-DNA de los MISMOS campos que 'bf' (Zenodo 10074471, Training-target-sirDNA)",
        "src": DATASETS_DIR / "sirdna-mdamb231",
        "pattern": "**/*.tif",
        "um_per_px": 1 / 1.539376,
        "dt_s": 300.016,
        "frames_per_movie": 100,
        "cellpose_diameter": None,
        "area_min_um2": 20.0,
    },
    "camad": {
        "descripcion": "CAMAD (Iheme et al. 2024, IEEE Data Descr., Zenodo 12806149): contraste de fase, 40x",
        "src": CACHE / "camad_frames",       # stacks .npy generados por 10_preparar_camad.py
        "videos": DATASETS_DIR / "camad-mdamb231" / "camad" / "videos",
        "gt_images": DATASETS_DIR / "camad-mdamb231" / "camad" / "images",
        "gt_rois": DATASETS_DIR / "camad-mdamb231" / "camad" / "rois",
        "um_per_px": 0.117 * CAMAD_DOWNSAMPLE,
        "um_per_px_full": 0.117,
        # 600 frames en 5 h (paper) = 30 s/frame; el script oficial de
        # extracción del dataset usa "frames_per_minute = 2", consistente.
        "dt_s": 30.0,
        "frames_per_movie": 600,
        "cellpose_diameter": None,
        "area_min_um2": 60.0,
    },
}

# Condición de cada experimento de CAMAD, decodificada del nombre del video
# (convención: <sustrato><línea imageada>liveimaging<fecha>) y del paper
# (Iheme et al. 2024, sección "Substrate Preparation"). OJO exp8/exp9: el
# nombre es "glass231matrixconfluentRAWlive..." (orden de palabras distinto al
# de exp3/7/10/13 "glassrawmatrixconfluent231live..."). Con la convención
# <sustrato><célula filmada>live, serían macrófagos RAW 264.7 sobre matriz de
# MDA-MB-231 confluente (el paper dice que CAMAD incluye células RAW 264.7
# filmadas). Pero a la vista las células son alargadas y parecidas a las
# MDA-MB-231 de exp3/exp7 (algo más chicas), y la planilla oficial de
# experimentos no está disponible. Como la identidad es INCIERTA, exp8/9
# quedan FUERA de la comparación principal entre sustratos de MDA-MB-231 y
# se muestran aparte.
CAMAD_CONDICION = {
    1: "Matrigel", 2: "Matrigel", 6: "Matrigel", 11: "Matrigel",
    12: "Colágeno I", 16: "Colágeno I",
    14: "Vidrio", 15: "Vidrio",
    4: "Matriz RAW dispersa", 5: "Matriz RAW dispersa",
    3: "Matriz RAW confluente", 7: "Matriz RAW confluente",
    10: "Matriz RAW confluente", 13: "Matriz RAW confluente",
    8: "Matriz 231 (exp8-9)", 9: "Matriz 231 (exp8-9)",
}
CAMAD_LINEA = {k: ("incierta" if k in (8, 9) else "MDA-MB-231") for k in CAMAD_CONDICION}
CAMAD_ORDEN_CONDICIONES = ["Vidrio", "Matrigel", "Colágeno I", "Matriz RAW dispersa",
                           "Matriz RAW confluente", "Matriz 231 (exp8-9)"]


def cache_dir(*parts):
    p = CACHE.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def res_dir(*parts):
    p = RES.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
