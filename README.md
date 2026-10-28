# Pipeline de migración celular (Cellpose + laptrack + entropía de Shannon)

Análisis de videos de microscopía de células MDA-MB-231: segmentación con
Cellpose-SAM, seguimiento de cada célula con laptrack y medidas de
rapidez y persistencia direccional. Incluye las medidas clásicas (MSD,
caminata aleatoria persistente con error de posición, VACF) y las dos
medidas de persistencia basadas en entropía de Shannon:

- **SE**, entropía del espectro de la velocidad, por Fourier y por wavelet.
  Liu et al., *Biophys J* 120:2552 (2021).
- **EAD**, entropía de la distribución de ángulos de giro, con TL₁, EAD₁ y
  EAD(t). Liu et al., *Biophys J* 123:730 (2024).

A ambas se les agrega una corrección por tamaño de muestra (ver `CAMBIOS_v2.md`).

## Datos

Los datos no se versionan: se descargan en `datasets/`.

| Dataset | Fuente | Carpeta |
|---|---|---|
| Brightfield MDA-MB-231 (16 videos × 100 frames, 5 min, 0.65 µm/px) | Zenodo [10.5281/zenodo.10074471](https://doi.org/10.5281/zenodo.10074471), `Training-source-BF.zip` | `datasets/brightfield-mdamb231/` |
| Núcleos SiR-DNA de los mismos campos | mismo registro, `Training-target-sirDNA.zip` | `datasets/sirdna-mdamb231/` |
| CAMAD y WHAD (sustratos, 30 s/frame, 40×; cierre de herida) | Zenodo [10.5281/zenodo.12806149](https://doi.org/10.5281/zenodo.12806149), `whad_camad.zip` | `datasets/camad-mdamb231/` |

## Instalación

**Rápida (recomendada), en cualquier PC:** `./instalar.sh` en Linux o
`instalar.bat` en Windows. Detecta la GPU (NVIDIA → CUDA, AMD → ROCm, o CPU),
instala todo y lo verifica. Guía completa en [INSTALAR.md](INSTALAR.md).

**Interfaz gráfica** (recepción desde la Raspberry, segmentación en vivo y
tracking, con vista previa): `./iniciar_interfaz.sh` o `iniciar_interfaz.bat`.

Manual, reproduciendo exactamente el entorno de la PC de desarrollo (ROCm):

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

El `venv` no se comparte entre PCs: las rutas quedan escritas dentro. Hay que
recrearlo en cada máquina.

## Uso

```bash
bash scripts/correr_v2.sh          # todo el pipeline v2 (≈3–4 h con GPU)
bash scripts/correr_v2.sh bf       # brightfield + núcleos + linajes
bash scripts/correr_v2.sh camad    # CAMAD
bash scripts/correr_v2.sh reporte  # simulaciones, WHAD, rendimiento y reporte HTML
```

El reporte queda en `resultados/v2/reporte_html/index.html` (con sus carpetas
`figuras/`, `fotos/` y `videos/`). Los resultados quedan en `resultados/v2/`;
los archivos pesados (máscaras, flujos, frames de CAMAD) en `~/microscopio_cache/`.

## Estructura

| Script | Qué hace |
|---|---|
| `01_…07_*.py` | Pipeline v1 (se conserva como referencia) |
| `10_preparar_camad.py` | Decodifica los AVI de CAMAD (reducción 4×, frames negros) |
| `11_segmentar_gpu.py` | Cellpose-SAM en GPU (fp16) para cualquier dataset; guarda máscaras y flujos |
| `12_ajuste_segmentacion_camad.py` | Escala y umbrales de CAMAD contra la anotación manual (.roi) |
| `13_ajuste_segmentacion_bf.py` | Umbrales de brightfield contra los núcleos SiR-DNA |
| `14_tracking.py` | Detecciones y tracking (variantes) en paralelo por película |
| `15_validacion_tracking.py` | Tracking brightfield comparado con el tracking de núcleos |
| `16_estadisticas.py` | Todas las medidas por película: MSD, PRW, VACF, SE, EAD, colectivas, morfología |
| `17_inferencia.py` | Bootstrap entre películas, pruebas contra el nulo, BF vs núcleos, sustratos (permutación y modelo mixto) |
| `18_figuras.py` | Figuras |
| `19_simulaciones_metodos.py` | Validación de SE y EAD con simulaciones PRW y análisis del sesgo |
| `21_whad.py` | Cinética de cierre de herida (WHAD) |
| `22_animaciones.py` | Videos MP4 para el reporte |
| `23_reporte_html.py` | Reporte HTML (plantilla en `scripts/plantillas/reporte.html`) |
| `24_rendimiento.py` | FLOPs de Cellpose-SAM, tiempos en GPU/CPU y estimación para Raspberry Pi 5 |
| `25_cellpose3_vs_sam.py` | Comparación con Cellpose 3 (requiere instalarlo aparte; ver encabezado) |
| `26_linajes.py` | Prototipo: detección de divisiones con núcleos y árboles genealógicos |
| `lib/` | Configuración, entropías, motilidad, tracking, métricas, inferencia, estilo |

Cada decisión y su justificación están en `CAMBIOS_v2.md` y en el
encabezado de cada script. `ESTADO.md` es el registro de avance.
