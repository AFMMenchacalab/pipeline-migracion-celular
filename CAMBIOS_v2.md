# Pipeline v2: qué cambió y por qué (2026-09-23/24)

Los scripts v1 (`scripts/01_*.py` … `07_*.py`) se dejaron **intactos** como
referencia. La versión nueva son los scripts `10_*.py` … `21_*.py` más la
librería `scripts/lib/`, y guarda sus resultados en `resultados/v2/`.
Los archivos pesados (máscaras, flows y frames de CAMAD) van a
`~/microscopio_cache/`, **fuera de Syncthing**, para no llenar el disco de
la otra PC. Todo se regenera con `bash scripts/correr_v2.sh`.

Cada decisión de abajo se tomó **midiendo**, no a ojo. Los números salen de
los CSV indicados.

---

## 1. Datos nuevos

| Dataset | Qué es | Para qué se usa |
|---|---|---|
| **SiR-DNA** (Zenodo 10074471, `Training-target-sirDNA.zip`, MD5 verificado) | 1600 imágenes de **núcleos por fluorescencia** tomadas en el mismo instante y campo que las 1600 de brightfield | Sirve de **referencia independiente** para validar la segmentación y el tracking de BF, algo que en v1 no existía porque LFCT no se pudo bajar. También da un segundo pipeline (tracking de núcleos) contra el cual comparar los estadísticos. |
| **CAMAD** (Zenodo 12806149; paper Iheme et al. 2024, IEEE Data Descriptions) | 16 videos de 600 frames: **30 s/frame, 5 h**, contraste de fase 40×, 0.117 µm/px | Comparación entre **sustratos**: vidrio, Matrigel, colágeno I y matriz de macrófagos RAW 264.7 dispersa o confluente. exp8 y exp9 (`glass231matrixconfluentRAWlive`) podrían ser macrófagos RAW 264.7 sobre matriz de MDA-MB-231 según el nombre, pero morfológicamente parecen MDA-MB-231. Como la identidad es **incierta**, se analizan aparte. |
| **WHAD** (mismo Zenodo) | Ensayos de herida MCF10A/MCF7 con la herida delineada a mano | Cinética de cierre y desprendimiento celular (análisis secundario, `21_whad.py`). |

Calibración de CAMAD: el paper del dataset dice "600 frames … 5 h"
(30 s/frame), y el script oficial de extracción usa `frames_per_minute = 2`,
así que coincide. Algunos videos traen frames negros de relleno: exp3 (480–539),
exp8 (124–229) y un frame suelto en exp1 y exp10. Se marcan como inválidos y
el tracking los trata como tiempo perdido (`10_preparar_camad.py`).

## 2. Segmentación (`11_segmentar_gpu.py`, `12_*`, `13_*`)

1. **fp16 en la RX 6800 XT.** Cellpose 4 usa bf16 por defecto, pero RDNA2
   no tiene bf16 nativo: 4.1 s/imagen, contra 2.8 s en fp32 y ~1.7 s en fp16.
   Las máscaras en fp16 son las mismas que en fp32 (AP@0.5 = 0.995–1.000).
2. **Se guardan los flows.** Los umbrales (`cellprob_threshold`,
   `flow_threshold`) solo afectan un post-proceso de ~0.1 s, así que se
   ajustan sin volver a correr la red.
3. **Umbrales BF elegidos contra núcleos.** Cada célula BF debería contener
   exactamente un núcleo. Se barren los umbrales, se elige por F1 con
   validación cruzada dejando una película afuera, y se reportan
   precisión, recall, fusiones y células perdidas
   (`resultados/v2/validacion_seg_bf/`).
4. **CAMAD a resolución reducida 4×.** A resolución completa (lo que hacía
   v1) Cellpose-SAM recupera ~53% de las células anotadas. Reduciendo 4×
   (0.47 µm/px, similar a BF) recupera ~93%. Se barrieron las escalas 1, 2,
   4 y 6× y los umbrales, con validación cruzada por experimento
   (`resultados/v2/validacion_seg_camad/`).
5. **Referencia CAMAD corregida.** v1 armaba las células de referencia con
   componentes conexas de la máscara binaria, así que dos células que se
   tocan contaban como una. v2 usa los polígonos de los .roi de ImageJ, una
   instancia por célula. Además el paper aclara que solo se anotaron las
   células **aisladas**, por eso la "precisión" de v1 daba mal (contaba
   células reales no anotadas como falsos positivos). v2 usa métricas
   pensadas para referencia parcial (`lib/metricas_seg.py`).

## 3. Tracking (`14_tracking.py`, `15_validacion_tracking.py`)

- Los umbrales ahora están en **µm** (sirven para BF y para CAMAD).
- Hay variantes comparadas contra el tracking de núcleos: `v1` (igual al
  anterior), `dist` (umbral físico y huecos de hasta 3 frames), `dist_tam`
  (penaliza el cambio de tamaño entre frames) y `dist_tam_corte` (corta la
  trayectoria en saltos anómalos). La elegida sale de
  `resultados/v2/validacion_tracking/eleccion.json`.
- **La detección de mitosis de laptrack se probó y se descartó.** En la
  película 1 da 0, 7 o 74 divisiones según el umbral sea 5, 8 o 12 µm, cuando
  lo esperable es ~50. Sin un marcador de mitosis no es confiable. Queda
  como trabajo futuro, usando el canal SiR-DNA, donde la cromatina
  condensada sí marca la mitosis.

## 4. Estadística (`16_estadisticas.py`, `lib/motilidad.py`, `lib/entropia.py`)

- **PRW con ruido de localización:** MSD = 4D[τ − P(1 − e^(−τ/P))] + 4σ².
  Sin el término 4σ², el error del centroide se confunde con difusión a
  lags cortos y achica P artificialmente. Es la explicación probable de
  los 4 ajustes "degenerados" (P ≈ 0) de v1.
- **Direccionalidad en ventana fija de 1 h**, en vez de sobre toda la
  trayectoria: el cociente baja solo con la duración.
- **Entropía de Shannon (pedido del director):**
  - SE del espectro de Fourier de la velocidad y SE(t) por wavelet de
    Morlet (Liu et al., Biophys J 2021).
  - EAD(τ), TL₁/EAD₁ y EAD(t) con ventana deslizante (Liu et al., Biophys J
    2024).
  - La implementación reproduce los valores del paper en simulaciones PRW:
    SE = 0.88 / 0.76 / 0.59 / 0.42 contra ~0.90 / 0.75 / 0.60 / 0.45
    (`19_simulaciones_metodos.py`).
- **Corrección por tamaño de muestra (mejora propia).** Con pocos ángulos
  por célula la entropía sale sesgada hacia abajo. Con 20 ángulos al azar
  en 36 bins la EAD da 0.74 en vez de 1. Como acá una célula tiene 20–100
  pasos, sin corregir la EAD mediría la longitud de la trayectoria. Se
  divide por el valor esperado bajo aleatoriedad para el mismo número de
  ángulos (Monte Carlo). En simulación la EAD corregida da 1.00 para
  movimiento aleatorio con cualquier número de ángulos.
- **Paso de análisis común de 5 min** en BF y CAMAD, más un análisis de
  sensibilidad al paso. Las simulaciones muestran que con ruido de
  centroide σ ≈ 2 µm y pasos de 5 min, la EAD₁ queda en ~0.98 aunque la
  persistencia real sea de 60 min. El ruido limita la sensibilidad y hay
  que tenerlo en cuenta al interpretar.
- **Inferencia (`17_inferencia.py`):** las películas y experimentos siguen
  siendo la réplica, con bootstrap entre películas. En CAMAD se usan
  Kruskal-Wallis y ANOVA por permutación a nivel de experimento, más un
  modelo mixto con efecto aleatorio por experimento a nivel de célula, y
  corrección de Holm entre métricas.

## 5. Rendimiento

- GPU: fp16 más lectura/escritura en hilos. La segmentación BF pasa de
  2 h 04 min a ~45–50 min con la GPU sola.
- CPU: detecciones, tracking y estadística corren en paralelo por película
  (16 procesos sobre 20 hilos).
- `systemd-inhibit` impide que la PC se suspenda mientras corre el pipeline.
