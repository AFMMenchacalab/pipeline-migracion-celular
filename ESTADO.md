# Estado del proyecto — pipeline Cellpose + laptrack

Última actualización: 2026-09-24 (pipeline v2)

---

## 🆕 PIPELINE v2 (noche del 2026-09-23 al 24)

Detalle de cada cambio y su porqué: `CAMBIOS_v2.md`. Cómo correrlo: `README.md`.
Repositorio (privado): https://github.com/AFMMenchacalab/pipeline-migracion-celular

- [x] **Datos nuevos**: núcleos SiR-DNA de los mismos 1600 campos BF (Zenodo 10074471,
  `datasets/sirdna-mdamb231/`, MD5 verificado) -> referencia para validar segmentación y
  tracking (reemplaza a LFCT). CAMAD procesado completo (30 s/frame, 0.117 um/px, 40x;
  datos del paper Iheme et al. 2024). WHAD (cierre de herida) analizado con sus máscaras manuales.
- [x] Segmentación GPU en fp16 (1.7 s/img vs 4.1 en bf16), flows guardados en `~/microscopio_cache/`.
- [x] CAMAD a resolución reducida 4x: recall contra anotación manual 62% -> 87% (IoU 0.55 -> 0.72; 16 exp). Difíciles: exp11 y exp12 (células desenfocadas).
- [x] Umbrales BF elegidos contra núcleos; variantes de tracking validadas contra tracking de núcleos.
- [x] Entropías de Shannon (SE Liu 2021, EAD Liu 2024) con corrección por tamaño de muestra;
  simulaciones que reproducen los valores publicados (`19_simulaciones_metodos.py`).
- [x] Reporte: HTML con videos, fotos y simulador interactivo (`23_reporte_html.py`).
- ⚠️ exp8/exp9 de CAMAD: identidad celular incierta (nombre sugiere RAW 264.7, morfología
  parece MDA-MB-231) -> fuera de la comparación entre sustratos.
- ⚠️ `.git` vive dentro de la carpeta de Syncthing: no hacer commits desde las dos PCs a la vez.

---

## ✅ HECHO

### Infraestructura
- [x] Estructura de carpetas (`datasets/`, `scripts/`, `resultados/`)
- [x] venv con cellpose 4.2.1, laptrack 0.17, scikit-image, seaborn, tifffile
- [x] `requirements.txt` generado y verificado (imports OK)
- [x] **Fix venv tras sync entre PCs**: el venv se creó en la otra PC bajo `/home/AlexFlores/...` (mayúscula); acá el usuario es `alexflores` (minúscula) y Linux distingue mayúsculas, así que 31 shebangs + `activate` + `pyvenv.cfg` apuntaban a una ruta inexistente y `pip`/`python` caían al Python del sistema. Corregido con `sed` sobre `venv/bin/*` y `venv/pyvenv.cfg`. **Si se vuelve a sincronizar el venv entre las dos PCs y el nombre de usuario difiere en mayúsculas, va a romperse de nuevo** — mejor no versionar/sincronizar `venv/` por Syncthing, recrearlo por PC con `requirements.txt`.

### Datasets
- [x] **Brightfield MDA-MB-231** — Zenodo `10.5281/zenodo.10074471`, 1600 TIFF, MD5 verificado
- [x] **CAMAD/WHAD** — Zenodo `10.5281/zenodo.12806149` (12.6 GB), evita el paywall de IEEE DataPort
- [ ] ~~LFCT~~ — **imposible**: DOI y enlaces de Box.com dan 404 (ver `ESTADO` → Bloqueado)

### Pipeline (sobre 50 imágenes de la película _01)
- [x] `01_segmentacion.py` — Cellpose `cpsam`, paralelizado (6 workers × 4 threads, ~1.8× vs secuencial), barra de progreso
- [x] `02_tracking.py` — laptrack, 365 trayectorias de 10,798 detecciones
- [x] `03_animacion.py` — reescrito para generar un GIF por película (16 en total, `animacion_pelicula_01.gif`...`_16.gif` en `resultados/tracking/`, ~855 MB total), coloreadas por `track_id` + estelas. Corrida completa: 22m36s.
- [x] `04_estadisticas.py` — MSD (TA/EA/TA-EA), α, PRW, VACF, direccionalidad, morfología
- [x] `05_figuras.py` — 15 figuras estilo publicación (PNG 300dpi + PDF + SVG) con captions
- [x] `07_validacion_segmentacion.py` — corre el mismo Cellpose de `01_segmentacion.py` sobre las 96 imágenes con ground truth humano de CAMAD (16 exp × 6, misma línea MDA-MB-231) y compara. **Recall ≈ 59%** (de las células que un humano marcó, Cellpose encuentra el 59% con IoU≥0.5; cuando encuentra, la superposición es buena, IoU medio 0.75 en los matches). **Precisión/F1 NO son confiables acá**: en varios experimentos (exp4, exp5, exp11, exp12, exp14) Cellpose predice 4-9× más células que las anotadas, y aun así el recall se mantiene alto en la mayoría de esos casos (ej. exp5: 9.3× más predicciones, recall=0.89) — evidencia visual (`validacion_ejemplo_bueno.png` vs. imágenes de exp4/5) de que el ground truth de CAMAD es un **subconjunto curado**, no todas las células del campo, así que cuenta como "falso positivo" a células reales que Cellpose sí encontró bien pero que no estaban en la anotación. Además hay un desacuerdo sistemático de criterio de borde entre el humano y Cellpose (`validacion_ejemplo_dificil.png`, exp6: el contorno humano incluye un halo mucho más amplio que el core que segmenta Cellpose), que baja el IoU a nivel píxel (mediana 0.27) sin que sea necesariamente un error de Cellpose. Salida en `resultados/validacion_camad/`.
- [x] `06_morfologia_movimiento.py` — correlación forma↔movimiento célula por célula (elongación/área/irregularidad vs. rapidez/direccionalidad/α), calculada pooled y dentro-de-cada-película por separado (para no confundir diferencia entre campos con relación real). Resultado: correlaciones débiles (r≤0.16) — la forma casi no predice el movimiento individual; el campo de origen explica más que la forma de la célula. Salida en `resultados/estadisticas/correlaciones_forma_movimiento.csv` + `resultados/figuras/rel_*.png`

### Hallazgos importantes
- [x] Calibración real leída de metadatos: **0.6496 µm/px, 300.02 s/frame**
- [x] **Las 1600 imágenes son 16 películas de 100 frames**, no una serie continua
- [x] Revisión crítica del código → 5 defectos confirmados empíricamente

---

## 🔧 ARREGLADO (2026-09-21) — re-corrido sobre las 16 películas completas

`04_estadisticas.py` y `05_figuras.py` reescritos. Detalle técnico completo
en el docstring de `04_estadisticas.py`. Los 6 arreglos, resueltos con un
mismo cambio de diseño: calcular todo POR PELÍCULA primero (16 réplicas
independientes) y recién ahí agregar entre películas.

- [x] **1. Ajuste de α por tramos**, no uno global. Detección automática de
  quiebres (fuerza bruta minimizando SSE) sobre la curva de gran media:
  3 tramos en τ (s) = [300-1200] / [1500-6900] / [7200-8700], con
  α = 0.93 / 1.16 / 1.12 respectivamente (bootstrap entre películas, ver
  `resumen_bootstrap.csv`). Ya no hay un α=1.255 único que no describe nada.
- [x] **2. Sesgo de supervivencia**: cada curva EA-MSD (por película) se
  trunca en el primer τ donde la cohorte cae debajo del 50% de su tamaño en
  τ=1 (`SURVIVORSHIP_MIN_FRAC`), antes de ajustar nada.
- [x] **3. Barras de error por bootstrap**: ya no es el SE analítico del OLS
  sobre la curva pooled (puntos no independientes). Ahora es bootstrap
  sobre las 16 PELÍCULAS (unidad genuinamente independiente) — resample con
  reemplazo de películas, no de células ni de puntos del MSD.
- [x] **4. Bug de huecos**: TA-MSD, EA-MSD, VACF y ángulos de giro usan la
  diferencia real de `frame` (no el índice de fila) en todos lados.
  *(Nota: al implementar este arreglo metí un bug de signo — `frames[i]-frames[j]`
  en vez de `frames[j]-frames[i]` — que dejaba todos los τ individuales
  negativos y por eso `msd_ta.csv`/`por_celula.csv` no tenían ningún α
  individual calculado. Lo agarré probando antes de mostrar resultados y
  quedó corregido — mencionado acá porque si alguien revisa el historial de
  este archivo puede encontrar una corrida vieja con esos campos vacíos.)*
- [x] **5. Barras log → violín+caja**: `05_figuras.py` reescrito, ya no hay
  ninguna barra en escala log. Ahora es violín+caja CON UNA CAJA POR
  PELÍCULA (no una barra pooleando miles de células), con la media±IC95%
  bootstrap entre películas superpuesta como banda de referencia.
- [x] **6. Replicación correcta**: `resultados/estadisticas/resumen_por_pelicula.csv`
  es el dataset real para inferencia (n=16, una fila por película). Todo lo
  demás (por_celula.csv, msd_ta.csv, etc.) es descriptivo/insumo, no la base
  de ningún número final reportado.

**Archivos nuevos clave:** `resumen_por_pelicula.csv` (las 16 películas) y
`resumen_bootstrap.csv` (media/SEM/IC95% bootstrap — los números a citar).
`resumen.md` en `resultados/` tiene la versión narrativa.

**Fix adicional (mismo día):** 4 de las 16 películas (2, 5, 11, 13) daban un
ajuste PRW degenerado (τ_p≈0, A≈0 pero R² igual alto) — degeneración
matemática conocida del modelo PRW: cuando no hay curvatura de persistencia
detectable en el rango de τ disponible, el modelo colapsa a una recta con
A y τ_p empujados los dos hacia cero (su cociente D=A/4τ_p se mantiene
finito). Esos ceros estaban arrastrando el τ_p/D/A promedio hacia abajo.
Arreglado: se descarta el ajuste PRW cuando τ_p < finterval_s (no es un
tiempo de persistencia medible con este muestreo de 5 min); quedan n=12
películas para τ_p/D/A (marcado en `prw_por_pelicula.csv`, columna
`valido`). τ_p pasó de 720±147s a 938±147s.

---

## 🚀 SIGUIENTE (en casa, con la RX 6800 XT)

- [x] Instalar ROCm + PyTorch-ROCm (la RX 6800 XT sí tiene soporte oficial) — ROCm 7.2.4 ya estaba instalado en el sistema; venv reparado (ver nota) y `torch==2.14.0+rocm7.2` instalado
- [x] Verificar que Cellpose detecta la GPU (`torch.cuda.is_available()` con ROCm) — `torch.cuda.is_available()` → `True`, `torch.cuda.get_device_name(0)` → `AMD Radeon RX 6800 XT`; `cellpose.core.use_gpu()` → `True`
- [x] **Segmentar las 1600 imágenes** — 2h04min en GPU (vs. ~6.7h estimadas en CPU), 1 proceso, `torch==2.14.0+rocm7.2`. Promedio 192.7 células/imagen. 1600 `_masks.npy` + 1600 `_overlay.png` en `resultados/segmentacion/`
- [x] **Trackear cada película por separado** (16 × 100 frames) — `02_tracking.py` reescrito: confirma los límites de película leyendo metadatos TIFF (finterval/min cambian cada 100 imágenes), corre LapTrack independiente por película (frame reinicia 0-99), y arma `track_id` global único (`película*100000 + id_local`) para poder concatenar todo en `resultados/tracking/tracks.csv` con columna `movie`. 8051 tracks (>=3 frames) en 308,241 detecciones sobre 16 películas. Visualización de control por película en `trayectorias_pelicula_XX.png`.
  - ⚠️ **Película 15 es un outlier de densidad** (revisado visualmente, no es bug): 60,620 detecciones vs. 7,189-26,163 en el resto, ~600 células/frame vs. ~190 de promedio. Comparando `1450_overlay.png` contra `0050_overlay.png` la segmentación se ve correcta en ambas — es un campo genuinamente mucho más confluente, no ruido de Cellpose. Igual conviene tenerlo presente: al mezclar los 16 campos en el análisis, esta película va a pesar mucho más si se pseudorreplica por célula (ver arreglo #6 de más arriba).
- [x] Re-correr estadísticos con los 6 arreglos aplicados (ver sección "ARREGLADO" arriba)
- [ ] Ahora sí: gráficas comparativas entre los 16 campos, con tests de significancia reales — **nota:** con una sola condición experimental (brightfield MDA-MB-231) no hay un segundo grupo con el que comparar; esto tiene sentido recién cuando se procese CAMAD (sustratos distintos, ver Ideas pendientes) o algún otro grupo real

---

## ⛔ BLOQUEADO / DESCARTADO

- **LFCT**: enlaces muertos (Zenodo 404 + ~195 links de Box.com 404). Sin él no hay ground truth, así que no se pueden calcular métricas validadas de tracking (ID switches, FP/FN, fragmentación). Contacto autores: fiurici@clemson.edu
- **iGPU Radeon 890M**: no soportada oficialmente por ROCm. No vale la pena pelearse con ella teniendo la RX 6800 XT.
- **Muestreo cada 5 min**: demasiado grueso frente al τ_p de ~30 min (la VACF pierde 85% en un frame). No se arregla por software; requeriría re-adquirir a ~1 min.

---

## 💡 IDEAS PENDIENTES

- [ ] Procesar CAMAD (videos AVI → frames → Cellpose → laptrack) para comparar motilidad entre sustratos (vidrio / matrigel / colágeno) — ahí sí habría grupos reales que comparar
- [ ] Usar `OverLapTrack` (linking por solapamiento de máscara) en vez de solo centroide, más robusto en zonas densas
- [ ] Habilitar splitting en laptrack para capturar mitosis y reconstruir linaje
