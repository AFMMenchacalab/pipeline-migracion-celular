#!/usr/bin/env bash
# Pipeline v2 completo, en el orden en que se generaron los resultados del
# reporte del 2026-09-25. Cada paso deja su log en resultados/v2/logs/.
# La mayoría de los pasos es reanudable (saltea lo ya hecho).
#
#   bash scripts/correr_v2.sh            # todo (~3-4 h con la RX 6800 XT)
#   bash scripts/correr_v2.sh bf         # brightfield + núcleos + linajes
#   bash scripts/correr_v2.sh camad      # CAMAD
#   bash scripts/correr_v2.sh reporte    # simulaciones, WHAD, rendimiento y reporte HTML
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
LOG=resultados/v2/logs
mkdir -p "$LOG"
PARTE="${1:-todo}"

paso() {  # paso <nombre> <comando...>
  local n="$1"; shift
  echo "[$(date +%H:%M:%S)] >>> $n" | tee -a "$LOG/orquestador.log"
  if "$@" > "$LOG/$n.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK  $n" | tee -a "$LOG/orquestador.log"
  else
    echo "[$(date +%H:%M:%S)] FALLA $n (ver $LOG/$n.log)" | tee -a "$LOG/orquestador.log"
    exit 1
  fi
}
json() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

if [[ "$PARTE" == "todo" || "$PARTE" == "bf" ]]; then
  # segmentación en GPU (fp16, guarda flows); núcleos y células del mismo campo
  paso seg_bf          $PY scripts/11_segmentar_gpu.py --dataset bf
  paso seg_sirdna      $PY scripts/11_segmentar_gpu.py --dataset sirdna
  # umbrales BF contra núcleos (con regla de desempate: resultan los de por defecto)
  paso ajuste_seg_bf   $PY scripts/13_ajuste_segmentacion_bf.py --aplicar
  paso tracking_sirdna $PY scripts/14_tracking.py --dataset sirdna --variante dist
  paso tracking_bf     $PY scripts/14_tracking.py --dataset bf --todas
  paso val_tracking    $PY scripts/15_validacion_tracking.py
  VBF=$(json resultados/v2/validacion_tracking/eleccion.json elegida)   # dist_tam
  echo "$VBF" > resultados/v2/variante_bf.txt
  paso stats_bf        $PY scripts/16_estadisticas.py --dataset bf --variante "$VBF"
  paso stats_sirdna    $PY scripts/16_estadisticas.py --dataset sirdna --variante dist
  paso inferencia_bf   $PY scripts/17_inferencia.py --bf "$VBF" --sirdna dist
  paso linajes         $PY scripts/26_linajes.py
  paso figuras_bf      $PY scripts/18_figuras.py --bf "$VBF" --sirdna dist
  paso animaciones_bf  $PY scripts/22_animaciones.py bf
fi

if [[ "$PARTE" == "todo" || "$PARTE" == "camad" ]]; then
  paso preparar_camad    $PY scripts/10_preparar_camad.py
  paso ajuste_seg_camad  $PY scripts/12_ajuste_segmentacion_camad.py
  paso seg_camad         $PY scripts/11_segmentar_gpu.py --dataset camad
  paso aplicar_seg_camad $PY scripts/12_ajuste_segmentacion_camad.py --aplicar
  # variante propia de CAMAD: cierre de huecos largos, sin penalización de tamaño
  echo "camad_huecos" > resultados/v2/variante_camad.txt
  paso tracking_camad    $PY scripts/14_tracking.py --dataset camad --variante camad_huecos
  paso stats_camad       $PY scripts/16_estadisticas.py --dataset camad --variante camad_huecos
  paso inferencia_camad  $PY scripts/17_inferencia.py --camad camad_huecos
  paso figuras_camad     $PY scripts/18_figuras.py --camad camad_huecos
  paso animaciones_camad $PY scripts/22_animaciones.py camad
fi

if [[ "$PARTE" == "todo" || "$PARTE" == "reporte" ]]; then
  paso simulaciones  $PY scripts/19_simulaciones_metodos.py
  paso whad          $PY scripts/21_whad.py
  paso rendimiento   $PY scripts/24_rendimiento.py
  # formas y búsqueda exploratoria (usan tracks de BF y CAMAD ya calculados)
  paso morfoespacio  $PY scripts/27_morfoespacio.py
  paso descubrimiento $PY scripts/28_descubrimiento.py
  paso arquetipos    $PY scripts/29_arquetipos_forma.py
  paso hallazgos     $PY scripts/30_hallazgos.py
  paso figuras_todas $PY scripts/18_figuras.py --bf "$(cat resultados/v2/variante_bf.txt)" --sirdna dist \
                        --camad "$(cat resultados/v2/variante_camad.txt)"
  paso reporte_html  $PY scripts/23_reporte_html.py
fi
echo "[$(date +%H:%M:%S)] FIN $PARTE" | tee -a "$LOG/orquestador.log"
# Nota: scripts/25_cellpose3_vs_sam.py (comparación con Cellpose 3) necesita
# instalar Cellpose 3 aparte (ver su encabezado) y no forma parte de la corrida.
