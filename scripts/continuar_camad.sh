#!/usr/bin/env bash
# Continuación de la parte CAMAD de correr_v2.sh (2026-09-24): usa la variante
# de tracking "camad_huecos" (ver 14_tracking.py) en lugar de la elegida en BF.
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
LOG=resultados/v2/logs
paso() {
  local n="$1"; shift
  echo "[$(date +%H:%M:%S)] >>> $n" | tee -a "$LOG/orquestador.log"
  if "$@" > "$LOG/$n.log" 2>&1; then echo "[$(date +%H:%M:%S)] OK  $n" | tee -a "$LOG/orquestador.log"
  else echo "[$(date +%H:%M:%S)] FALLA $n (ver $LOG/$n.log)" | tee -a "$LOG/orquestador.log"; return 1; fi
}
while pgrep -f "scripts/1[1]_segmentar_gpu.py --dataset camad" > /dev/null; do sleep 30; done
V=camad_huecos
echo "$V" > resultados/v2/variante_camad.txt
paso aplicar_seg_camad $PY scripts/12_ajuste_segmentacion_camad.py --aplicar
paso tracking_camad    $PY scripts/14_tracking.py --dataset camad --variante $V
paso stats_camad       $PY scripts/16_estadisticas.py --dataset camad --variante $V
paso inferencia_camad  $PY scripts/17_inferencia.py --camad $V
paso figuras_camad     $PY scripts/18_figuras.py --bf dist_tam --sirdna dist --camad $V
paso animaciones_camad $PY scripts/22_animaciones.py camad
paso reporte_html      $PY scripts/23_reporte_html.py
echo "[$(date +%H:%M:%S)] FIN camad" | tee -a "$LOG/orquestador.log"
