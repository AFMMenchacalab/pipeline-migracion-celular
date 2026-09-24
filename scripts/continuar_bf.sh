#!/usr/bin/env bash
# Continuación de la parte BF de correr_v2.sh (se usó una vez, el 2026-09-24,
# tras agregar la regla de desempate de umbrales en 13_ajuste_segmentacion_bf.py).
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
while pgrep -f "scripts/1[4]_tracking.py --dataset sirdna" > /dev/null; do sleep 15; done
paso ajuste_seg_bf_desempate $PY scripts/13_ajuste_segmentacion_bf.py --aplicar
paso tracking_bf     $PY scripts/14_tracking.py --dataset bf --todas
paso val_tracking    $PY scripts/15_validacion_tracking.py
VBF=$($PY -c "import json; print(json.load(open('resultados/v2/validacion_tracking/eleccion.json'))['elegida'])")
echo "variante BF elegida: $VBF" | tee -a "$LOG/orquestador.log"
echo "$VBF" > resultados/v2/variante_bf.txt
paso stats_bf        $PY scripts/16_estadisticas.py --dataset bf --variante "$VBF"
paso stats_sirdna    $PY scripts/16_estadisticas.py --dataset sirdna --variante dist
paso inferencia_bf   $PY scripts/17_inferencia.py --bf "$VBF" --sirdna dist
paso figuras_bf      $PY scripts/18_figuras.py --bf "$VBF" --sirdna dist
paso animaciones_bf  $PY scripts/22_animaciones.py bf
echo "[$(date +%H:%M:%S)] FIN bf" | tee -a "$LOG/orquestador.log"
