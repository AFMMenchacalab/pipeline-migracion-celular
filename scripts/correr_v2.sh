#!/usr/bin/env bash
# Orquestador del pipeline v2: corre en orden todo lo que depende de la
# segmentación en GPU. Cada paso deja su log en resultados/v2/logs/.
# Reanudable: cada script saltea lo ya hecho (o es barato de repetir).
#
#   bash scripts/correr_v2.sh            # todo
#   bash scripts/correr_v2.sh bf         # solo la parte BF + núcleos
#   bash scripts/correr_v2.sh camad      # solo la parte CAMAD
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python
LOG=resultados/v2/logs
mkdir -p "$LOG"
PARTE="${1:-todo}"

esperar() {  # espera a que termine cualquier proceso cuyo comando matchee el patrón
  while pgrep -f "$1" > /dev/null; do sleep 30; done
}
paso() {  # paso <nombre> <comando...>
  local n="$1"; shift
  echo "[$(date +%H:%M:%S)] >>> $n" | tee -a "$LOG/orquestador.log"
  if "$@" > "$LOG/$n.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] OK  $n" | tee -a "$LOG/orquestador.log"
  else
    echo "[$(date +%H:%M:%S)] FALLA $n (ver $LOG/$n.log)" | tee -a "$LOG/orquestador.log"
    return 1
  fi
}
json() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

if [[ "$PARTE" == "todo" || "$PARTE" == "bf" ]]; then
  esperar "scripts/1[1]_segmentar_gpu.py --dataset bf"
  esperar "scripts/1[1]_segmentar_gpu.py --dataset sirdna"
  paso ajuste_seg_bf   $PY scripts/13_ajuste_segmentacion_bf.py --aplicar
  paso tracking_sirdna $PY scripts/14_tracking.py --dataset sirdna --variante dist
  paso tracking_bf     $PY scripts/14_tracking.py --dataset bf --todas
  paso val_tracking    $PY scripts/15_validacion_tracking.py
  VBF=$(json resultados/v2/validacion_tracking/eleccion.json elegida)
  echo "variante BF elegida: $VBF" | tee -a "$LOG/orquestador.log"
  echo "$VBF" > resultados/v2/variante_bf.txt
  paso stats_bf        $PY scripts/16_estadisticas.py --dataset bf --variante "$VBF"
  paso stats_sirdna    $PY scripts/16_estadisticas.py --dataset sirdna --variante dist
  paso inferencia_bf   $PY scripts/17_inferencia.py --bf "$VBF" --sirdna dist
  paso figuras_bf      $PY scripts/18_figuras.py --bf "$VBF" --sirdna dist
fi

if [[ "$PARTE" == "todo" || "$PARTE" == "camad" ]]; then
  esperar "scripts/1[1]_segmentar_gpu.py --dataset camad"
  esperar "scripts/1[2]_ajuste_segmentacion_camad.py"
  paso aplicar_seg_camad $PY scripts/12_ajuste_segmentacion_camad.py --aplicar
  # misma familia de tracking que la elegida en BF (no hay referencia nuclear en CAMAD):
  # esperar (máx. 3 h) a que la parte BF la haya elegido
  for i in $(seq 360); do [[ -f resultados/v2/variante_bf.txt ]] && break; sleep 30; done
  VBF=$(cat resultados/v2/variante_bf.txt 2>/dev/null || echo dist_tam)
  [[ "$VBF" == "v1" ]] && VBF=dist
  echo "$VBF" > resultados/v2/variante_camad.txt
  paso tracking_camad  $PY scripts/14_tracking.py --dataset camad --variante "$VBF"
  paso stats_camad     $PY scripts/16_estadisticas.py --dataset camad --variante "$VBF"
  paso inferencia_camad $PY scripts/17_inferencia.py --camad "$VBF"
  paso figuras_camad   $PY scripts/18_figuras.py --camad "$VBF"
fi
echo "[$(date +%H:%M:%S)] FIN $PARTE" | tee -a "$LOG/orquestador.log"
