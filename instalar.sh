#!/usr/bin/env bash
# Instalador del pipeline de migración celular (Linux).
#
#   ./instalar.sh            detecta la GPU sola
#   ./instalar.sh --cuda     fuerza NVIDIA (CUDA)
#   ./instalar.sh --rocm     fuerza AMD (ROCm)
#   ./instalar.sh --cpu      sin GPU (unas 40× más lento)
#
# Crea venv/ en esta carpeta, instala PyTorch con la variante correcta,
# el resto de dependencias, descarga el modelo Cellpose-SAM (~1.2 GB) y
# verifica que todo funcione. Se puede volver a correr sin problema.
set -euo pipefail
cd "$(dirname "$0")"

TORCH=2.14.0
TORCHVISION=0.29.0
PY=${PYTHON:-python3}

variante=""
case "${1:-}" in
  --cuda) variante=cuda ;; --rocm) variante=rocm ;; --cpu) variante=cpu ;;
  "") ;;
  *) echo "opción desconocida: $1"; exit 1 ;;
esac

if [[ -z "$variante" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then variante=cuda
  elif command -v rocminfo >/dev/null 2>&1 || [[ -d /opt/rocm ]]; then variante=rocm
  else variante=cpu; fi
fi

case "$variante" in
  cuda)
    # CUDA 13 necesita driver >= 580; con uno más viejo se usa CUDA 12.6.
    # Las dos variantes funcionan con la RTX 3070 (Ampere).
    cuda_drv=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9]*\)\..*/\1/p' | head -1)
    if [[ "${cuda_drv:-0}" -ge 13 ]]; then idx=cu130; else idx=cu126; fi
    echo ">> GPU NVIDIA: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)  ->  PyTorch $idx" ;;
  rocm) idx=rocm7.2; echo ">> GPU AMD (ROCm)  ->  PyTorch $idx" ;;
  cpu)  idx=cpu;     echo ">> Sin GPU compatible  ->  PyTorch CPU (lento)" ;;
esac

$PY - <<'PYV'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Hace falta Python 3.10 o más nuevo (hay {sys.version.split()[0]}). "
             "Instalar uno más nuevo o indicar cuál usar: PYTHON=python3.12 ./instalar.sh")
PYV

[[ -d venv ]] || $PY -m venv venv
venv/bin/python -m pip install --upgrade pip wheel
venv/bin/python -m pip install "torch==$TORCH" "torchvision==$TORCHVISION" \
    --index-url "https://download.pytorch.org/whl/$idx"
venv/bin/python -m pip install -r requisitos/base.txt
echo ">> verificando la instalación y descargando el modelo (la primera vez tarda)…"
venv/bin/python scripts/verificar_instalacion.py
echo
echo "Listo. Para abrir la interfaz:  ./iniciar_interfaz.sh"
