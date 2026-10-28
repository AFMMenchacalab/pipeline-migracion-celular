#!/usr/bin/env bash
# Abre la interfaz gráfica (http://127.0.0.1:8770). Opciones: ver
#   venv/bin/python scripts/33_interfaz.py --help
cd "$(dirname "$0")"
exec venv/bin/python scripts/33_interfaz.py "$@"
