"""
Receptor de imágenes del microscopio propio (corre en la PC).

La Raspberry Pi (MicroscopeOS, core/envio.py) manda cada imagen apenas la
guarda. Este programa la recibe y la deja en una carpeta que vigila
32_segmentar_en_vivo.py, que la segmenta en cuanto llega.

Por qué así:
  - Solo biblioteca estándar (http.server): no hace falta instalar nada.
  - Cada archivo se escribe primero como temporal (.nombre.parcial), se
    verifica su SHA-256 contra el que calculó la Pi y recién entonces se
    renombra. El segmentador nunca ve una imagen a medias.
  - Idempotente: si la Pi reenvía algo que ya está (tras un corte de red),
    GET /existe responde 200 y no se vuelve a transferir.
  - Clave compartida (X-Token) para que no cualquier equipo de la red
    pueda escribir en la PC. Los nombres se validan con expresiones
    regulares: no se aceptan rutas arbitrarias.

Rutas:
  GET /salud                                   -> {"ok": true, ...}
  GET /existe/<experimento>/<camara>/<archivo>?sha256=...   -> 200 | 404
  PUT /subir/<experimento>/<camara>/<archivo>  (X-Sha256, X-Token)

Uso:
  ../venv/bin/python scripts/31_receptor_microscopio.py --token MI_CLAVE
  (en la Pi, panel "Envío a computadora": http://<IP de esta PC>:8765 y la
  misma clave). Si la PC tiene cortafuegos, abrir el puerto 8765/tcp en la
  red local.
"""
import argparse
import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT  # noqa: E402
from lib.receptor import crear_manejador, ips_locales  # noqa: E402

DESTINO_DEF = ROOT / "datasets" / "microscopio_propio"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--destino", type=Path, default=DESTINO_DEF)
    ap.add_argument("--token", default=os.environ.get("MICROSCOPIO_TOKEN", ""),
                    help="Clave compartida con la Pi (o variable MICROSCOPIO_TOKEN)")
    ap.add_argument("--sin-token", action="store_true",
                    help="Aceptar envíos sin clave (solo para pruebas)")
    args = ap.parse_args()
    if not args.token and not args.sin_token:
        ap.error("falta --token (o --sin-token para pruebas)")
    args.destino.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", args.puerto), crear_manejador(args.destino, args.token))
    ip = (ips_locales() or ["<IP de esta PC>"])[0]
    print(f"[receptor] escuchando en http://{ip}:{args.puerto}  ->  {args.destino}", flush=True)
    print("[receptor] Ctrl+C para detener", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[receptor] detenido", flush=True)


if __name__ == "__main__":
    main()
