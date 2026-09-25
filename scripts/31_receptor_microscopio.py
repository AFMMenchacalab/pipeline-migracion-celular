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
  ../venv/bin/python scripts/31_receptor_microscopio.py
  (muestra un código de 6 dígitos; en la Pi: «Buscar en la red», elegir esta
  PC y escribir el código)
  Si la PC tiene cortafuegos, abrir en la red local el puerto 8765/tcp
  (imágenes) y el 8766/udp (búsqueda automática desde la Pi).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT, cache_dir  # noqa: E402
from lib.receptor import Receptor, cargar_o_crear_codigo, ips_locales  # noqa: E402

DESTINO_DEF = ROOT / "datasets" / "microscopio_propio"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--destino", type=Path, default=DESTINO_DEF)
    ap.add_argument("--token", default=os.environ.get("MICROSCOPIO_TOKEN", ""),
                    help="Clave propia en vez del código de 6 dígitos guardado (o variable MICROSCOPIO_TOKEN)")
    ap.add_argument("--sin-token", action="store_true",
                    help="Aceptar envíos sin clave (solo para pruebas)")
    args = ap.parse_args()
    token = "" if args.sin_token else (args.token or cargar_o_crear_codigo(cache_dir() / "receptor.json"))
    rec = Receptor()
    rec.iniciar(args.destino, token, args.puerto)
    ips = " o ".join(f"http://{ip}:{args.puerto}" for ip in ips_locales()) or "<IP de esta PC>"
    print(f"[receptor] recibiendo en el puerto {args.puerto}  ->  {args.destino}", flush=True)
    if token:
        print(f"[receptor] CÓDIGO PARA LA PI: {token}", flush=True)
    if rec.anunciador is not None:
        print("[receptor] en la Pi: «Buscar en la red», elegir esta PC y escribir el código", flush=True)
    else:
        print(f"[receptor] {rec.aviso_descubrimiento}", flush=True)
    print(f"[receptor] dirección manual: {ips}", flush=True)
    print("[receptor] Ctrl+C para detener", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        rec.detener()
        print("\n[receptor] detenido", flush=True)


if __name__ == "__main__":
    main()
