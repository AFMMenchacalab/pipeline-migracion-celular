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
import hashlib
import hmac
import json
import os
import re
import shutil
import socket
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT  # noqa: E402

DESTINO_DEF = ROOT / "datasets" / "microscopio_propio"
RE_EXP = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
RE_CAM = re.compile(r"^(cam\d|_)$")
RE_ARCH = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,119}$")
MAX_BYTES = 256 * 1024 * 1024


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def crear_manejador(destino, token):

    class Manejador(BaseHTTPRequestHandler):
        server_version = "ReceptorMicroscopio/1.0"

        def log_message(self, fmt, *args):     # el log propio es más legible
            pass

        def _json(self, codigo, datos):
            cuerpo = json.dumps(datos, ensure_ascii=False).encode()
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def _autorizado(self):
            if not token:
                return True
            return hmac.compare_digest(self.headers.get("X-Token", ""), token)

        def _ruta(self, partes):
            """[experimento, camara, archivo] -> Path validado dentro de destino."""
            if len(partes) != 3:
                return None
            exp, cam, arch = (unquote(p) for p in partes)
            if not (RE_EXP.match(exp) and RE_CAM.match(cam) and RE_ARCH.match(arch)):
                return None
            carpeta = destino / exp if cam == "_" else destino / exp / cam
            ruta = (carpeta / arch).resolve()
            if destino.resolve() not in ruta.parents:
                return None
            return ruta

        def do_GET(self):
            if not self._autorizado():
                return self._json(401, {"error": "clave incorrecta"})
            url = urlparse(self.path)
            partes = [p for p in url.path.split("/") if p]
            if partes == ["salud"]:
                libre = shutil.disk_usage(destino).free
                return self._json(200, {"ok": True, "equipo": socket.gethostname(),
                                        "destino": str(destino),
                                        "libre_gb": round(libre / 1e9, 1)})
            if partes and partes[0] == "existe":
                ruta = self._ruta(partes[1:])
                if ruta is None:
                    return self._json(400, {"error": "ruta inválida"})
                sha = parse_qs(url.query).get("sha256", [""])[0]
                if ruta.is_file() and (not sha or sha256_archivo(ruta) == sha):
                    return self._json(200, {"existe": True})
                return self._json(404, {"existe": False})
            return self._json(404, {"error": "no encontrado"})

        def do_PUT(self):
            if not self._autorizado():
                return self._json(401, {"error": "clave incorrecta"})
            partes = [p for p in urlparse(self.path).path.split("/") if p]
            if not partes or partes[0] != "subir":
                return self._json(404, {"error": "no encontrado"})
            ruta = self._ruta(partes[1:])
            if ruta is None:
                return self._json(400, {"error": "ruta inválida"})
            try:
                n = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BYTES:
                return self._json(411 if n < 0 else 413, {"error": "tamaño inválido"})
            sha_esperado = self.headers.get("X-Sha256", "")
            ruta.parent.mkdir(parents=True, exist_ok=True)
            tmp = ruta.with_name(f".{ruta.name}.parcial")
            h = hashlib.sha256()
            t0 = time.time()
            with open(tmp, "wb") as f:
                restante = n
                while restante > 0:
                    bloque = self.rfile.read(min(1 << 20, restante))
                    if not bloque:
                        break
                    f.write(bloque)
                    h.update(bloque)
                    restante -= len(bloque)
            if restante > 0 or (sha_esperado and h.hexdigest() != sha_esperado):
                tmp.unlink(missing_ok=True)
                print(f"[receptor] RECHAZADO {ruta.relative_to(destino)} (incompleto o hash distinto)",
                      flush=True)
                return self._json(422, {"error": "archivo incompleto o hash distinto"})
            os.replace(tmp, ruta)
            dt = time.time() - t0
            print(f"[receptor] {time.strftime('%H:%M:%S')}  {ruta.relative_to(destino)}  "
                  f"{n / 1e6:.1f} MB en {dt:.1f} s", flush=True)
            return self._json(201, {"guardado": str(ruta.relative_to(destino))})

    return Manejador


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
    try:
        ip = socket.gethostbyname(socket.gethostname())
    except OSError:
        ip = "<IP de esta PC>"
    print(f"[receptor] escuchando en http://{ip}:{args.puerto}  ->  {args.destino}", flush=True)
    print("[receptor] Ctrl+C para detener", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[receptor] detenido", flush=True)


if __name__ == "__main__":
    main()
