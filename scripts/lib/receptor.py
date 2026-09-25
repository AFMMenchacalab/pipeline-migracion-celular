"""
Receptor HTTP de las imágenes que manda la Raspberry Pi (MicroscopeOS,
core/envio.py). Lo usan 31_receptor_microscopio.py (consola) y
33_interfaz.py (interfaz gráfica). Ver el encabezado de
31_receptor_microscopio.py para el protocolo y las decisiones de diseño.

DESCUBRIMIENTO EN LA RED LOCAL
==============================

Para que en la Pi no haya que escribir la IP de la PC: mientras la
recepción está activa, la PC escucha en UDP 8766. La Pi manda por
difusión (broadcast) "MICROSCOPIO_BUSCAR" y cada PC con la recepción
activa responde con su nombre, puerto y GPU. La Pi arma la lista y el
usuario elige. Es el mismo mecanismo que usan impresoras y otros equipos
para "descubrirse"; no requiere instalar nada (biblioteca estándar).

La respuesta NO incluye la clave: la clave es un código de 6 dígitos que
la PC muestra en pantalla y que se escribe en la Pi (emparejamiento). Así,
otro equipo de la red puede ver que la PC existe, pero no mandarle
archivos.

Límite: el broadcast no cruza routers. Si la Pi y la PC están en redes
distintas (o la red Wi-Fi aísla a los clientes), se escribe la IP a mano.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

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


def crear_manejador(destino, token, al_recibir=None):

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
            if al_recibir is not None:
                try:
                    al_recibir(ruta.relative_to(destino), n)
                except Exception:
                    pass
            print(f"[receptor] {time.strftime('%H:%M:%S')}  {ruta.relative_to(destino)}  "
                  f"{n / 1e6:.1f} MB en {dt:.1f} s", flush=True)
            return self._json(201, {"guardado": str(ruta.relative_to(destino))})

    return Manejador




PUERTO_DESCUBRIMIENTO = 8766
MENSAJE_BUSCAR = b"MICROSCOPIO_BUSCAR"


def codigo_nuevo():
    """Código de emparejamiento de 6 dígitos (fácil de escribir en la Pi)."""
    return f"{secrets.randbelow(10 ** 6):06d}"


def cargar_o_crear_codigo(ruta):
    """El código se guarda para que no cambie al reiniciar la PC: la Pi lo
    recuerda y no hay que volver a emparejar."""
    ruta = Path(ruta)
    try:
        c = json.loads(ruta.read_text()).get("codigo", "")
        if re.fullmatch(r"\d{6}", c):
            return c
    except (OSError, json.JSONDecodeError):
        pass
    c = codigo_nuevo()
    guardar_codigo(ruta, c)
    return c


def guardar_codigo(ruta, codigo):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps({"codigo": codigo}))


class Anunciador:
    """Responde a las búsquedas de la Pi mientras la recepción está activa."""

    def __init__(self, info):
        self.info = info            # dict que se manda como respuesta (sin la clave)
        self.sock = None
        self.hilo = None

    def iniciar(self):
        if self.sock is not None:
            return
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", PUERTO_DESCUBRIMIENTO))
        s.settimeout(0.5)
        self.sock = s
        self.hilo = threading.Thread(target=self._bucle, daemon=True)
        self.hilo.start()

    def _bucle(self):
        while self.sock is not None:
            try:
                datos, origen = self.sock.recvfrom(512)
            except socket.timeout:
                continue
            except OSError:
                break
            if datos.startswith(MENSAJE_BUSCAR):
                try:
                    self.sock.sendto(json.dumps(self.info).encode(), origen)
                except OSError:
                    pass

    def detener(self):
        s, self.sock = self.sock, None
        if s is not None:
            s.close()


class Receptor:
    """Receptor en un hilo, para arrancarlo y detenerlo desde la interfaz."""

    def __init__(self):
        self.anunciador = None
        self.aviso_descubrimiento = None
        self.srv = None
        self.hilo = None
        self.puerto = None
        self.destino = None
        self.recibidos = 0
        self.bytes = 0
        self.ultimo = None
        self.ultimo_t = None

    def _contar(self, rel, n):
        self.recibidos += 1
        self.bytes += n
        self.ultimo = str(rel)
        self.ultimo_t = time.time()

    def iniciar(self, destino, token, puerto=8765, info_extra=None):
        if self.activo():
            return
        destino = Path(destino)
        destino.mkdir(parents=True, exist_ok=True)
        self.srv = ThreadingHTTPServer(("0.0.0.0", int(puerto)),
                                       crear_manejador(destino, token, self._contar))
        self.puerto, self.destino = int(puerto), destino
        self.hilo = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.hilo.start()
        self.aviso_descubrimiento = None
        try:
            self.anunciador = Anunciador({"servicio": "receptor-microscopio", "version": 1,
                                          "nombre": socket.gethostname(), "puerto": self.puerto,
                                          **(info_extra or {})})
            self.anunciador.iniciar()
        except OSError as e:          # p. ej. otro receptor ya usa el 8766: se puede seguir con IP a mano
            self.anunciador = None
            self.aviso_descubrimiento = f"descubrimiento no disponible ({e}); usar la IP a mano"

    def detener(self):
        if self.anunciador is not None:
            self.anunciador.detener()
            self.anunciador = None
        if self.srv is not None:
            self.srv.shutdown()
            self.srv.server_close()
            self.srv = None

    def activo(self):
        return self.srv is not None

    def estado(self):
        return {"activo": self.activo(), "puerto": self.puerto,
                "descubrible": self.anunciador is not None,
                "aviso_descubrimiento": self.aviso_descubrimiento,
                "destino": str(self.destino) if self.destino else None,
                "recibidos": self.recibidos, "mb": round(self.bytes / 1e6, 1),
                "ultimo": self.ultimo, "ultimo_t": self.ultimo_t}


def ips_locales():
    """Direcciones IPv4 de esta PC en la red local (para configurar la Pi)."""
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))       # no envía nada; solo elige la interfaz de salida
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))
