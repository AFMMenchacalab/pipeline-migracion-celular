"""
Interfaz gráfica del pipeline para el microscopio propio (corre en la PC).

Junta en un solo programa, con una página web local:
  1. Recepción de las imágenes que manda la Raspberry Pi (lib/receptor.py).
  2. Segmentación en vivo con Cellpose-SAM, ciclo por ciclo (lib/en_vivo.py),
     con la última imagen segmentada a la vista.
  3. Tracking y estadística de cada experimento (lib/propio.py): trayectorias
     sobre la imagen, MSD, rapidez, direccionalidad, α y entropía.

Por qué una página web local y no una ventana de escritorio: funciona igual
en Linux y en Windows sin instalar nada gráfico (solo el navegador), se
puede mirar desde otra computadora de la red con --publico, y usa la misma
biblioteca estándar de Python que el receptor.

Uso:
  venv/bin/python scripts/33_interfaz.py            # abre http://127.0.0.1:8080
  venv/bin/python scripts/33_interfaz.py --publico  # visible en la red local
"""
import argparse
import csv
import json
import re
import sys
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT, cache_dir  # noqa: E402
from lib import en_vivo as EV  # noqa: E402
from lib.receptor import (Receptor, cargar_o_crear_codigo, codigo_nuevo,  # noqa: E402
                          guardar_codigo, ips_locales)

ESTATICOS = Path(__file__).resolve().parent / "interfaz"
RE_NOMBRE = re.compile(r"^[A-Za-z0-9_\-]{1,80}$")
RE_CAM = re.compile(r"^cam\d$")
IMAGENES = {"ultima": ("masks", "ultima.png"), "trayectorias": ("analisis", "trayectorias.png"),
            "msd": ("analisis", "msd.png")}


class Estado:
    """Todo lo que corre en segundo plano y lo que la página consulta."""

    def __init__(self, entrada_dir, salida_dir=None):
        self.entrada_dir = Path(entrada_dir)
        if salida_dir:
            self.masks = Path(salida_dir) / "masks"
            self.analisis = Path(salida_dir) / "analisis"
            self.masks.mkdir(parents=True, exist_ok=True)
            self.analisis.mkdir(parents=True, exist_ok=True)
        else:
            self.masks = cache_dir("masks", "propio")
            self.analisis = cache_dir("analisis", "propio")
        self.receptor = Receptor()
        # código de emparejamiento: se genera una vez y se guarda, así la Pi
        # no tiene que volver a escribirlo cada vez que se reinicia la PC
        self.ruta_codigo = cache_dir() / "receptor.json"
        self.token = cargar_o_crear_codigo(self.ruta_codigo)
        self.log = deque(maxlen=300)
        self.lock = threading.Lock()
        self.seg = {"estado": "detenido", "entrada": "suma", "reducir": 2,
                    "flow_threshold": 0.4, "cellprob_threshold": 0.0,
                    "hechos": 0, "segundos": 0.0, "cola": 0, "ultimo": None, "error": None}
        self._seg_activo = threading.Event()
        self._modelo = None
        self.trabajos = {}              # (exp, cam) -> estado del análisis
        self.disp = EV.dispositivo()

    def anotar(self, texto):
        linea = f"{time.strftime('%H:%M:%S')}  {texto}"
        print(linea, flush=True)
        self.log.append(linea)

    # ---------- segmentación ----------
    def iniciar_segmentacion(self, opciones):
        with self.lock:
            if self.seg["estado"] in ("cargando", "activo"):
                return
            for k in ("entrada", "reducir", "flow_threshold", "cellprob_threshold"):
                if k in opciones:
                    self.seg[k] = opciones[k]
            self.seg["estado"] = "cargando"
            self.seg["error"] = None
            self._seg_activo.set()
        threading.Thread(target=self._bucle_segmentacion, daemon=True).start()

    def detener_segmentacion(self):
        self._seg_activo.clear()

    def _bucle_segmentacion(self):
        try:
            if self._modelo is None:
                self.anotar(f"cargando Cellpose-SAM ({self.disp['backend']} {self.disp['nombre']})…")
                self._modelo = EV.cargar_modelo()
            self.seg["estado"] = "activo"
            self.anotar(f"segmentación activa: entrada={self.seg['entrada']}, reducir={self.seg['reducir']}")
            while self._seg_activo.is_set():
                pend = EV.pendientes(self.entrada_dir, self.masks)
                self.seg["cola"] = len(pend)
                for exp, cam, fecha, fotos, sufijos, destino in pend:
                    if not self._seg_activo.is_set():
                        break
                    try:
                        n, dt = EV.segmentar_ciclo(
                            self._modelo, exp, cam, fecha, fotos, sufijos, destino,
                            entrada=self.seg["entrada"], reducir=int(self.seg["reducir"]),
                            flow_threshold=float(self.seg["flow_threshold"]),
                            cellprob_threshold=float(self.seg["cellprob_threshold"]))
                    except Exception as e:
                        self.anotar(f"error en {exp}/{cam}/{fecha}: {e}")
                        continue
                    self.seg["hechos"] += 1
                    self.seg["segundos"] += dt
                    self.seg["cola"] = max(0, self.seg["cola"] - 1)
                    self.seg["ultimo"] = {"exp": exp, "cam": cam, "fecha": fecha, "celulas": n,
                                          "segundos": round(dt, 2), "t": time.time()}
                    self.anotar(f"{exp}/{cam}/{fecha}: {n} células en {dt:.1f} s")
                time.sleep(2)
        except Exception as e:
            self.seg["error"] = str(e)
            self.anotar(f"la segmentación se detuvo por un error: {e}")
            traceback.print_exc()
        finally:
            self.seg["estado"] = "detenido"
            self.anotar("segmentación detenida")

    # ---------- análisis (tracking + estadística) ----------
    def analizar(self, exp, cam):
        clave = (exp, cam)
        t = self.trabajos.get(clave)
        if t and t["estado"] == "corriendo":
            return
        self.trabajos[clave] = {"estado": "corriendo", "paso": "iniciando", "t": time.time()}

        def trabajar():
            from lib import propio
            try:
                def paso(p):
                    self.trabajos[clave]["paso"] = p
                r = propio.analizar(self.masks, exp, cam, self.analisis / exp / cam,
                                    entrada_dir=self.entrada_dir, progreso=paso)
                self.trabajos[clave] = {"estado": "listo", "resumen": r, "t": time.time()}
                self.anotar(f"análisis {exp}/{cam}: {r['trayectorias_analizadas']} trayectorias")
            except Exception as e:
                self.trabajos[clave] = {"estado": "error", "error": str(e), "t": time.time()}
                self.anotar(f"análisis {exp}/{cam}: error ({e})")
                traceback.print_exc()

        threading.Thread(target=trabajar, daemon=True).start()

    def resumen_analisis(self, exp, cam):
        t = self.trabajos.get((exp, cam))
        if t:
            return t
        r = self.analisis / exp / cam / "resumen.json"
        if r.exists():
            return {"estado": "listo", "resumen": json.loads(r.read_text(encoding="utf-8")), "t": r.stat().st_mtime}
        return {"estado": "sin analizar"}

    # ---------- panorama ----------
    def experimentos(self):
        exps = {}
        for exp, cam, fecha, fotos, suf, completo in EV.ciclos(self.entrada_dir):
            c = exps.setdefault(exp, {}).setdefault(cam, {"ciclos": 0, "incompletos": 0,
                                                          "segmentados": 0, "ultima_fecha": None})
            c["ciclos" if completo else "incompletos"] += 1
            c["ultima_fecha"] = fecha
        if self.masks.is_dir():
            for exp_dir in self.masks.iterdir():
                for cam_dir in exp_dir.glob("cam*") if exp_dir.is_dir() else []:
                    c = exps.setdefault(exp_dir.name, {}).setdefault(
                        cam_dir.name, {"ciclos": 0, "incompletos": 0, "segmentados": 0, "ultima_fecha": None})
                    c["segmentados"] = sum(1 for _ in cam_dir.glob("*.npz"))
        salida = []
        for exp in sorted(exps, reverse=True):
            meta = {}
            mj = self.entrada_dir / exp / "experimento.json"
            if mj.exists():
                try:
                    meta = json.loads(mj.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            camaras = {}
            for cam, c in sorted(exps[exp].items()):
                a = self.resumen_analisis(exp, cam)
                c["analisis"] = a.get("estado")
                camaras[cam] = c
            salida.append({"nombre": exp, "meta": meta, "camaras": camaras})
        return salida

    def estado(self):
        seg = dict(self.seg)
        seg["promedio_s"] = round(seg["segundos"] / seg["hechos"], 2) if seg["hechos"] else None
        return {"dispositivo": self.disp, "ips": ips_locales(),
                "entrada_dir": str(self.entrada_dir), "masks_dir": str(self.masks),
                "analisis_dir": str(self.analisis),
                "receptor": self.receptor.estado(), "codigo": self.token,
                "segmentacion": seg, "experimentos": self.experimentos(),
                "log": list(self.log)[-80:]}


def crear_manejador(est):

    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _enviar(self, codigo, cuerpo, tipo="application/json"):
            if isinstance(cuerpo, (dict, list)):
                cuerpo = json.dumps(cuerpo, ensure_ascii=False).encode()
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def _exp_cam(self, q):
            exp, cam = q.get("exp", [""])[0], q.get("cam", [""])[0]
            if RE_NOMBRE.match(exp) and RE_CAM.match(cam):
                return exp, cam
            return None, None

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                return self._enviar(200, (ESTATICOS / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/estado":
                return self._enviar(200, est.estado())
            if url.path == "/api/imagen":
                exp, cam = self._exp_cam(q)
                tipo = q.get("tipo", [""])[0]
                if not exp or tipo not in IMAGENES:
                    return self._enviar(400, {"error": "parámetros inválidos"})
                base, nombre = IMAGENES[tipo]
                raiz = est.masks if base == "masks" else est.analisis
                ruta = raiz / exp / cam / nombre
                if not ruta.is_file():
                    return self._enviar(404, {"error": "todavía no hay imagen"})
                return self._enviar(200, ruta.read_bytes(), "image/png")
            if url.path == "/api/serie":
                exp, cam = self._exp_cam(q)
                if not exp:
                    return self._enviar(400, {"error": "parámetros inválidos"})
                reg = est.masks / exp / "segmentacion.csv"
                filas = []
                if reg.exists():
                    with open(reg, encoding="utf-8") as f:
                        for r in csv.DictReader(f):
                            if r["camara"] == cam:
                                filas.append({"fecha": r["fecha"], "celulas": int(r["celulas"]),
                                              "segundos": float(r["segundos"])})
                return self._enviar(200, {"filas": filas})
            if url.path == "/api/analisis":
                exp, cam = self._exp_cam(q)
                if not exp:
                    return self._enviar(400, {"error": "parámetros inválidos"})
                a = dict(est.resumen_analisis(exp, cam))
                pc = est.analisis / exp / cam / "por_celula.csv"
                if a.get("estado") == "listo" and pc.exists():
                    with open(pc, encoding="utf-8") as f:
                        filas = list(csv.DictReader(f))
                    filas.sort(key=lambda r: -float(r["duracion_min"]))
                    a["celulas"] = filas[:100]
                return self._enviar(200, a)
            return self._enviar(404, {"error": "no encontrado"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0") or 0)
            try:
                datos = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._enviar(400, {"error": "JSON inválido"})
            ruta = urlparse(self.path).path
            if ruta == "/api/receptor":
                if datos.get("accion") == "nuevo_codigo":
                    if est.receptor.activo():
                        return self._enviar(400, {"error": "Detén la recepción antes de cambiar el código"})
                    est.token = codigo_nuevo()
                    guardar_codigo(est.ruta_codigo, est.token)
                    est.anotar("código de emparejamiento nuevo: hay que escribirlo otra vez en la Pi")
                    return self._enviar(200, {"codigo": est.token})
                if datos.get("accion") == "iniciar":
                    try:
                        est.receptor.iniciar(est.entrada_dir, est.token, int(datos.get("puerto") or 8765),
                                             info_extra={"gpu": est.disp.get("nombre", "")})
                    except OSError as e:
                        return self._enviar(400, {"error": f"no se pudo abrir el puerto: {e}"})
                    est.anotar(f"recepción activa en el puerto {est.receptor.puerto}; "
                               f"código para la Pi: {est.token}")
                    if est.receptor.aviso_descubrimiento:
                        est.anotar(est.receptor.aviso_descubrimiento)
                else:
                    est.receptor.detener()
                    est.anotar("recepción detenida")
                return self._enviar(200, est.receptor.estado())
            if ruta == "/api/segmentacion":
                if datos.get("accion") == "iniciar":
                    op = {}
                    if datos.get("entrada") in EV.ENTRADAS:
                        op["entrada"] = datos["entrada"]
                    for k, tipo, lo, hi in (("reducir", int, 1, 8), ("flow_threshold", float, 0, 3),
                                            ("cellprob_threshold", float, -6, 6)):
                        if k in datos:
                            try:
                                op[k] = min(hi, max(lo, tipo(datos[k])))
                            except (TypeError, ValueError):
                                pass
                    est.iniciar_segmentacion(op)
                else:
                    est.detener_segmentacion()
                return self._enviar(200, est.seg)
            if ruta == "/api/analizar":
                exp, cam = datos.get("exp", ""), datos.get("cam", "")
                if not (RE_NOMBRE.match(exp) and RE_CAM.match(cam)):
                    return self._enviar(400, {"error": "parámetros inválidos"})
                est.analizar(exp, cam)
                return self._enviar(200, {"estado": "corriendo"})
            if ruta == "/api/carpeta":
                if est.receptor.activo() or est.seg["estado"] != "detenido":
                    return self._enviar(400, {"error": "Detén la recepción y la segmentación antes de cambiar de carpeta"})
                p = Path(str(datos.get("ruta", ""))).expanduser()
                if not p.is_dir():
                    return self._enviar(400, {"error": f"No existe la carpeta {p}"})
                est.entrada_dir = p.resolve()
                est.anotar(f"carpeta de entrada: {est.entrada_dir}")
                return self._enviar(200, {"entrada_dir": str(est.entrada_dir)})
            return self._enviar(404, {"error": "no encontrado"})

    return Manejador


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8080, help="Puerto de la interfaz")
    ap.add_argument("--entrada-dir", type=Path, default=ROOT / "datasets" / "microscopio_propio")
    ap.add_argument("--salida-dir", type=Path, default=None,
                    help="Carpeta para máscaras y resultados (por defecto ~/microscopio_cache)")
    ap.add_argument("--publico", action="store_true",
                    help="Aceptar conexiones de otras computadoras de la red (por defecto solo esta PC)")
    ap.add_argument("--no-abrir", action="store_true", help="No abrir el navegador")
    args = ap.parse_args()
    args.entrada_dir.mkdir(parents=True, exist_ok=True)
    est = Estado(args.entrada_dir, args.salida_dir)
    host = "0.0.0.0" if args.publico else "127.0.0.1"
    srv = ThreadingHTTPServer((host, args.puerto), crear_manejador(est))
    url = f"http://127.0.0.1:{args.puerto}"
    est.anotar(f"interfaz en {url}  (GPU: {est.disp['backend']} {est.disp['nombre']})")
    if not args.no_abrir:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    est.detener_segmentacion()
    est.receptor.detener()


if __name__ == "__main__":
    main()
