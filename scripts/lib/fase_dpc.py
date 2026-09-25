"""
Reconstrucción de fase cuantitativa a partir de las 4 fotos DPC (L, R, T, B)
del microscopio propio, método de Tian y Waller (Optics Express 23:11394,
2015): funciones de transferencia de objeto débil + deconvolución de
Tikhonov de los dos ejes a la vez.

Qué hace, en palabras:
  1. Cada foto se normaliza por su fondo (la iluminación de cada media
     matriz es distinta y tiene viñeteado).
  2. DPC_LR = (L − R)/(L + R) y DPC_TB = (T − B)/(T + B): la "pendiente" de
     la fase en cada eje.
  3. Con la geometría real (posición de cada LED, distancia a la muestra,
     apertura numérica y longitud de onda) se calcula cómo transforma el
     sistema la fase en DPC (función de transferencia de cada eje).
  4. Se invierten los dos ejes juntos en el espacio de Fourier con
     regularización (Tikhonov): el resultado es la fase φ en radianes, la
     "montaña" de cada célula sobre un fondo en cero.

Geometría del montaje (valores por defecto = microscopio propio):
  objetivo 20×/0.40 (160/0.17); cámara 0.2159 µm/px en la muestra;
  matriz Waveshare ESP32-S3-Matrix, 8×8 WS2812B, paso ~2.70 mm (medido
  sobre el dibujo del fabricante); patrones del firmware dpc_matrix.ino:
  LEFT = columnas 0–3, RIGHT = 4–7, TOP = filas 0–3, BOTTOM = 4–7.
  Longitud de onda: 0.525 µm (iluminar SOLO con verde, color 00FF00).

Lo que hay que medir en el montaje: `distancia_mm` (de los LEDs a la
muestra). La orientación de la matriz respecto de la cámara (espejo y
rotación) se puede estimar sola con `calibrar_orientacion()` a partir de
una foto con células: se elige la orientación con la que las células salen
como montañas positivas (más densas que el medio).

Limitaciones: aproximación de objeto débil (fases de hasta ~1–2 rad se
recuperan con algo de subestimación, ver la prueba); las frecuencias muy
bajas (fondos suaves) no se recuperan, y no se modelan las aberraciones del
objetivo (tubo distinto de 160 mm, fondo de Petri de plástico).
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class GeometriaDPC:
    distancia_mm: float                 # LEDs -> muestra (medir en el montaje)
    na_obj: float = 0.40
    lambda_um: float = 0.525            # verde de la WS2812B
    um_por_px: float = 0.2159           # en la muestra, sin reducir
    paso_led_mm: float = 2.70
    n_leds: int = 8
    centro_mm: tuple = (0.0, 0.0)       # desplazamiento de la matriz respecto del eje óptico (x, y)
    espejo_x: int = 1                   # ±1: orientación de la matriz vista desde la cámara
    espejo_y: int = 1
    rotacion_grados: float = 0.0
    # Difusor sobre la matriz: cada LED se modela como un cuadrado de luz del
    # tamaño del paso (sub_leds × sub_leds fuentes) en vez de un punto.
    difusor: bool = False
    sub_leds: int = 5
    patrones: dict = field(default_factory=lambda: {
        "_L": lambda r, c: c < 4, "_R": lambda r, c: c >= 4,
        "_T": lambda r, c: r < 4, "_B": lambda r, c: r >= 4})

    def leds(self, sufijo):
        """Frecuencias espaciales de iluminación (ciclos/µm, en ejes de la
        cámara) y pesos de los LEDs de un patrón que caen dentro del cono
        del objetivo. Los de afuera (campo oscuro) se excluyen del modelo."""
        n = self.n_leds
        a = np.deg2rad(self.rotacion_grados)
        frec, pesos, fuera = [], [], 0
        for r in range(n):
            for c in range(n):
                if not self.patrones[sufijo](r, c):
                    continue
                k = self.sub_leds if self.difusor else 1
                offs = (np.arange(k) - (k - 1) / 2) / k * self.paso_led_mm
                dentro = False
                for ox in offs:
                    for oy in offs:
                        x = ((c - (n - 1) / 2) * self.paso_led_mm + ox) * self.espejo_x + self.centro_mm[0]
                        y = ((r - (n - 1) / 2) * self.paso_led_mm + oy) * self.espejo_y + self.centro_mm[1]
                        x, y = x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)
                        rr = np.sqrt(x ** 2 + y ** 2 + self.distancia_mm ** 2)
                        sx, sy = x / rr, y / rr            # senos directores
                        if np.hypot(sx, sy) > self.na_obj:
                            continue
                        dentro = True
                        cos_t = self.distancia_mm / rr
                        frec.append((sx / self.lambda_um, sy / self.lambda_um))
                        pesos.append(cos_t ** 4 / k ** 2)  # emisor ~lambertiano sobre un plano
                if not dentro:
                    fuera += 1
        return np.array(frec), np.array(pesos), fuera

    def na_iluminacion(self):
        """NA máxima de iluminación de los LEDs que quedan dentro del cono."""
        todas = np.vstack([self.leds(s)[0] for s in ("_L", "_R", "_T", "_B")])
        return float(np.hypot(*todas.T).max() * self.lambda_um) if len(todas) else 0.0


def _rejilla(forma, um_px):
    fy = np.fft.fftfreq(forma[0], um_px)
    fx = np.fft.fftfreq(forma[1], um_px)
    return np.meshgrid(fx, fy)


def transferencias(geo, forma, reducir=1):
    """Funciones de transferencia de fase de los ejes LR y TB (complejas,
    imaginarias puras) para imágenes de tamaño `forma`."""
    FX, FY = _rejilla(forma, geo.um_por_px * reducir)
    corte = geo.na_obj / geo.lambda_um
    P = lambda ux, uy: (ux ** 2 + uy ** 2 <= corte ** 2).astype(float)   # noqa: E731

    def H_patron(suf):
        frec, w, _ = geo.leds(suf)
        if not len(frec):
            raise ValueError(f"ningún LED del patrón {suf} cae dentro de NA {geo.na_obj}")
        H = np.zeros(forma, complex)
        for (sx, sy), wj in zip(frec, w):
            H += wj * 1j * (P(FX + sx, FY + sy) - P(sx - FX, sy - FY))
        return H / w.sum()

    H_lr = (H_patron("_L") - H_patron("_R")) / 2
    H_tb = (H_patron("_T") - H_patron("_B")) / 2
    return H_lr, H_tb


def normalizar(img, fondo=None, sigma_px=None):
    """Divide por el fondo: una foto de campo vacío con el mismo patrón o, si
    no hay, una versión muy suavizada de la propia imagen."""
    img = img.astype(np.float64)
    if fondo is not None:
        return img / np.maximum(fondo.astype(np.float64), 1e-6)
    from scipy.ndimage import gaussian_filter
    s = sigma_px or max(img.shape) / 8
    return img / np.maximum(gaussian_filter(img, s), 1e-6)


def reconstruir(fotos, geo, reducir=1, alfa=1e-3, fondos=None, _H=None):
    """fotos: {"_L", "_R", "_T", "_B"} -> fase en radianes (misma forma)."""
    fondos = fondos or {}
    n = {s: normalizar(fotos[s], fondos.get(s)) for s in ("_L", "_R", "_T", "_B")}
    eps = 1e-9
    d_lr = (n["_L"] - n["_R"]) / (n["_L"] + n["_R"] + eps)
    d_tb = (n["_T"] - n["_B"]) / (n["_T"] + n["_B"] + eps)
    forma = d_lr.shape
    H_lr, H_tb = _H if _H is not None else transferencias(geo, forma, reducir)
    D_lr = np.fft.fft2(d_lr - d_lr.mean())
    D_tb = np.fft.fft2(d_tb - d_tb.mean())
    denom = np.abs(H_lr) ** 2 + np.abs(H_tb) ** 2
    reg = alfa * denom.max()
    phi = np.fft.ifft2((np.conj(H_lr) * D_lr + np.conj(H_tb) * D_tb) / (denom + reg)).real
    return phi - np.median(phi)


def calibrar_orientacion(fotos, geo, reducir=1, alfa=1e-3):
    """Prueba las 4 combinaciones de espejo (x, y) y devuelve la geometría
    con la que la fase sale más "positiva y compacta" (asimetría más alta:
    células como montañas sobre fondo plano). Se hace una vez por montaje."""
    from scipy.stats import skew
    import copy
    mejor, puntajes = None, {}
    for ex in (1, -1):
        for ey in (1, -1):
            g = copy.copy(geo)
            g.espejo_x, g.espejo_y = ex, ey
            phi = reconstruir(fotos, g, reducir, alfa)
            p = float(skew(phi.ravel()))
            puntajes[(ex, ey)] = p
            if mejor is None or p > puntajes[(mejor.espejo_x, mejor.espejo_y)]:
                mejor = g
    return mejor, puntajes


# ---------------------------------------------------------------------------
# Configuración guardada (la escribe 34_fase_dpc.py calibrar)
# ---------------------------------------------------------------------------
CAMPOS = ("distancia_mm", "na_obj", "lambda_um", "um_por_px", "paso_led_mm", "espejo_x", "espejo_y",
          "rotacion_grados", "difusor")


def ruta_config():
    from .config import cache_dir
    return cache_dir() / "fase_dpc.json"


def cargar_geometria(ruta=None):
    """(GeometriaDPC, dict con todo lo guardado) o (None, None) si no hay."""
    try:
        c = json.loads(Path(ruta or ruta_config()).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    return GeometriaDPC(**{k: c[k] for k in CAMPOS if k in c}), c


def guardar_geometria(g, extra=None, ruta=None):
    datos = {k: v for k, v in asdict(g).items() if k in CAMPOS}
    datos.update(extra or {})
    Path(ruta or ruta_config()).write_text(json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8")


class ReconstructorEnVivo:
    """Para la segmentación en vivo: carga la calibración una vez y reutiliza
    las funciones de transferencia mientras no cambie el tamaño de imagen."""

    def __init__(self):
        self.geo, self.meta = cargar_geometria()
        if self.geo is None:
            raise RuntimeError("no hay calibración de fase: correr 34_fase_dpc.py calibrar")
        self._H = {}

    def __call__(self, fotos, reducir):
        forma = fotos["_L"].shape
        if (forma, reducir) not in self._H:
            self._H[(forma, reducir)] = transferencias(self.geo, forma, reducir)
        return reconstruir(fotos, self.geo, reducir, _H=self._H[(forma, reducir)])


# ---------------------------------------------------------------------------
# Simulación (para validar la reconstrucción; misma física, sin aproximar)
# ---------------------------------------------------------------------------
def simular(phi, geo, sufijo, reducir=1, absorcion=None):
    """Imagen de intensidad de un objeto de fase `phi` (rad) iluminado por
    el patrón `sufijo`: suma incoherente de la imagen coherente de cada LED
    (pupila binaria, sin aproximación de objeto débil)."""
    forma = phi.shape
    FX, FY = _rejilla(forma, geo.um_por_px * reducir)
    corte = geo.na_obj / geo.lambda_um
    P = (FX ** 2 + FY ** 2 <= corte ** 2)
    t = np.exp(1j * phi) * (1 if absorcion is None else np.exp(-absorcion))
    T = np.fft.fft2(t)
    dfx = FX[0, 1] - FX[0, 0]
    dfy = FY[1, 0] - FY[0, 0]
    frec, w, _ = geo.leds(sufijo)
    I = np.zeros(forma)
    for (sx, sy), wj in zip(frec, w):
        # onda inclinada: el espectro del objeto se corre en +s
        Ts = np.roll(np.roll(T, int(round(sx / dfx)), axis=1), int(round(sy / dfy)), axis=0)
        I += wj * np.abs(np.fft.ifft2(Ts * P)) ** 2
    return I
