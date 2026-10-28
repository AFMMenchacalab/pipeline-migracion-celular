"""
Fase cuantitativa del microscopio propio a partir de las 4 fotos DPC
(ver lib/fase_dpc.py para el método y sus límites).

1) Calibrar una vez por montaje, con un ciclo que tenga células:
     venv/bin/python scripts/34_fase_dpc.py calibrar CARPETA_CAM --fecha 20260926_101500 \
         --distancia-mm 23 [--difusor]
   Detecta cómo está orientada la matriz respecto de la cámara (espejo en x/y),
   guarda la configuración en ~/microscopio_cache/fase_dpc.json y deja
   fase_<fecha>.png para revisarla: las células tienen que verse como
   manchas claras sobre fondo gris parejo.

2) Reconstruir un ciclo (o todos) de una carpeta:
     venv/bin/python scripts/34_fase_dpc.py reconstruir CARPETA_CAM [--fecha ...]
   Guarda fase_<fecha>.tif (float32, radianes) y fase_<fecha>.png.

Con la configuración guardada, la segmentación en vivo (32 y la interfaz)
acepta la entrada "fase".

Recomendaciones de montaje que salen de la simulación (lib/fase_dpc.py):
  - matriz a 20–26 mm de la muestra (ideal ~23 mm); a 30 mm o más la DPC
    casi no transmite la fase de estructuras del tamaño de una célula;
  - un difusor sobre la matriz (papel vegetal o acrílico opalino): con él la
    reconstrucción correlaciona 0.92–0.94 con la fase real en todo ese rango;
  - iluminar solo con verde (patrón con color 00FF00).
La fase sale ~25% subestimada en las estructuras grandes (las frecuencias
más bajas no se transmiten): sirve para segmentar y comparar entre
condiciones con el mismo montaje, no como masa seca absoluta sin calibrar.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import fase_dpc as F  # noqa: E402

CONFIG = F.ruta_config()
cargar_geometria, guardar_geometria = F.cargar_geometria, F.guardar_geometria


def leer_ciclo(carpeta, fecha, reducir):
    """Las 4 crudas del ciclo o, si la Pi las borró, sus _dpcLR y _dpcTB."""
    from lib.en_vivo import grupos, leer
    hay = grupos(carpeta).get(fecha, {})
    for sufijos in (("_L", "_R", "_T", "_B"), ("_dpcLR", "_dpcTB")):
        if all(s in hay for s in sufijos):
            return {s: leer(hay[s], reducir) for s in sufijos}
    raise FileNotFoundError(f"{fecha}: no están ni las 4 crudas ni _dpcLR/_dpcTB")


def fechas(carpeta):
    from lib.en_vivo import grupos
    return sorted(f for f, hay in grupos(carpeta).items()
                  if "_L" in hay or "_dpcLR" in hay)


def guardar(phi, carpeta, fecha):
    import cv2
    import tifffile
    tifffile.imwrite(Path(carpeta) / f"fase_{fecha}.tif", phi.astype(np.float32))
    lo, hi = np.percentile(phi, (0.5, 99.5))
    g8 = (np.clip((phi - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(Path(carpeta) / f"fase_{fecha}.png"), g8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="accion", required=True)
    c = sub.add_parser("calibrar")
    c.add_argument("carpeta", type=Path, help="carpeta camN con las 4 crudas L/R/T/B o los _dpcLR/_dpcTB de la Pi")
    c.add_argument("--fecha", help="ciclo a usar (por defecto el primero)")
    c.add_argument("--distancia-mm", type=float, required=True, help="LEDs -> muestra, medido")
    c.add_argument("--difusor", action="store_true")
    c.add_argument("--na", type=float, default=0.40)
    c.add_argument("--reducir", type=int, default=2)
    r = sub.add_parser("reconstruir")
    r.add_argument("carpeta", type=Path)
    r.add_argument("--fecha", help="un solo ciclo (por defecto todos)")
    r.add_argument("--reducir", type=int, default=2)
    r.add_argument("--alfa", type=float, default=1e-3, help="regularización de Tikhonov")
    args = ap.parse_args()

    if args.accion == "calibrar":
        fecha = args.fecha or fechas(args.carpeta)[0]
        fotos = leer_ciclo(args.carpeta, fecha, args.reducir)
        g0 = F.GeometriaDPC(distancia_mm=args.distancia_mm, na_obj=args.na, difusor=args.difusor)
        print(f"NA de iluminación: {g0.na_iluminacion():.2f} (objetivo {args.na}); "
              f"LEDs fuera del cono por patrón: {[g0.leds(s)[2] for s in ('_L', '_R', '_T', '_B')]}")
        g, puntajes = F.calibrar_orientacion(fotos, g0, args.reducir)
        print("asimetría de la fase para cada orientación (mayor = células positivas):",
              {f"espejo {k}": round(v, 2) for k, v in puntajes.items()})
        orden = sorted(puntajes.values(), reverse=True)
        if orden[0] - orden[1] < 0.5:
            print("AVISO: las orientaciones quedan muy parejas; revisar la distancia o usar un ciclo con más células.")
        guardar_geometria(g, {"calibrado_con": f"{args.carpeta}/{fecha}"})
        guardar(F.reconstruir(fotos, g, args.reducir), args.carpeta, fecha)
        print(f"orientación elegida: espejo_x={g.espejo_x}, espejo_y={g.espejo_y} -> {CONFIG}")
        print(f"revisar {args.carpeta}/fase_{fecha}.png: células claras sobre fondo parejo")
        return

    g, meta = cargar_geometria()
    if g is None:
        sys.exit("No hay calibración: correr primero `34_fase_dpc.py calibrar …`")
    lista = [args.fecha] if args.fecha else fechas(args.carpeta)
    H = None
    for f in lista:
        fotos = leer_ciclo(args.carpeta, f, args.reducir)
        if H is None:
            H = F.transferencias(g, F.forma_de(fotos), args.reducir)
        guardar(F.reconstruir(fotos, g, args.reducir, args.alfa, _H=H), args.carpeta, f)
        print(f"{f}: listo")


if __name__ == "__main__":
    main()
