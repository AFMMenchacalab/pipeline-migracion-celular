"""
Paquete para marcar células a mano en otra computadora (p. ej. la PC con
Windows del laboratorio), sin copiar el experimento completo (~50 GB).

Lleva solo las fotos de la muestra para marcar (lib/anotar.muestra: 14
fotos repartidas en el tiempo y entre cámaras), con sus nombres originales
de la Pi, y en _anotar/ sus máscaras de Cellpose y la lista de fotos:

  <salida>/<exp>/cam0/0123_2026-10-08_05-16-10_dpcLR.tif   (y _dpcTB, _suma)
  <salida>/<exp>/_anotar/muestra.json
  <salida>/<exp>/_anotar/cellpose/cam0/<fecha>.npz
  <salida>/<exp>/LEEME.txt

En la otra PC: interfaz → «Agregar fotos de una carpeta» → elegir
<salida>/<exp> (sin cambiar el nombre) → elegir el experimento → «Contornos
a mano». Al terminar, «Descargar marcas (.zip)» en esa página; el .zip se
descomprime aquí en ~/microscopio_cache/anotaciones/propio/.

Uso:
  venv/bin/python scripts/38_paquete_anotar.py 2026-10-07_1654_prueba_11_tratamiento_cam_1 \\
      --salida ~/Documentos/paquete_marcar
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import anotar as ANOT  # noqa: E402
from lib import en_vivo as EV  # noqa: E402
from lib.config import ROOT, cache_dir  # noqa: E402

LEEME = """Fotos para marcar células a mano: {exp}

1. Abrir la interfaz (iniciar_interfaz.bat).
2. «Agregar fotos de una carpeta» → elegir ESTA carpeta (sin cambiarle el
   nombre) → «Agregar fotos».
3. En la lista de experimentos, elegir {exp} y hacer clic en
   «Contornos a mano».
4. Marcar todas las células de cada foto (las instrucciones están en la
   página). Las fotos con la etiqueta «desde cero» se marcan sin ver lo de
   Cellpose; las demás se pueden corregir a partir de Cellpose.
5. Al terminar: «Descargar marcas (.zip)» y mandar ese archivo.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="Experimento (nombre en datasets/microscopio_propio)")
    ap.add_argument("--salida", type=Path, required=True, help="Carpeta donde dejar el paquete")
    ap.add_argument("--entrada-dir", type=Path, default=ROOT / "datasets" / "microscopio_propio")
    args = ap.parse_args()

    exp_dir = args.entrada_dir / args.exp
    masks = cache_dir("masks", "propio") / args.exp
    fotos = ANOT.muestra(args.entrada_dir, args.exp)
    if not fotos:
        sys.exit(f"No hay fotos completas en {exp_dir}")
    destino = args.salida.expanduser() / args.exp
    sin_mascara, total = [], 0
    for x in fotos:
        cam, fecha = x["cam"], x["fecha"]
        grupo = EV.grupos(exp_dir / cam)[fecha]
        (destino / cam).mkdir(parents=True, exist_ok=True)
        for s in ("_dpcLR", "_dpcTB", "_suma"):
            origen = grupo[s].resolve()          # los enlaces del importador → archivo de la Pi
            shutil.copy2(origen, destino / cam / origen.name)
            total += origen.stat().st_size
        npz = masks / cam / f"{fecha}.npz"
        if npz.exists():
            (destino / "_anotar" / "cellpose" / cam).mkdir(parents=True, exist_ok=True)
            shutil.copy2(npz, destino / "_anotar" / "cellpose" / cam / npz.name)
        else:
            sin_mascara.append(f"{cam}/{fecha}")
    (destino / "_anotar").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ANOT.carpeta(args.exp) / "muestra.json", destino / "_anotar" / "muestra.json")
    (destino / "LEEME.txt").write_text(LEEME.format(exp=args.exp), encoding="utf-8")
    print(f"{len(fotos)} fotos ({sum(x.get('prueba', False) for x in fotos)} desde cero), "
          f"{total / 1e6:.0f} MB -> {destino}")
    if sin_mascara:
        print(f"Aviso: {len(sin_mascara)} fotos todavía sin segmentar (sin «Partir de Cellpose» "
              f"ni comparación allá): {', '.join(sin_mascara)}")


if __name__ == "__main__":
    main()
