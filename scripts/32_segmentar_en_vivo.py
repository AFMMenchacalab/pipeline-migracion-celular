"""
Segmentación en vivo de las imágenes del microscopio propio (corre en la PC).

Vigila la carpeta donde 31_receptor_microscopio.py deja lo que manda la
Raspberry Pi y segmenta cada ciclo en cuanto está completo. El modelo se
carga una sola vez en la GPU (CUDA en NVIDIA, ROCm en AMD; ver
lib/en_vivo.py); después cada imagen tarda unos segundos, muy por debajo
del intervalo entre fotos (2–5 min). La misma función la usa la interfaz
gráfica (33_interfaz.py).

Qué es "un ciclo completo": en modo DPC la Pi guarda cuatro fotos por
tiempo (img_<fecha>_L/_R/_T/_B.tif). Se espera a tener las cuatro; el modo
y sus sufijos se leen de experimento.json, que la Pi manda al empezar.

Qué imagen recibe Cellpose (--entrada). Todavía NO está decidido cuál
segmenta mejor las imágenes DPC; se decide en el piloto comparando contra
contornos dibujados a mano. Opciones:
  suma       (L+R+T+B)/4: equivale a campo claro. Por defecto, porque es la
             modalidad en la que el pipeline está validado.
  dpc        magnitud del DPC de los dos ejes, sqrt(DPC_LR² + DPC_TB²):
             resalta bordes en todas las direcciones.
  combinada  tres canales [suma, DPC_LR, DPC_TB]; Cellpose-SAM acepta
             varios canales.

Salidas, reanudables (si se corta, al relanzarlo sigue donde quedó):
  ~/microscopio_cache/masks/propio/<experimento>/cam<N>/<fecha>.npz
  ~/microscopio_cache/masks/propio/<experimento>/cam<N>/ultima.png
  ~/microscopio_cache/masks/propio/<experimento>/segmentacion.csv

Uso:
  ../venv/bin/python scripts/32_segmentar_en_vivo.py
  ../venv/bin/python scripts/32_segmentar_en_vivo.py --entrada combinada --una-vez
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import ROOT, cache_dir  # noqa: E402
from lib import en_vivo as EV  # noqa: E402

ENTRADA_DEF = ROOT / "datasets" / "microscopio_propio"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada-dir", type=Path, default=ENTRADA_DEF,
                    help="Carpeta que llena 31_receptor_microscopio.py")
    ap.add_argument("--salida-dir", type=Path, default=None,
                    help="Dónde guardar las máscaras (por defecto ~/microscopio_cache/masks/propio)")
    ap.add_argument("--entrada", choices=EV.ENTRADAS, default="suma",
                    help="Qué imagen recibe Cellpose (ver encabezado)")
    ap.add_argument("--reducir", type=int, default=2,
                    help="Reducción de resolución (2: 0.43 µm/px, sigue por debajo de la "
                         "resolución óptica del 20x y es ~4× más rápido)")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--guardar-flows", action="store_true",
                    help="Guardar flows para reajustar umbrales sin volver a correr la red")
    ap.add_argument("--una-vez", action="store_true",
                    help="Procesar lo que haya y salir (sin quedarse esperando)")
    ap.add_argument("--espera", type=float, default=2.0, help="Segundos entre revisiones")
    args = ap.parse_args()

    args.entrada_dir.mkdir(parents=True, exist_ok=True)
    salida = args.salida_dir or cache_dir("masks", "propio")
    salida.mkdir(parents=True, exist_ok=True)
    disp = EV.dispositivo()
    print(f"[en vivo] vigilando {args.entrada_dir}", flush=True)
    print(f"[en vivo] entrada={args.entrada}  reducir={args.reducir}  "
          f"({EV.UM_POR_PX * args.reducir:.3f} µm/px)  ->  {salida}", flush=True)
    print(f"[en vivo] dispositivo: {disp['backend']} {disp['nombre']}", flush=True)
    print("[en vivo] cargando Cellpose-SAM…", flush=True)
    model = EV.cargar_modelo()
    print("[en vivo] listo; esperando imágenes (Ctrl+C para detener)", flush=True)

    n_hechos, t_total = 0, 0.0
    try:
        while True:
            for exp, cam, fecha, fotos, sufijos, destino in EV.pendientes(args.entrada_dir, salida):
                try:
                    n_cel, dt = EV.segmentar_ciclo(
                        model, exp, cam, fecha, fotos, sufijos, destino, entrada=args.entrada,
                        reducir=args.reducir, flow_threshold=args.flow_threshold,
                        cellprob_threshold=args.cellprob_threshold, guardar_flows=args.guardar_flows)
                except Exception as e:      # archivo ilegible: avisar y seguir
                    print(f"[en vivo] {exp}/{cam}/{fecha}: error ({e})", flush=True)
                    continue
                n_hechos += 1
                t_total += dt
                print(f"[en vivo] {time.strftime('%H:%M:%S')}  {exp}/{cam}/{fecha}: "
                      f"{n_cel} células en {dt:.1f} s  (promedio {t_total / n_hechos:.1f} s)",
                      flush=True)
            if args.una_vez:
                break
            time.sleep(args.espera)
    except KeyboardInterrupt:
        pass
    print(f"\n[en vivo] detenido: {n_hechos} ciclos segmentados", flush=True)


if __name__ == "__main__":
    main()
