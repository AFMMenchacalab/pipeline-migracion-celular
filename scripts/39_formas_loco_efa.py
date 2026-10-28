"""
Forma de cada célula a lo largo del tiempo con LOCO-EFA (lib/formas.py):
Shape Complexity (SC), entropía y diferencia acumulada de cada célula en
cada ciclo, unidas por el track_id del tracking. Cada célula se recorta y se
mide SOLA (masks == label); el track_id solo sirve para seguirla en el tiempo.

Entrada: lo que deja lib/propio.analizar (analisis/propio/<exp>/<cam>/
tracks.csv, o <cam>/hasta_N/ con --max-grupo) y las máscaras de la
segmentación (masks/propio/<exp>/<cam>/<fecha>.npz).
Salida, junto al tracks.csv de cada cámara:
  formas.csv           una fila por célula y ciclo (tracks.csv + descriptores:
                       sc, entropia, cd_px, xor_final_rel, lobulos_dominante,
                       L1…L12, ...). NaN si la célula no se midió (en un
                       grupo, tocando el borde de la foto, contorno chico).
  formas_por_celula.csv  una fila por track_id: SC medio, mediana, desviación,
                       pendiente por hora, entropía media, y rapidez y
                       direccionalidad si existe por_celula.csv
  sc_tiempo.png, sc_vs_rapidez.png

Por omisión solo se miden las células sueltas (--grupo-max 1) que no tocan
el borde. LOCO-EFA es muy sensible al contorno: una máscara que junta o
parte dos células da una forma que no es de una célula.

Uso:
  venv/bin/python scripts/39_formas_loco_efa.py 2026-10-07_1654_prueba_11_tratamiento
  venv/bin/python scripts/39_formas_loco_efa.py EXPERIMENTO --camara cam_1 --suavizar 1
  venv/bin/python scripts/39_formas_loco_efa.py --probar        # pruebas con formas sintéticas
En Windows: venv\\Scripts\\python.exe scripts\\39_formas_loco_efa.py ...

El cálculo es de CPU (NumPy), no de GPU: el costo es el XOR (reconstruir
cada contorno con 2…50 modos). --sin-xor lo omite; --lienzo-xor lo hace más
barato (128 por omisión, en vez de 512).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import formas as F  # noqa: E402
from lib.config import cache_dir  # noqa: E402


def resumen_por_celula(fm):
    filas = []
    for tid, g in fm.dropna(subset=["sc"]).groupby("track_id"):
        g = g.sort_values("t_s")
        t_h = g["t_s"].to_numpy() / 3600
        sc = g["sc"].to_numpy()
        pend = float(np.polyfit(t_h, sc, 1)[0]) if len(g) >= 3 and np.ptp(t_h) > 0 else np.nan
        filas.append({"track_id": tid, "ciclos_medidos": len(g),
                      "ciclos_trayectoria": int((fm["track_id"] == tid).sum()),
                      "sc_medio": sc.mean(), "sc_mediana": float(np.median(sc)), "sc_sd": sc.std(ddof=1) if len(g) > 1 else np.nan,
                      "sc_cv": sc.std(ddof=1) / sc.mean() if len(g) > 1 and sc.mean() else np.nan,
                      "sc_pendiente_por_h": pend, "entropia_media": g["entropia"].mean(),
                      "lobulos_dominante_moda": int(g["lobulos_dominante"].mode().iloc[0])})
    return pd.DataFrame(filas)


def figuras(fm, pc, destino):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = fm.dropna(subset=["sc"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    largos = ok.groupby("track_id").size().sort_values(ascending=False).index[:15]
    for tid in largos:
        g = ok[ok["track_id"] == tid].sort_values("t_s")
        axs[0].plot(g["t_s"] / 3600, g["sc"], lw=0.9, alpha=0.7)
    ok = ok.assign(h=(ok["t_s"] // 3600))
    med = ok.groupby("h")["sc"].median()
    axs[0].plot(med.index + 0.5, med.to_numpy(), "k-", lw=2.5, label="mediana de todas las células por hora")
    axs[0].set_xlabel("tiempo (h)")
    axs[0].set_ylabel("Shape Complexity (SC)")
    axs[0].set_title("SC de las 15 trayectorias más largas")
    axs[0].legend(frameon=False)
    axs[1].hist(ok["sc"], bins=40, color="#2563eb")
    axs[1].set_xlabel("SC")
    axs[1].set_ylabel("células × ciclos")
    axs[1].set_title("distribución de SC")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino / "sc_tiempo.png", dpi=140)
    plt.close(fig)
    if "rapidez_um_min" in pc.columns and pc["rapidez_um_min"].notna().sum() >= 5:
        from scipy.stats import spearmanr
        d = pc.dropna(subset=["sc_medio", "rapidez_um_min"])
        rho, p = spearmanr(d["sc_medio"], d["rapidez_um_min"])
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.scatter(d["sc_medio"], d["rapidez_um_min"], s=14, alpha=0.6, color="#2563eb")
        ax.set_xlabel("SC medio de la célula")
        ax.set_ylabel("rapidez (µm/min)")
        ax.set_title(f"n = {len(d)}  ·  Spearman ρ = {rho:.2f}  (p = {p:.2g})")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(destino / "sc_vs_rapidez.png", dpi=140)
        plt.close(fig)
        print(f"  SC medio vs rapidez: Spearman ρ = {rho:.2f} (p = {p:.2g}, n = {len(d)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experimento", nargs="?")
    ap.add_argument("--camara", default=None, help="solo esta cámara (por omisión, todas las analizadas)")
    ap.add_argument("--max-grupo", type=int, default=None,
                    help="usar el análisis hecho con este --max-grupo (carpeta hasta_N)")
    ap.add_argument("--grupo-max", type=int, default=1,
                    help="medir solo células en grupos de hasta este tamaño (1 = sueltas)")
    ap.add_argument("--con-borde", action="store_true", help="medir también las que tocan el borde de la foto")
    ap.add_argument("--modos", type=int, default=F.N_MODOS)
    ap.add_argument("--lienzo", type=int, default=F.LIENZO)
    ap.add_argument("--lienzo-xor", type=int, default=F.LIENZO_XOR)
    ap.add_argument("--sin-xor", action="store_true", help="omitir XOR y diferencia acumulada (lo más lento)")
    ap.add_argument("--suavizar", type=float, default=0.0,
                    help="σ (px de la máscara) para suavizar el contorno; 0 = máscara tal cual")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--probar", action="store_true", help="pruebas con formas sintéticas y salir")
    args = ap.parse_args()

    if args.probar:
        fallos = F.autoprueba()
        print("\nTODO BIEN" if not fallos else f"\n{len(fallos)} FALLA(S)")
        sys.exit(1 if fallos else 0)
    if not args.experimento:
        ap.error("falta el experimento (o usar --probar)")

    exp, mg = args.experimento, args.max_grupo
    an = cache_dir("analisis", "propio") / exp
    masks = cache_dir("masks", "propio") / exp
    cams = [args.camara] if args.camara else sorted(
        p.name for p in an.glob("cam*") if (p / (f"hasta_{mg}" if mg else "") / "tracks.csv").exists())
    if not cams:
        sys.exit(f"no hay tracks.csv en {an}: primero hay que correr 'Analizar trayectorias' "
                 f"en la interfaz (o lib.propio.analizar)")
    kw = dict(n_modos=args.modos, con_xor=not args.sin_xor, lienzo=args.lienzo,
              lienzo_xor=args.lienzo_xor, suavizar=args.suavizar)
    for cam in cams:
        d = an / cam / (f"hasta_{mg}" if mg else "")
        print(f"{exp}/{cam}: leyendo {d / 'tracks.csv'}", flush=True)
        tr = pd.read_csv(d / "tracks.csv")
        t0 = time.time()
        fm = F.formas_de_tracks(tr, masks / cam, grupo_max=args.grupo_max, quitar_borde=not args.con_borde,
                                procesos=args.workers, progreso=lambda s: print("  " + s, flush=True), **kw)
        n_ok = int(fm["sc"].notna().sum()) if "sc" in fm.columns else 0
        print(f"  {n_ok} de {len(tr)} detecciones medidas ({time.time() - t0:.0f} s)")
        if not n_ok:
            print("  ninguna célula cumplió las condiciones (grupo, borde, contorno)")
            continue
        fm.to_csv(d / "formas.csv", index=False)
        pc = resumen_por_celula(fm)
        pcel = d / "por_celula.csv"
        if pcel.exists():
            pc = pc.merge(pd.read_csv(pcel)[["track_id", "rapidez_um_min", "direccionalidad", "alfa"]],
                          on="track_id", how="left")
        pc.to_csv(d / "formas_por_celula.csv", index=False)
        figuras(fm, pc, d)
        print(f"  SC mediana {fm['sc'].median():.1f}  (cuartiles {fm['sc'].quantile(.25):.1f}–{fm['sc'].quantile(.75):.1f})"
              f" · {len(pc)} células con forma -> {d}")


if __name__ == "__main__":
    main()
