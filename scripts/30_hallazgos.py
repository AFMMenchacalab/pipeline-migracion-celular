"""
Pruebas y figura de los hallazgos de la búsqueda exploratoria
(28_descubrimiento.py), con las películas como réplicas (Wilcoxon pareado).

  H1  Núcleo "adelante" en la dirección de avance -> más probabilidad de dar
      media vuelta en el paso siguiente (giro > 120 grados).
  H2  Células con ~2x ADN (G2) más lentas que en G1, excluyendo células
      redondeadas y cromatina muy condensada (para no confundir con mitosis).
  H3  Células en contacto con otra: rapidez medida con el NÚCLEO (no con el
      contorno, que se deforma al tocarse) vs células libres.
  T1  (técnico) Brillo de SiR-DNA a lo largo de la película (fotoblanqueo).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.config import DATASETS, res_dir  # noqa: E402
from lib import estilo as S  # noqa: E402
from lib.estilo import plt  # noqa: E402

OUT = res_dir("descubrimiento")
UM = DATASETS["bf"]["um_per_px"]


def main():
    t = pd.read_csv(OUT / "pasos_combinados.csv.gz").sort_values(["track_id", "frame"])
    g = t.groupby("track_id")
    ok = (g.frame.shift(-1) - t.frame) == 1
    t["rap_nuc"] = np.where(ok, np.hypot(g.nx.shift(-1) - t.nx, g.ny.shift(-1) - t.ny) * UM / 5.0, np.nan)
    filas = []
    for mv, h in t.groupby("movie"):
        a = h.dropna(subset=["cos_nucleo_mov", "cos_giro_sig"])
        b = h.dropna(subset=["adn_rel", "rapidez"])
        b = b[(b.circularidad < 0.8) & (b.nuc_intensidad < b.nuc_intensidad.quantile(0.95))]
        c = h.dropna(subset=["rap_nuc", "contacto"])
        filas.append({
            "movie": mv,
            "revertir_nucleo_delante": (a[a.cos_nucleo_mov > 0.5].cos_giro_sig < -0.5).mean(),
            "revertir_nucleo_detras": (a[a.cos_nucleo_mov < -0.5].cos_giro_sig < -0.5).mean(),
            "frac_nucleo_detras": (h.cos_nucleo_mov.dropna() < 0).mean(),
            "rapidez_G1": b[(b.adn_rel > 0.7) & (b.adn_rel < 1.2)].rapidez.mean(),
            "rapidez_G2": b[(b.adn_rel > 1.6) & (b.adn_rel < 2.6)].rapidez.mean(),
            "area_G1": b[(b.adn_rel > 0.7) & (b.adn_rel < 1.2)].area_um2.mean(),
            "area_G2": b[(b.adn_rel > 1.6) & (b.adn_rel < 2.6)].area_um2.mean(),
            "rap_nuc_contacto": c[c.contacto == True].rap_nuc.mean(),       # noqa: E712
            "rap_nuc_libre": c[c.contacto == False].rap_nuc.mean(),         # noqa: E712
            "frac_contacto": (c.contacto == True).mean(),                   # noqa: E712
        })
    r = pd.DataFrame(filas)
    r.to_csv(OUT / "hallazgos_por_pelicula.csv", index=False)
    pruebas = []
    for nom, a, b_, alt in (("H1 núcleo delante -> más reversas", "revertir_nucleo_delante", "revertir_nucleo_detras", "greater"),
                            ("H2 G2 más lentas que G1", "rapidez_G2", "rapidez_G1", "less"),
                            ("H2b G2 más grandes que G1", "area_G2", "area_G1", "greater"),
                            ("H3 en contacto más rápidas (núcleo)", "rap_nuc_contacto", "rap_nuc_libre", "greater")):
        d = r[a] - r[b_]
        pruebas.append({"hallazgo": nom, "mediana_diferencia": d.median(), "peliculas_a_favor": int(((d > 0) if alt == "greater" else (d < 0)).sum()),
                        "n": int(d.notna().sum()), "p_wilcoxon": stats.wilcoxon(d.dropna(), alternative=alt).pvalue})
    fb = t.groupby(["movie", pd.cut(t.t_h, [0, 2, 4, 6, 8.5])], observed=True).nuc_intensidad.median().unstack()
    bleach = (fb.div(fb.iloc[:, 0], axis=0)).median()
    pruebas.append({"hallazgo": "T1 brillo SiR-DNA a las 6-8 h / 0-2 h (fotoblanqueo)", "mediana_diferencia": bleach.iloc[-1] - 1,
                    "peliculas_a_favor": int((fb.iloc[:, -1] < fb.iloc[:, 0]).sum()), "n": len(fb), "p_wilcoxon": np.nan})
    pr = pd.DataFrame(pruebas)
    pr.to_csv(OUT / "hallazgos_pruebas.csv", index=False)
    print(pr.round(4).to_string(index=False))

    fig, axs = plt.subplots(1, 4, figsize=(18, 4.3))
    def pareado(ax, a, b_, la, lb, ylab, tit):
        for _, row in r.iterrows():
            ax.plot([0, 1], [row[a], row[b_]], color=S.NEUTRO, lw=0.8)
        ax.scatter(np.zeros(len(r)), r[a], color=S.CAT[1], s=22, zorder=3)
        ax.scatter(np.ones(len(r)), r[b_], color=S.CAT[0], s=22, zorder=3)
        ax.set_xticks([0, 1], [la, lb])
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel(ylab)
        ax.set_title(tit, fontsize=9)
    p = pr.set_index("hallazgo").p_wilcoxon
    pareado(axs[0], "revertir_nucleo_delante", "revertir_nucleo_detras", "núcleo delante", "núcleo detrás",
            "prob. de dar media vuelta en el paso siguiente", f"H1: la posición del núcleo anticipa las reversas\n(16 películas, p = {p.iloc[0]:.0e})")
    pareado(axs[1], "rapidez_G1", "rapidez_G2", "G1 (1× ADN)", "G2 (2× ADN)", "rapidez (µm/min)",
            f"H2: las células en G2 son más lentas\n(sin células redondeadas; p = {p.iloc[1]:.3f})")
    pareado(axs[2], "rap_nuc_libre", "rap_nuc_contacto", "libres", "en contacto", "rapidez del núcleo (µm/min)",
            f"H3: el contacto no las frena\n(rapidez medida con el núcleo; p = {p.iloc[3]:.3f})")
    for mv, row in fb.iterrows():
        axs[3].plot([1, 3, 5, 7.25], row.values / row.values[0], color=S.NEUTRO, lw=0.8)
    axs[3].plot([1, 3, 5, 7.25], bleach.values, color=S.CAT[0], lw=2.4, marker="o")
    axs[3].set_xlabel("tiempo (h)")
    axs[3].set_ylabel("brillo SiR-DNA relativo al inicio")
    axs[3].set_title("Técnico: el colorante pierde brillo\n(fotoblanqueo ~23% en 8 h)", fontsize=9)
    S.guardar(fig, res_dir("figuras") / "desc_hallazgos")

    imp = pd.read_csv(OUT / "importancia_cos_giro_sig.csv").head(8)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    nom = {"rapidez": "rapidez actual", "cos_giro": "giro actual", "cos_nucleo_mov": "posición del núcleo vs avance",
           "area_um2": "área", "adn_rel": "contenido de ADN", "off_rel": "desplazamiento del núcleo",
           "alargamiento": "alargamiento", "nuc_area": "área del núcleo", "t_h": "tiempo", "dist_vecina_um": "distancia a la vecina",
           "q": "índice de forma", "densidad_local": "densidad local", "solidez": "solidez"}
    ax.barh([nom.get(v, v) for v in imp.variable[::-1]], imp.importancia[::-1], color=S.CAT[0])
    ax.set_xlabel("importancia (pérdida de precisión al desordenar la variable)")
    ax.set_title("¿Qué anticipa el giro del paso siguiente?\n(modelo evaluado en películas que no vio)", fontsize=9)
    S.guardar(fig, res_dir("figuras") / "desc_importancia")


if __name__ == "__main__":
    main()
