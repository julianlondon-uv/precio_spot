# -*- coding: utf-8 -*-
"""Figura del embudo de seleccion de variables. SOLO ESCRIBE EN SCRATCHPAD.

QUE TIENE QUE RESPONDER, de un vistazo: cuantas variables habia en cada etapa, cual filtro las
quito y cuales siguieron vivas hasta el final. El documento lo cuenta en prosa repartida por dos
capitulos y es justo donde el lector se pierde.

LA FORMA ELEGIDA, y por que no un embudo al uso. Un embudo clasico muestra los CONTEOS pero pierde
la identidad: no se ve que variable murio donde. Aqui cada variable es una FILA y su barra avanza
hasta el filtro que la elimina, con el motivo al lado. Se lee en los dos sentidos: por columnas,
cuantas quedan en cada etapa; por filas, hasta donde llego cada una.

LAS CUATRO ETAPAS, con los conteos que imprime el notebook 4:
    29 variables del conjunto consolidado
    -7 descartes demostrables (2 identidades contables, 4 indices en nivel, 1 informacion futura)
    -2 por colinealidad: VIF maximo de 218 a 5,70
    = 20 que se expanden a 97 columnas en 21 grupos, mas 5 de ruido
    -> 5 grupos pasan la regla de tres condiciones: 38 columnas

NINGUN NOMBRE NI CONTEO SE ESCRIBE A MANO: salen de 2_dataset_procesado.csv, de la salida ejecutada
del notebook 4 y de 4_importancia_por_bloque.csv.
"""
import json
import os
import re
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

SCR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCR)  # la raíz del repositorio: la carpeta madre de esta
PROC = os.path.join(REPO, "datasets", "procesado")
# el alto se toma multiplo de PPP para que no haya redondeo al rasterizar
PPP, ANCHO, ALTO = 186, 1289, 186 * 9
# El cuerpo del documento. De el dependen el margen izquierdo, que tiene que dar cabida al
# nombre mas largo, y el alto, que son veintitantas filas con su interlinea.
CUERPO = 10.0

C_VIVA, C_F1, C_F2, C_F3, C_SEL = "#cfd8e0", "#b5453c", "#d08c2a", "#7f8c8d", "#2c6fa8"

#: los siete descartes demostrables y su motivo, tal como los imprime el notebook 4
F1 = {"generacion": "identidad contable",
      "disponibilidad_declarada": "identidad contable",
      "ipc_nivel": "tendencia compartida",
      "ipp_general_nivel": "tendencia compartida",
      "ipp_importados_nivel": "tendencia compartida",
      "ipp_mineria_nivel": "tendencia compartida",
      "indice_ONI": "informaci\u00f3n futura"}
F2 = {"demanda": "colinealidad (VIF 217)",
      "ipp_importados_variacion_interanual": "colinealidad (VIF 218)"}

PROSA = {
    "precio_bolsa": "precio de bolsa", "precio_oferta": "precio de oferta",
    "precio_escasez": "precio de escasez", "aportes_energia": "aportes de energ\u00eda",
    "gen_hidraulica": "generaci\u00f3n hidr\u00e1ulica", "gen_termica": "generaci\u00f3n t\u00e9rmica",
    "gen_no_termica": "generaci\u00f3n no t\u00e9rmica", "generacion": "generaci\u00f3n total",
    "disp_dec_hidraulica": "disp. declarada hidr\u00e1ulica",
    "disp_dec_termica": "disp. declarada t\u00e9rmica",
    "disp_dec_no_termica": "disp. declarada no t\u00e9rmica",
    "disponibilidad_declarada": "disp. declarada total",
    "vol_util": "volumen \u00fatil", "vertimientos": "vertimientos", "demanda": "demanda real",
    "indice_ONI": "\u00edndice ONI centrado", "indice_ONI_publicado": "\u00edndice ONI publicado",
    "ipc_nivel": "IPC, nivel", "ipc_variacion_interanual": "IPC, variaci\u00f3n",
    "ipp_general_nivel": "IPP general, nivel",
    "ipp_general_variacion_interanual": "IPP general, variaci\u00f3n",
    "ipp_mineria_nivel": "IPP miner\u00eda, nivel",
    "ipp_mineria_variacion_interanual": "IPP miner\u00eda, variaci\u00f3n",
    "ipp_importados_nivel": "IPP importados, nivel",
    "ipp_importados_variacion_interanual": "IPP importados, variaci\u00f3n",
    "tipo_dia": "tipo de d\u00eda", "dia_semana": "d\u00eda de la semana",
    "dia_mes": "d\u00eda del mes", "mes": "mes del a\u00f1o",
    "festivos_ventana": "festivos de la ventana",
}


def datos():
    cols = [c for c in pd.read_csv(os.path.join(PROC, "2_dataset_procesado.csv"), nrows=1).columns
            if c != "fecha"]
    dec = pd.read_csv(os.path.join(PROC, "4_importancia_por_bloque.csv"))
    dec = dec[~dec["grupo"].str.startswith("ruido_")]
    sel = set(dec.loc[dec["decision"] == "seleccionada", "grupo"])
    motivo = dict(zip(dec["grupo"], dec["motivo"]))
    imp = dict(zip(dec["grupo"], dec["media"]))
    j = json.load(open(os.path.join(REPO, "4_SeleccionVariables.ipynb"), encoding="utf-8"))
    sal = "\n".join("".join(o.get("text", "")) for c in j["cells"]
                    for o in c.get("outputs", []) if o.get("output_type") == "stream")
    n_col = int(re.search(r"Variables predictoras: (\d+)", sal).group(1))
    n_fin = int(re.search(r"\((\d+) predictoras de", sal).group(1))
    vif_a = float(re.search(r"gen_hidraulica\s+(\d+\.\d)", sal).group(1))
    vif_d = float(re.search(r"VIF m\u00e1ximo tras la poda: (\d+\.\d+)", sal).group(1))
    return cols, sel, motivo, imp, n_col, n_fin, vif_a, vif_d


def main():
    cols, sel, motivo, imp, n_col, n_fin, vif_a, vif_d = datos()
    assert len(cols) == 29, "el conjunto tiene %d variables" % len(cols)
    assert len(sel) == 5, "las seleccionadas son %d" % len(sel)

    # etapa en que muere cada variable: 1, 2, 3 o 4 (sobrevive)
    etapa, razon = {}, {}
    for c in cols:
        if c in F1:
            etapa[c], razon[c] = 1, F1[c]
        elif c in F2:
            etapa[c], razon[c] = 2, F2[c]
        elif c in sel:
            etapa[c], razon[c] = 4, "%.3f COP/kWh" % imp.get(c, float("nan"))
        else:
            etapa[c] = 3
            m = motivo.get(c, "")
            razon[c] = ("no supera al ruido" if "no supera al ruido" in m else
                        "inconsistente entre pliegues" if "consistente" in m else
                        "inestable entre semillas" if "estable" in m else "descartada")
    # festivos_ventana se construye en la expansion: aparece como grupo pero no como variable
    extra = [g for g in motivo if g not in cols]
    for g in extra:
        etapa[g] = 4 if g in sel else 3
        razon[g] = razon.get(g, "descartada en la permutaci\u00f3n")
        m = motivo.get(g, "")
        razon[g] = ("no supera al ruido" if "no supera al ruido" in m else
                    "inconsistente entre pliegues" if "consistente" in m else
                    "inestable entre semillas" if "estable" in m else razon[g])
    nombres = cols + extra
    n1 = sum(1 for c in nombres if etapa[c] >= 2)
    n2 = sum(1 for c in nombres if etapa[c] >= 3)
    n3 = sum(1 for c in nombres if etapa[c] >= 4)
    print("  conjunto %d | tras filtro 1: %d | tras VIF: %d | seleccionadas: %d"
          % (len(cols), n1 - len(extra), n2 - len(extra), n3))

    orden = sorted(nombres, key=lambda c: (-etapa[c], -imp.get(c, -9), c))
    fig, ax = plt.subplots(figsize=(ANCHO / PPP, ALTO / PPP))
    fig.subplots_adjust(left=0.300, right=0.988, top=0.935, bottom=0.200)
    X = [0.0, 1.0, 2.0, 3.0, 4.0]
    color = {1: C_F1, 2: C_F2, 3: C_F3, 4: C_SEL}
    for k, c in enumerate(orden):
        y = len(orden) - 1 - k
        e = etapa[c]
        ax.barh(y, X[min(e, 4)] - X[0], left=X[0], height=0.70, color=C_VIVA, lw=0, zorder=2)
        if e < 4:
            ax.barh(y, 0.11, left=X[e] - 0.11, height=0.70, color=color[e], lw=0, zorder=3)
            ax.text(X[e] + 0.07, y, razon[c], va="center", ha="left", fontsize=CUERPO,
                    color=color[e], zorder=4)
        else:
            ax.barh(y, X[4] - X[0], left=X[0], height=0.70, color=C_SEL, alpha=0.75, lw=0, zorder=3)
            ax.text(X[4] + 0.07, y, razon[c], va="center", ha="left", fontsize=CUERPO,
                    color=C_SEL, fontweight="bold", zorder=4)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([PROSA.get(c, c) for c in reversed(orden)], fontsize=CUERPO)
    ax.set_ylim(-0.7, len(orden) + 3.1)      # sitio para los encabezados de etapa
    ax.set_xlim(-0.02, 5.60)

    # LOS ENCABEZADOS VAN ESCALONADOS en dos alturas. A cuerpo del documento cada uno mide m\u00e1s
    # que la separaci\u00f3n entre dos divisorias, y en una sola fila se pisaban unos con otros.
    # R\u00f3tulos de UNA l\u00ednea m\u00e1s el recuento: con dos l\u00edneas de nombre, tres filas de texto por
    # encabezado no caben entre dos divisorias ni siquiera escalonadas. El nombre completo de
    # cada filtro est\u00e1 en el texto del documento.
    etiquetas = [
        (X[0], "consolidado", len(cols)),
        (X[1], "1. descartes", n1 - len(extra)),
        (X[2], "2. colinealidad", n2 - len(extra)),
        (X[3], "3. permutaci\u00f3n", n3),
    ]
    for k, (x, et, n) in enumerate(etiquetas):
        ax.axvline(x, color="#999", lw=0.7, ls=":", zorder=1)
        ax.annotate("%s\n%d variables" % (et, n),
                    (x, len(orden) + (0.15 if k % 2 == 0 else 1.75)), ha="left", va="bottom",
                    fontsize=CUERPO, color="#333", fontweight="bold", linespacing=1.35)
    ax.axvline(X[4], color="#999", lw=0.7, ls=":", zorder=1)
    ax.annotate("conjunto final\n%d columnas" % n_fin, (X[4], len(orden) + 0.15), ha="left",
                va="bottom", fontsize=CUERPO, color=C_SEL, fontweight="bold", linespacing=1.35)
    ax.set_xticks([])
    for lado in ("top", "right", "bottom", "left"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # OJO: el notebook imprime 102 predictoras INCLUYENDO las 5 de ruido; las derivadas de
    # las variables reales son 97. Decir 102 y ademas «mas 5 de ruido» contaria el ruido dos veces.
    n_der = n_col - 5
    # El pie se parte en l\u00edneas de unos sesenta caracteres: a cuerpo del documento, las de
    # noventa que ten\u00eda antes se sal\u00edan por los dos costados del panel.
    pie = ("Entre el filtro 2 y el 3, las %d variables se expanden a %d\n"
           "columnas derivadas: rezagos de 1, 7, 14 y 30 d\u00edas, medias\n"
           "m\u00f3viles de 7 y 30, y diferencias de 1 y 30. Se a\u00f1aden 5 columnas\n"
           "de ruido aleatorio como control, de modo que la permutaci\u00f3n eval\u00faa\n"
           "%d. El conjunto final tiene %d columnas, de las %d variables que\n"
           "pasan los tres filtros."
           % (n2 - len(extra), n_der, n_col, n_fin, n3))
    ax.annotate(pie, (0.0, -0.045), xycoords="axes fraction", fontsize=CUERPO, color="#444",
                va="top", linespacing=1.6)
    ax.legend(handles=[Patch(facecolor=C_F1, label="descarte demostrable"),
                       Patch(facecolor=C_F2, label="colinealidad"),
                       Patch(facecolor=C_F3, label="no pasa la regla de permutaci\u00f3n"),
                       Patch(facecolor=C_SEL, alpha=0.75, label="seleccionada")],
              loc="upper right", fontsize=CUERPO, frameon=False, ncol=2,
              bbox_to_anchor=(1.0, 1.085), handlelength=1.5, columnspacing=1.4)

    ruta = os.path.join(SCR, "fig_seleccion.png")
    fig.savefig(ruta, dpi=PPP, facecolor="white")
    plt.close(fig)
    b = open(ruta, "rb").read()
    w, h = struct.unpack(">II", b[16:24])
    assert (w, h) == (ANCHO, ALTO), "salio %dx%d" % (w, h)
    print("\n[ok] fig_seleccion.png  %dx%d  %.1f KB" % (w, h, len(b) / 1024))


if __name__ == "__main__":
    main()
