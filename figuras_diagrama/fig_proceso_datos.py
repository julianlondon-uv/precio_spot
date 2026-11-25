# -*- coding: utf-8 -*-
u"""El recorrido de los datos, de las fuentes al pronóstico evaluado.

POR QUÉ ESTA FIGURA. El documento describe el ETL, la selección, la partición y el modelado en
secciones separadas, y no hay ningún sitio donde se vea el recorrido entero.

LAS CIFRAS DE CADA ETAPA SE DERIVAN de los archivos: días, columnas y orígenes salen de leerlos,
no de escribirlos. Si el dataset cambia, la figura cambia con él.

QUÉ TIENE QUE DEJAR CLARO:
  1. que cada etapa produce un archivo con nombre, que es lo que hace el trabajo reproducible
  2. dónde se reduce el conjunto y por qué (5.964 días → 5.754 orígenes → 38 columnas)
  3. que las cinco configuraciones leen EL MISMO archivo, que es la condición de la comparación

LA LETRA ES LA DEL CUERPO DEL DOCUMENTO, 10 puntos. No se consigue agrandando la fuente sobre la
disposición anterior —a ese tamaño el texto de una caja invadía la de al lado—, sino recortando el
texto y DEDUCIENDO la geometría de los tamaños en puntos. Un guard mide cada texto ya dibujado y
comprueba que cabe en su caja: así un retoque futuro falla aquí y no en la página impresa.

SE QUITÓ EL TÍTULO INTERIOR, que repetía la leyenda de Word.

LA ESCALA. 1.289 píxeles a 186 ppp, es decir las 6,93 pulgadas del ancho útil de la página, con un
`assert` sobre el PNG escrito.
"""
import os
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCR)  # la raíz del repositorio: la carpeta madre de esta
PROC = os.path.join(REPO, "datasets", "procesado")
PPP = 186
ANCHO_PULG = 1289 / float(PPP)

CUERPO = 10.0
TITULO_CAJA = 10.5
ARCHIVO = 10.0
INTERLINEA = 1.52 * CUERPO

LINEAS = 4
BARRA = 1.95 * TITULO_CAJA
PAD_SUP, PAD_INF = 6.0, 7.0
HUECO_ARCHIVO = 5.0
CAJA_PT = (BARRA + PAD_SUP + LINEAS * INTERLINEA + HUECO_ARCHIVO
           + 1.45 * ARCHIVO + PAD_INF)
HUECO_FILA = 42.0
MARGEN_PT = 8.0
ALTO_PULG = (3 * CAJA_PT + 2 * HUECO_FILA + 2 * MARGEN_PT) / 72.0

SEPARACION_PT = 16.0
MARGEN_LAT_PT = 7.0
ANCHO_PT = ANCHO_PULG * 72.0
CAJA_ANCHO_PT = (ANCHO_PT - 2 * MARGEN_LAT_PT - 2 * SEPARACION_PT) / 3.0
PAD_IZQ = 9.0

AZUL, VERDE, NARANJA, ROJO, GRIS, MORADO = (
    "#2c6fa8", "#3f7d58", "#e8a33d", "#b5453c", "#6b6b6b", "#7a5195")
mil = lambda n: u"{:,}".format(n).replace(u",", u".")


def cifras():
    d1 = pd.read_csv(os.path.join(PROC, "1_dataset_unificado.csv"), parse_dates=["fecha"])
    d2 = pd.read_csv(os.path.join(PROC, "2_dataset_procesado.csv"), parse_dates=["fecha"])
    d4 = pd.read_csv(os.path.join(PROC, "4_dataset_seleccion.csv"), parse_dates=["fecha"])
    NO_PRED = ["fecha", "fecha_objetivo", "precio_hoy", "objetivo", "objetivo_dia30",
               "indice_ONI_publicado"]
    pred = [c for c in d4.columns if c not in NO_PRED]
    # El entrenamiento NO es «todo lo que no es prueba»: entre los dos bloques hay un embargo de
    # 31 días —enero de 2023— que no pertenece a ninguno. Restar la prueba del total devolvía
    # 4.599 en vez de los 4.568 que reporta el documento.
    pr = d4[d4["fecha"] >= pd.Timestamp("2023-02-01")]
    en = d4[d4["fecha"] <= pd.Timestamp("2022-12-31")]
    assert len(d4) - len(en) - len(pr) == 31, \
        u"el embargo no son 31 orígenes sino %d" % (len(d4) - len(en) - len(pr))
    return dict(n1=len(d1), c1=d1.shape[1], n2=len(d2), c2=d2.shape[1],
                n4=len(d4), c4=d4.shape[1], npred=len(pred),
                nentren=len(en), nprueba=len(pr))


def ux(pt):
    return pt / 72.0 / ANCHO_PULG


def uy(pt):
    return pt / 72.0 / ALTO_PULG


def caja(ax, cx, cy, titulo, color, lineas, archivo):
    u"""Cabecera de color, cuatro líneas y, al pie, el archivo que la etapa produce."""
    w, h = ux(CAJA_ANCHO_PT), uy(CAJA_PT)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=%.4f" % ux(6),
                                linewidth=1.4, edgecolor=color, facecolor="white", zorder=3))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy + h / 2 - uy(BARRA)), w, uy(BARRA),
                                boxstyle="round,pad=0,rounding_size=%.4f" % ux(6),
                                linewidth=0, facecolor=color, zorder=4))
    textos = [ax.text(cx, cy + h / 2 - uy(BARRA) / 2, titulo, ha="center", va="center",
                      fontsize=TITULO_CAJA, color="white", fontweight="bold", zorder=5)]
    y0 = cy + h / 2 - uy(BARRA + PAD_SUP + INTERLINEA / 2)
    for k, linea in enumerate(lineas):
        if not linea:
            continue
        textos.append(ax.text(cx - w / 2 + ux(PAD_IZQ), y0 - k * uy(INTERLINEA), u"·  " + linea,
                              ha="left", va="center", fontsize=CUERPO, color="#1a1a1a", zorder=5))
    textos.append(ax.text(cx, cy - h / 2 + uy(PAD_INF + 1.45 * ARCHIVO / 2), archivo,
                          ha="center", va="center", fontsize=ARCHIVO, color=color,
                          style="italic", zorder=5))
    return (cx - w / 2, cy - h / 2, w, h), textos


def flecha(ax, p0, p1, color=GRIS, lw=1.6):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=lw,
                                 color=color, zorder=2, shrinkA=1, shrinkB=1))


def main():
    c = cifras()
    fig, ax = plt.subplots(figsize=(ANCHO_PULG, ALTO_PULG))
    fig.patch.set_facecolor("white")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = ux(CAJA_ANCHO_PT), uy(CAJA_PT)
    X = [ux(MARGEN_LAT_PT + CAJA_ANCHO_PT / 2 + i * (CAJA_ANCHO_PT + SEPARACION_PT))
         for i in range(3)]
    Y = [1 - uy(MARGEN_PT) - h / 2 - f * (h + uy(HUECO_FILA)) for f in range(3)]

    #: (columna, fila, título, color, líneas, archivo) en el orden del recorrido
    ETAPAS = [
        (0, 0, u"Fuentes", AZUL,
         [u"XM · API pydataxm", u"NOAA · índice ONI", u"DANE · IPC e IPP", u""],
         u"notebook 1"),
        (1, 0, u"ETL y unificación", AZUL,
         [u"Horaria y mensual", u"a resolución diaria", u"Calendario colombiano",
          u"%s días × %d col." % (mil(c["n1"]), c["c1"])],
         u"1_dataset_unificado"),
        (2, 0, u"Análisis exploratorio", AZUL,
         # No se repite aquí el recuento de la caja anterior: el análisis exploratorio no añade
         # ni quita filas ni columnas, y verlo dos veces hace pensar que sí.
         [u"Univariado y bivariado", u"Faltantes e indicadores", u"Logaritmo y fase ENSO", u""],
         u"2_dataset_procesado"),
        (2, 1, u"Problema supervisado", VERDE,
         [u"Un origen por día", u"Objetivo: t+1 … t+30", u"Rezagos y diferencias",
          u"%s × %d columnas" % (mil(c["n4"]), c["c4"])],
         u"4_dataset_seleccion"),
        (1, 1, u"Selección de variables", VERDE,
         [u"Correlación y VIF", u"Permutación y SHAP", u"5 variables de origen",
          u"%d columnas" % c["npred"]],
         u"notebook 4"),
        (0, 1, u"Partición", NARANJA,
         [u"Entrenamiento: %s" % mil(c["nentren"]), u"Embargo: enero de 2023",
          u"Prueba: %s" % mil(c["nprueba"]), u"5 pliegues, purga 30 d"],
         u"por fecha, no al azar"),
        (0, 2, u"Modelado", NARANJA,
         [u"SARIMAX · XGBoost", u"LSTM · GRU · BiLSTM", u"El mismo protocolo",
          u"5 semillas cada una"],
         u"notebooks 5 a 9"),
        (1, 2, u"Pronóstico", MORADO,
         [u"30 días por origen", u"%s × 30 valores" % mil(c["nprueba"]),
          u"Trayectoria completa", u"y la persistencia"],
         u"*_predicciones_*"),
        (2, 2, u"Evaluación", ROJO,
         [u"MAE, RMSE y SMAPE", u"Promedio del mes", u"Por horizonte y fase",
          u"Una fila por corrida"],
         u"resultados_modelos"),
    ]
    dibujadas = [caja(ax, X[i], Y[f], t, col, li, arch) for i, f, t, col, li, arch in ETAPAS]

    for f in (0, 2):                                    # las filas que van de izquierda a derecha
        flecha(ax, (X[0] + w / 2, Y[f]), (X[1] - w / 2, Y[f]))
        flecha(ax, (X[1] + w / 2, Y[f]), (X[2] - w / 2, Y[f]))
    flecha(ax, (X[2] - w / 2, Y[1]), (X[1] + w / 2, Y[1]))   # la fila de en medio, de vuelta
    flecha(ax, (X[1] - w / 2, Y[1]), (X[0] + w / 2, Y[1]))
    flecha(ax, (X[2], Y[0] - h / 2), (X[2], Y[1] + h / 2))   # los dos saltos de fila
    flecha(ax, (X[0], Y[1] - h / 2), (X[0], Y[2] + h / 2))

    # El guard que mide el texto ya dibujado: ver el encabezado del archivo.
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    desbordan = []
    for (bx, by, bw, bh), textos in dibujadas:
        for t in textos:
            r = t.get_window_extent(renderer=fig.canvas.get_renderer())
            (x0, y0b), (x1, y1b) = inv.transform([(r.x0, r.y0), (r.x1, r.y1)])
            sobra = max(bx - x0, x1 - (bx + bw), by - y0b, y1b - (by + bh))
            if sobra > 0.0005:
                desbordan.append((t.get_text(), sobra * ANCHO_PT))
    assert not desbordan, u"texto fuera de su caja: %s" % [
        u"%s (+%.1f pt)" % (t, s) for t, s in desbordan]
    print(u"[ok] los %d textos de las nueve cajas caben dentro de ellas"
          % sum(len(t) for _r, t in dibujadas))

    ruta = os.path.join(SCR, "fig_proceso_datos.png")
    fig.savefig(ruta, dpi=PPP, facecolor="white")
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(SCR, "fig_proceso_datos.%s" % ext), facecolor="white")
    plt.close(fig)
    b = open(ruta, "rb").read()
    px, py = struct.unpack(">II", b[16:24])
    assert px == 1289, u"salió de %d píxeles de ancho, no de 1.289" % px
    print(u"[ok] fig_proceso_datos.png  %dx%d  %.1f KB  (+ .svg y .pdf)"
          % (px, py, len(b) / 1024.0))
    print(u"     %.2f x %.2f pulgadas a %d ppp; letra de %.0f pt, la del cuerpo"
          % (ANCHO_PULG, ALTO_PULG, PPP, CUERPO))
    print(u"     derivado: %s días x %d col -> %s orígenes x %d col -> %d predictoras"
          % (mil(c["n2"]), c["c2"], mil(c["n4"]), c["c4"], c["npred"]))
    print(u"     partición: %s de entrenamiento y %s de prueba"
          % (mil(c["nentren"]), mil(c["nprueba"])))


if __name__ == "__main__":
    main()
