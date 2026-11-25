# -*- coding: utf-8 -*-
u"""La metodología CRISP-DM y lo que este trabajo hizo en cada fase.

POR QUÉ ESTA FIGURA. El documento nombra CRISP-DM y enumera sus fases en prosa, pero no muestra ni
el ciclo ni —sobre todo— que las fases NO son una secuencia: la comprensión de los datos y su
preparación se realimentan, y la evaluación puede devolver al planteamiento. Esa es justo la parte
que un párrafo no transmite.

CADA CAJA DICE LO QUE SE HIZO AQUÍ, no la definición de manual: sirve de mapa del documento.

LA LETRA ES LA DEL CUERPO DEL DOCUMENTO, 10 puntos, y no se consigue agrandando la fuente sobre la
disposición anterior: a 10 puntos el texto de una caja invadía la de al lado. Lo que se hizo fue
RECORTAR el texto —ninguna línea pasa de 24 caracteres— y DEDUCIR la geometría de los tamaños en
puntos, en vez de fijar la geometría y ver qué letra cabía. Toda medida de este dibujo se expresa
en puntos tipográficos y se convierte a coordenadas con `u()`; si mañana cambia el cuerpo, las
cajas cambian con él y nada queda descolocado.

LO QUE SE QUITÓ, porque el documento ya lo dice en la leyenda o en el texto: el título interior,
que repetía la leyenda de Word, y el pie, que repetía lo que ya dice el rótulo de la flecha roja.

LA ESCALA. El dibujo son 1.289 píxeles a 186 puntos por pulgada, es decir las 6,93 pulgadas del
ancho útil de la página, y un `assert` lo comprueba sobre el PNG escrito. Así un punto del dibujo
es un punto sobre el papel, siempre que la figura se inserte al ancho de la columna.
"""
import os
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCR = os.path.dirname(os.path.abspath(__file__))
PPP = 186
ANCHO_PULG = 1289 / float(PPP)             # 6,93": el ancho útil de la página

# --- los tamaños de letra, que son el origen de toda la geometría -----------------------------
CUERPO = 10.0                              # el cuerpo del documento
TITULO_CAJA = 11.0
NOTA = 9.0
INTERLINEA = 1.55 * CUERPO

# --- el presupuesto vertical, en puntos --------------------------------------------------------
LINEAS = 4                                 # líneas de texto por caja
LINEAS_BARRA = 2                           # el nombre de la fase cabe en dos líneas
BARRA = LINEAS_BARRA * 1.35 * TITULO_CAJA + 7.0
PAD_SUP, PAD_INF = 6.0, 8.0
CAJA_PT = BARRA + PAD_SUP + LINEAS * INTERLINEA + PAD_INF
HUECO_PT = 115.0                           # entre las dos filas: la flecha roja y su rótulo
MARGEN_PT = 10.0
ALTO_PULG = (2 * CAJA_PT + HUECO_PT + 2 * MARGEN_PT) / 72.0

# --- el presupuesto horizontal -----------------------------------------------------------------
SEPARACION_PT = 18.0
MARGEN_LAT_PT = 7.0                        # si no, el borde redondeado de las cajas de los
ANCHO_PT = ANCHO_PULG * 72.0               # extremos queda cortado por el canto de la imagen
CAJA_ANCHO_PT = (ANCHO_PT - 2 * MARGEN_LAT_PT - 2 * SEPARACION_PT) / 3.0
PAD_IZQ = 9.0

AZUL, AZUL_O, ROJO, VERDE, NARANJA, GRIS = (
    "#2c6fa8", "#1f5580", "#b5453c", "#3f7d58", "#e8a33d", "#6b6b6b")

#: (nombre de la fase, color, cuatro líneas de a lo sumo 24 caracteres)
FASES = [
    (u"1. Comprensión\ndel negocio", AZUL_O,
     [u"Precio spot a 30 días",
      u"Comparar cuatro familias",
      u"Métricas y horizonte",
      u"Comparar, no optimizar"]),
    (u"2. Comprensión\nde los datos", AZUL,
     [u"XM, NOAA y DANE",
      u"5.964 días: 2010-2026",
      u"Univariado y bivariado",
      u"Logaritmo y fase ENSO"]),
    (u"3. Preparación\nde los datos", VERDE,
     [u"ETL, calendario, vacíos",
      u"Rezagos y medias",
      u"5.754 orígenes diarios",
      u"38 columnas, 5 variables"]),
    (u"4. Modelado\n ", NARANJA,
     [u"Cinco configuraciones",
      u"SARIMAX · XGBoost",
      u"LSTM · GRU · BiLSTM",
      u"5 pliegues y 5 semillas"]),
    (u"5. Evaluación\n ", ROJO,
     [u"MAE, RMSE y SMAPE",
      u"1.155 orígenes de prueba",
      u"Promedio del mes",
      u"Por horizonte y fase"]),
    (u"6. Despliegue\n ", GRIS,
     [u"Fuera del alcance",
      u"Solo se documenta",
      u"Pronóstico reproducible",
      u"Modelos guardados"]),
]
TOPE_CARACTERES = 24


def ux(pt):
    u"""Puntos a coordenadas del eje, en horizontal."""
    return pt / 72.0 / ANCHO_PULG


def uy(pt):
    u"""Puntos a coordenadas del eje, en vertical."""
    return pt / 72.0 / ALTO_PULG


def caja(ax, cx, cy, titulo, color, lineas):
    u"""Una fase, centrada en (cx, cy), con su banda de color y sus líneas."""
    w, h = ux(CAJA_ANCHO_PT), uy(CAJA_PT)
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=%.4f" % ux(6),
                                linewidth=1.4, edgecolor=color, facecolor="white", zorder=3))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy + h / 2 - uy(BARRA)), w, uy(BARRA),
                                boxstyle="round,pad=0,rounding_size=%.4f" % ux(6),
                                linewidth=0, facecolor=color, zorder=4))
    textos = [ax.text(cx, cy + h / 2 - uy(BARRA) / 2, titulo, ha="center", va="center",
                      fontsize=TITULO_CAJA, color="white", fontweight="bold", zorder=5,
                      multialignment="center", linespacing=1.35)]
    y0 = cy + h / 2 - uy(BARRA + PAD_SUP + INTERLINEA / 2)
    for k, linea in enumerate(lineas):
        textos.append(ax.text(cx - w / 2 + ux(PAD_IZQ), y0 - k * uy(INTERLINEA), u"·  " + linea,
                              ha="left", va="center", fontsize=CUERPO, color="#1a1a1a", zorder=5))
    return (cx - w / 2, cy - h / 2, w, h), textos


def flecha(ax, p0, p1, color=GRIS, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=14, linewidth=lw,
                                 color=color, linestyle=ls, zorder=2, shrinkA=2, shrinkB=2))


def main():
    largas = [(t, l) for t, _c, ls in FASES for l in ls if len(l) > TOPE_CARACTERES]
    assert not largas, u"líneas más largas que %d caracteres: %s" % (TOPE_CARACTERES, largas)

    fig, ax = plt.subplots(figsize=(ANCHO_PULG, ALTO_PULG))
    fig.patch.set_facecolor("white")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    w, h = ux(CAJA_ANCHO_PT), uy(CAJA_PT)
    cx = [ux(MARGEN_LAT_PT + CAJA_ANCHO_PT / 2 + i * (CAJA_ANCHO_PT + SEPARACION_PT))
          for i in range(3)]
    y_sup = 1 - uy(MARGEN_PT) - h / 2
    y_inf = uy(MARGEN_PT) + h / 2
    # el ciclo: 1->2->3 arriba de izquierda a derecha, baja por la derecha, 4->5->6 de vuelta
    POS = [(cx[0], y_sup), (cx[1], y_sup), (cx[2], y_sup),
           (cx[2], y_inf), (cx[1], y_inf), (cx[0], y_inf)]
    dibujadas = [caja(ax, x, y, tit, col, lin)
                 for (x, y), (tit, col, lin) in zip(POS, FASES)]

    flecha(ax, (cx[0] + w / 2, y_sup), (cx[1] - w / 2, y_sup))
    flecha(ax, (cx[1] + w / 2, y_sup), (cx[2] - w / 2, y_sup))
    flecha(ax, (cx[2], y_sup - h / 2), (cx[2], y_inf + h / 2))
    flecha(ax, (cx[2] - w / 2, y_inf), (cx[1] + w / 2, y_inf))
    flecha(ax, (cx[1] - w / 2, y_inf), (cx[0] + w / 2, y_inf))

    # LA REALIMENTACIÓN, que es lo que distingue a CRISP-DM de una cascada: sube por el centro,
    # entre las dos filas, y su rótulo queda a la izquierda para no cruzarla.
    medio = (y_sup - h / 2 + y_inf + h / 2) / 2
    flecha(ax, (cx[1], y_inf + h / 2), (cx[1], y_sup - h / 2), color=ROJO, lw=1.4,
           ls=(0, (4, 2)))
    ax.text(cx[1] - ux(10), medio,
            u"si la evaluación no satisface,\nse vuelve sobre los datos\ny sobre el modelo",
            ha="right", va="center", fontsize=NOTA, color=ROJO, style="italic",
            linespacing=1.5)

    # EL GUARD QUE MIDE EL TEXTO DIBUJADO. Subir la letra de una disposición hecha a mano rompe
    # cajas, y mirar la imagen no escala: aquí se le pregunta a matplotlib cuánto ocupa de verdad
    # cada texto y se comprueba que cabe en su caja. Si un día se alarga una línea, falla aquí y
    # no en la página impresa.
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    desbordan = []
    for (bx, by, bw, bh), textos in dibujadas:
        for t in textos:
            caja_t = t.get_window_extent(renderer=fig.canvas.get_renderer())
            (x0, y0b), (x1, y1b) = inv.transform([(caja_t.x0, caja_t.y0),
                                                  (caja_t.x1, caja_t.y1)])
            sobra = max(bx - x0, x1 - (bx + bw), by - y0b, y1b - (by + bh))
            if sobra > 0.0005:
                desbordan.append((t.get_text().replace(u"\n", u" "), sobra * ANCHO_PT))
    assert not desbordan, u"texto fuera de su caja: %s" % [
        u"%s (+%.1f pt)" % (t, s) for t, s in desbordan]
    print(u"[ok] los %d textos de las seis cajas caben dentro de ellas"
          % sum(len(t) for _r, t in dibujadas))

    ruta = os.path.join(SCR, "fig_crispdm.png")
    fig.savefig(ruta, dpi=PPP, facecolor="white")
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(SCR, "fig_crispdm.%s" % ext), facecolor="white")
    plt.close(fig)

    b = open(ruta, "rb").read()
    px, py = struct.unpack(">II", b[16:24])
    assert px == 1289, u"salió de %d píxeles de ancho, no de 1.289" % px
    print(u"[ok] fig_crispdm.png  %dx%d  %.1f KB  (+ .svg y .pdf)" % (px, py, len(b) / 1024.0))
    print(u"     %.2f x %.2f pulgadas a %d ppp" % (ANCHO_PULG, ALTO_PULG, PPP))
    print(u"     letra: cuerpo %.0f pt, nombre de fase %.0f pt, nota %.0f pt; "
          u"ninguna por debajo del cuerpo del documento" % (CUERPO, TITULO_CAJA, NOTA))
    print(u"     caja de %.1f x %.1f pt; la línea más larga, %d caracteres de %d"
          % (CAJA_ANCHO_PT, CAJA_PT,
             max(len(l) for _t, _c, ls in FASES for l in ls), TOPE_CARACTERES))


if __name__ == "__main__":
    main()
