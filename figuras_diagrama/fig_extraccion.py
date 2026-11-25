# -*- coding: utf-8 -*-
u"""Figura de la extracción: de qué fuente sale cada dato, cómo se descarga y dónde queda.

El documento tenía el recorrido COMPLETO (de las fuentes a la evaluación) pero no el detalle de la
extracción, que es la primera fase y la única que toca servicios de terceros.

LO QUE NO SE ESCRIBE A MANO. Todo lo que la figura afirma se LEE del cuaderno 1: las métricas
activas y su entidad, el día de inicio, el troceo en periodos, los días que se vuelven a pedir en
cada corrida y los nombres de los archivos que quedan en disco. Si se activa una métrica o se
cambia la ventana de revisión, la figura cambia sola; y si el cuaderno deja de decir lo que la
figura afirma, el script falla en vez de publicar un dibujo desfasado.

NO SE EJECUTA EL CUADERNO 1. Esos datos ya están descargados y volver a pedirlos a la API no
aporta nada: aquí se lee su CÓDIGO, no su salida.

LA LETRA ES LA DEL CUERPO DEL DOCUMENTO, 10 puntos, y la geometría se deduce de ella: cada medida
del dibujo se expresa en puntos tipográficos y se convierte a coordenadas con `ux()` y `uy()`. El
texto se recortó para que quepa; `comprobar_que_cabe` lo verifica midiendo lo ya dibujado.

SE QUITÓ EL TÍTULO INTERIOR, que decía lo mismo que los tres encabezados de columna.

LA ESCALA. 1.289 píxeles a 186 ppp, las 6,93 pulgadas del ancho útil de la página.
"""
import ast
import io
import json
import os
import re
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCR)  # la raíz del repositorio: la carpeta madre de esta
NB1 = os.path.join(REPO, "1_DataExtraction.ipynb")
PPP = 186
ANCHO_PULG = 1289 / float(PPP)
C_TIT, C_COM = "#333333", "#eef2f6"
C_XM, C_NOAA, C_DANE, C_DISCO = "#2c6fa8", "#3f7d58", "#9a6a3a", "#8e5aa8"

# --- la letra, y toda la geometría que se deriva de ella ---------------------------------------
CUERPO = 10.0
CABECERA = 10.0
INTERLINEA = 1.50 * CUERPO
LINEAS_FILA = 5                     # la fila más alta; las demás comparten su alto
PAD_CAJA = 8.0
FILA_PT = LINEAS_FILA * INTERLINEA + 2 * PAD_CAJA
HUECO_FILA_PT = 10.0
CABECERA_PT = 1.8 * CABECERA
LINEAS_BANDA = 5
BANDA_PT = LINEAS_BANDA * INTERLINEA + 2 * PAD_CAJA
HUECO_BANDA_PT = 14.0
MARGEN_PT = 8.0
ALTO_PULG = (CABECERA_PT + 3 * FILA_PT + 2 * HUECO_FILA_PT
             + HUECO_BANDA_PT + BANDA_PT + 2 * MARGEN_PT) / 72.0

ANCHO_PT = ANCHO_PULG * 72.0
MARGEN_LAT_PT = 6.0
HUECO_COL_PT = 14.0
COL_PT = (ANCHO_PT - 2 * MARGEN_LAT_PT - 2 * HUECO_COL_PT) / 3.0


def fuente_cuaderno():
    u"""El código de las celdas del cuaderno 1, concatenado."""
    nb = json.load(io.open(NB1, encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


SRC = fuente_cuaderno()


def valor(nombre, patron=r"(\d+)"):
    m = re.search(r"^%s\s*=\s*%s" % (re.escape(nombre), patron), SRC, re.M)
    assert m, u"el cuaderno 1 no define %s" % nombre
    return m.group(1)


def metricas_activas():
    u"""Los pares [MetricId, Entity] que el cuaderno pide, sin los comentados."""
    m = re.search(r"^metricas\s*=\s*\[(.*?)^\]", SRC, re.S | re.M)
    assert m, u"no se halla la lista `metricas`"
    out = []
    for linea in m.group(1).splitlines():
        limpio = linea.split("#")[0].strip().rstrip(",")
        if not limpio.startswith("["):
            continue
        out.append(tuple(ast.literal_eval(limpio)))
    return out


METRICAS = metricas_activas()
POR_ENTIDAD = {}
for mid, ent in METRICAS:
    POR_ENTIDAD.setdefault(ent, []).append(mid)
DIAS_REVISION = int(valor("DIAS_REVISION"))
MESES_BLOQUE = int(valor("months_per_period"))
ANIO_INI = re.search(r"start_date\s*=\s*dt\.date\((\d{4})", SRC).group(1)
ARCHIVOS = sorted(set(re.findall(r'"((?:IPC|IPP)_DANE\.xlsx)"', SRC)))
assert len(ARCHIVOS) == 2, u"se esperaban los dos anexos del DANE: %s" % ARCHIVOS
ONI_XLSX = re.search(r"(Indice_ONI\.xlsx)", SRC).group(1)
assert "ERSST.v6" in SRC, u"el cuaderno ya no dice con qué versión calcula NOAA el ONI"
assert "pydataxm" in SRC and "ReadDB" in SRC, u"el cuaderno ya no usa pydataxm/ReadDB"

DENTRO = []


def ux(pt):
    return pt / 72.0 / ANCHO_PULG


def uy(pt):
    return pt / 72.0 / ALTO_PULG


def caja(ax, x, y, an, al, lineas, color, fc="white", peso="normal", borde=1.2, fs=None):
    u"""Una caja con su texto centrado; se anota para comprobar después que cabe."""
    ax.add_patch(FancyBboxPatch((x, y), an, al, boxstyle="round,pad=0,rounding_size=%.4f" % ux(5),
                                linewidth=borde, edgecolor=color, facecolor=fc, zorder=3))
    t = ax.text(x + an / 2, y + al / 2, u"\n".join(lineas), ha="center", va="center",
                fontsize=fs or CUERPO, color=C_TIT, zorder=4, fontweight=peso, linespacing=1.5)
    DENTRO.append((t, (x, y, an, al)))
    return t


def flecha(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=10,
                                 color="#999", lw=1.1, zorder=5))


def comprobar_que_cabe(fig, ax, holgura_pt=1.5):
    u"""Ningún texto puede tocar el borde de su caja; se mide con el renderizador."""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    malos = []
    for t, (x, y, an, al) in DENTRO:
        bb = t.get_window_extent(ren)
        (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        sobra = max((x - x0) * ANCHO_PT, (x1 - (x + an)) * ANCHO_PT,
                    (y - y0) * ALTO_PULG * 72, (y1 - (y + al)) * ALTO_PULG * 72) + holgura_pt
        if sobra > 0:
            malos.append((sobra, u" ".join(t.get_text().split())[:60]))
    for sobra, txt in sorted(malos, reverse=True):
        print(u"   [NO CABE por %.1f pt]  %s" % (sobra, txt))
    assert not malos, u"%d textos tocan el borde de su caja" % len(malos)
    print(u"   [ok] los %d textos caben en su caja con %.1f pt de holgura"
          % (len(DENTRO), holgura_pt))


def main():
    fig, ax = plt.subplots(figsize=(ANCHO_PULG, ALTO_PULG))
    fig.patch.set_facecolor("white")
    ax.set_position([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xcol = [ux(MARGEN_LAT_PT + i * (COL_PT + HUECO_COL_PT)) for i in range(3)]
    an = ux(COL_PT)
    y_cab = 1 - uy(MARGEN_PT + CABECERA_PT / 2)
    for x0, tit in zip(xcol, (u"FUENTES", u"CÓMO SE DESCARGA", u"QUÉ QUEDA EN DISCO")):
        ax.text(x0 + an / 2, y_cab, tit, ha="center", va="center", fontsize=CABECERA,
                color="#555", fontweight="bold")

    #: (color, líneas de la fuente, líneas del procedimiento, líneas del archivo)
    FILAS = [
        (C_XM,
         [u"XM · Mercado de Energía",
          u"Mayorista",
          u"API pública: pydataxm",
          u"%d métricas: %d del" % (len(METRICAS), len(POR_ENTIDAD.get("Sistema", []))),
          u"sistema y %d por recurso" % len(POR_ENTIDAD.get("Recurso", []))],
         [u"ReadDB().request_data(",
          u"métrica, entidad)",
          u"troceado en bloques de",
          u"%d meses, que es el límite" % MESES_BLOQUE,
          u"de la API por solicitud"],
         [u"datasets/<MetricId>.csv",
          u"%d archivos," % len(METRICAS),
          u"uno por métrica"]),
        (C_NOAA,
         [u"NOAA · Climate",
          u"Prediction Center",
          u"Índice ONI con ERSST v6",
          u"Temporadas móviles",
          u"de tres meses"],
         [u"Se descarga la tabla",
          u"vigente y cada temporada",
          u"se fecha en su mes",
          u"central, que es la",
          u"convención de NOAA"],
         [u"datasets/%s" % ONI_XLSX,
          u"formato ancho: año × mes"]),
        (C_DANE,
         [u"DANE · índices de precios",
          u"IPC, serie de empalme",
          u"IPP, serie histórica"],
         [u"El nombre del anexo lleva",
          u"el mes de referencia: se",
          u"prueba desde el mes",
          u"actual hacia atrás",
          u"hasta encontrarlo"],
         [u"datasets/%s" % ARCHIVOS[0],
          u"datasets/%s" % ARCHIVOS[1]]),
    ]
    alto = uy(FILA_PT)
    y = 1 - uy(MARGEN_PT + CABECERA_PT)
    for color, origen, como, disco in FILAS:
        yb = y - alto
        caja(ax, xcol[0], yb, an, alto, origen, color, fc=color + "14", peso="bold", borde=1.4)
        caja(ax, xcol[1], yb, an, alto, como, "#cccccc")
        caja(ax, xcol[2], yb, an, alto, disco, C_DISCO, fc=C_DISCO + "10")
        flecha(ax, xcol[0] + an + ux(2), yb + alto / 2, xcol[1] - ux(2), yb + alto / 2)
        flecha(ax, xcol[1] + an + ux(2), yb + alto / 2, xcol[2] - ux(2), yb + alto / 2)
        y = yb - uy(HUECO_FILA_PT)

    # La banda lleva en negrita SOLO su encabezado: cinco líneas en negrita gritan, y lo que debe
    # destacar es la advertencia, no el párrafo que la explica. Por eso son dos textos y no uno.
    bx, bw, bh = xcol[0], xcol[2] + an - xcol[0], uy(BANDA_PT)
    ax.add_patch(FancyBboxPatch((bx, uy(MARGEN_PT)), bw, bh,
                                boxstyle="round,pad=0,rounding_size=%.4f" % ux(5),
                                linewidth=1.2, edgecolor="#999999", facecolor=C_COM, zorder=3))
    cuerpo_banda = [
        u"En cada corrida se descarga solo lo que falta desde %s hasta ayer," % ANIO_INI,
        u"y se vuelven a pedir los últimos %d días porque XM ajusta las cifras" % DIAS_REVISION,
        u"recientes; lo anterior se conserva tal cual. El corte del periodo que",
        u"analiza el documento se fija en la fase siguiente, no aquí."]
    y_enc = uy(MARGEN_PT) + bh - uy(PAD_CAJA + INTERLINEA / 2)
    DENTRO.append((ax.text(bx + bw / 2, y_enc,
                           u"LA EXTRACCIÓN ES INCREMENTAL Y NO FIJA EL PERIODO DE ESTUDIO",
                           ha="center", va="center", fontsize=CUERPO, color=C_TIT,
                           fontweight="bold", zorder=4),
                   (bx, uy(MARGEN_PT), bw, bh)))
    DENTRO.append((ax.text(bx + bw / 2, y_enc - uy(INTERLINEA * 2.5), u"\n".join(cuerpo_banda),
                           ha="center", va="center", fontsize=CUERPO, color=C_TIT,
                           zorder=4, linespacing=1.5),
                   (bx, uy(MARGEN_PT), bw, bh)))

    comprobar_que_cabe(fig, ax)
    ruta = os.path.join(SCR, "fig_extraccion.png")
    fig.savefig(ruta, dpi=PPP, facecolor="white")
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(SCR, "fig_extraccion.%s" % ext), facecolor="white")
    plt.close(fig)
    b = open(ruta, "rb").read()
    px, py = struct.unpack(">II", b[16:24])
    assert px == 1289, u"salió de %d píxeles de ancho, no de 1.289" % px
    print(u"[ok] fig_extraccion.png  %dx%d  %.1f KB  (+ .svg y .pdf)" % (px, py, len(b) / 1024.0))
    print(u"     %.2f x %.2f pulgadas a %d ppp; letra de %.0f pt, la del cuerpo"
          % (ANCHO_PULG, ALTO_PULG, PPP, CUERPO))
    print(u"     %d métricas de XM (%s) | desde %s | bloques de %d meses | revisión %d días"
          % (len(METRICAS), ", ".join("%d %s" % (len(v), k) for k, v in POR_ENTIDAD.items()),
             ANIO_INI, MESES_BLOQUE, DIAS_REVISION))


if __name__ == "__main__":
    main()
