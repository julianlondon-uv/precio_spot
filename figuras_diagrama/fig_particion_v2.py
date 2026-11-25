# -*- coding: utf-8 -*-
"""Figura 14: la particion, los cinco pliegues y lo que ve un origen. SOLO ESCRIBE EN SCRATCHPAD.

POR QUE HACE FALTA. Es la parte que el documento explica solo con prosa y que al leerla no se
entiende: tres bloques, cinco pliegues, una purga, un embargo y un horizonte de treinta dias, todo
en el mismo parrafo. Un esquema con fechas lo resuelve de un vistazo.

NINGUNA FECHA NI NINGUN CONTEO ESTA ESCRITO A MANO. Todo sale de 4_dataset_seleccion.csv y de las
fronteras del protocolo. Si el dataset cambia, la figura cambia con el, y si una frontera deja de
caer en fin de mes, el assert lo detiene.

LOS TRES PANELES responden a tres preguntas distintas, y por eso van juntos:

  A  ¿como se parte el conjunto?        entrenamiento, embargo y prueba, con fechas y porcentajes
  B  ¿que es un pliegue?                los cinco, cada uno con su ajuste, su purga y su validacion
  C  ¿que ve y que pregunta un origen?  los 180 dias de historia que leen los rezagos y los 30 que
                                        hay que pronosticar

LO QUE EL PANEL B TIENE QUE DEJAR CLARO, porque es justo lo que confunde: la purga NO se la queda la
validacion. Son tres tramos, y el de en medio no es de nadie. Por eso se dibuja con trama y con su
propia etiqueta, no como un hueco.

Y LA DIFERENCIA ENTRE FAMILIAS SE DECLARA EN EL PANEL B: la purga la necesitan los modelos que
ajustan sobre filas-origen, porque cada fila arrastra un objetivo que mira treinta dias adelante. El
SARIMAX ajusta sobre la serie diaria del precio y llega al ultimo dia del mes.
"""
import os
import struct
import sys

sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

SCR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCR)  # la raíz del repositorio: la carpeta madre de esta
PROC = os.path.join(REPO, "datasets", "procesado")
#: EL TAMAÑO SE DEFINE EN PULGADAS DE PAGINA, NO EN PIXELES SUELTOS.
#: La primera version se dibujo a 12,89 pulgadas de ancho y Word la encaja en los 6,92 de la
#: columna: todo el texto se reducia al 54 %, de modo que una anotacion de 6,6 puntos acababa en
#: 3,5 y era ilegible impresa. Dibujando a 6,93 pulgadas con 186 ppp salen los mismos 1289 px de
#: ancho que usan las demas figuras del documento, pero cada punto tipografico es un punto de
#: verdad sobre el papel.
PPP = 186
# La letra, de la que depende todo lo demás: el cuerpo del documento son 10 pt, y
# los catorce tamaños que este dibujo fijaba uno a uno quedaban entre 5,9 y 8,0.
CUERPO = 10.0
TITULO_PANEL = 11.0
ANCHO, ALTO = 1289, 1500
ANCHO_PULG, ALTO_PULG = ANCHO / PPP, ALTO / PPP

H = 30                      # el horizonte, en dias
# REZAGO_MAX se DERIVA de los nombres de columna, mas abajo: escribirlo a mano fue el error que
# hizo que este panel anunciara 180 dias de historia cuando el conjunto alcanza 30.
FIN_TR = pd.Timestamp("2022-12-31")
INI_PR = pd.Timestamp("2023-02-01")
PLIEGUES = [("2012-12-31", "2013-01-01", "2014-12-31"),
            ("2014-12-31", "2015-01-01", "2016-12-31"),
            ("2016-12-31", "2017-01-01", "2018-12-31"),
            ("2018-12-31", "2019-01-01", "2020-12-31"),
            ("2020-12-31", "2021-01-01", "2022-12-31")]

C_AJ, C_PU, C_VA, C_PR = "#2c6fa8", "#c9c9c9", "#e8a33d", "#b5453c"

s = pd.read_csv(os.path.join(PROC, "4_dataset_seleccion.csv"),
                parse_dates=["fecha", "fecha_objetivo"]).sort_values("fecha")
N = len(s)
ENTREN = s[s["fecha"] <= FIN_TR]
PRUEBA = s[s["fecha"] >= INI_PR]
EMBARGO = s[(s["fecha"] > FIN_TR) & (s["fecha"] < INI_PR)]
assert len(ENTREN) + len(EMBARGO) + len(PRUEBA) == N, "los tres bloques no cubren el conjunto"
assert ((s["fecha_objetivo"] - s["fecha"]).dt.days == H).all(), "el desfase del horizonte no es 30"

# --- cuanto mira hacia atras el conjunto, derivado de sus propias columnas ---
# Cada columna derivada declara su alcance en el sufijo: _lag30 lee t-30, _ma30 promedia [t-30,
# t-1], _dif30 resta t-30. El mayor de todos es lo que el origen necesita conocer.
import re as _re
_ALC = {"rezago": set(), "media": set(), "diferencia": set()}
for _c in s.columns:
    for _pat, _k in ((r"_lag(\d+)$", "rezago"), (r"_ma(\d+)$", "media"),
                     (r"_dif(\d+)$", "diferencia")):
        _m = _re.search(_pat, _c)
        if _m:
            _ALC[_k].add(int(_m.group(1)))
REZAGO_MAX = max(v for conj in _ALC.values() for v in conj)
assert REZAGO_MAX == 30, "el alcance del conjunto cambio a %d: revisar el rotulo" % REZAGO_MAX


def _lista(conj):
    v = sorted(conj)
    return ", ".join(str(x) for x in v[:-1]) + " y " + str(v[-1]) if len(v) > 1 else str(v[0])


ROTULO_DERIVADAS = ("rezagos de %s; medias de %s; diferencias de %s"
                    % (_lista(_ALC["rezago"]), _lista(_ALC["media"]), _lista(_ALC["diferencia"])))


def dia(f):
    return pd.Timestamp(f).strftime("%d/%m/%Y")


def barra(ax, yc, ini, fin, color, alto=0.52, trama=None, borde="white"):
    ini, fin = pd.Timestamp(ini), pd.Timestamp(fin)
    ax.barh(yc, (fin - ini).days + 1, left=mdates.date2num(ini), height=alto,
            color=color, edgecolor=borde, linewidth=0.8, hatch=trama, zorder=3)


def main():
    fig = plt.figure(figsize=(ANCHO_PULG, ALTO_PULG))
    # El margen izquierdo da sitio a «Pliegue N» a cuerpo del documento: con 0,092 la etiqueta
    # salía cortada por el canto de la imagen.
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 3.1, 1.15], hspace=0.60,
                          left=0.135, right=0.995, top=0.950, bottom=0.050)

    # ---------------------------------------------------------------- A. los tres bloques
    # El embargo son 31 dias sobre un eje de dieciseis años: la barra es un hilo y la etiqueta no
    # cabe dentro. Va fuera, con guia, y solo los dos bloques grandes se rotulan por dentro.
    axA = fig.add_subplot(gs[0])
    barra(axA, 0, ENTREN["fecha"].min(), FIN_TR, C_AJ, alto=0.62)
    barra(axA, 0, EMBARGO["fecha"].min(), EMBARGO["fecha"].max(), C_PU, alto=0.62, trama="////",
          borde="#555")
    barra(axA, 0, INI_PR, PRUEBA["fecha"].max(), C_PR, alto=0.62)
    def rotulo(n, et):
        return "%s · %s orígenes · %s %%" % (
            et, format(n, ",").replace(",", "."), ("%.1f" % (100 * n / N)).replace(".", ","))
    # ENTRENAMIENTO cabe dentro de su barra; PRUEBA son 1.155 dias sobre dieciseis años y no.
    axA.text(mdates.date2num(ENTREN["fecha"].min() + (FIN_TR - ENTREN["fecha"].min()) / 2), 0,
             rotulo(len(ENTREN), "ENTRENAMIENTO"), ha="center", va="center", fontsize=CUERPO,
             color="white", fontweight="bold", zorder=4)
    # El rótulo de la prueba se ancla al borde derecho del PANEL: anclado al centro de su barra
    # se le cruzaba encima la guía de la nota del embargo.
    axA.annotate(rotulo(len(PRUEBA), "PRUEBA"), (1.0, 0.40),
                 xycoords=("axes fraction", "data"),
                 ha="right", va="bottom", fontsize=CUERPO, color="#8d2f28", fontweight="bold")
    # Las dos fechas del bloque de prueba distan 38 meses sobre un eje de dieciséis años: a cuerpo
    # del documento se pisaban. Van juntas en un solo rótulo, alineado al final del eje.
    # Bajo la barra solo van los DOS EXTREMOS del periodo. Las dos fechas interiores —el fin del
    # entrenamiento y el principio de la prueba— distan 38 meses sobre un eje de dieciséis años y
    # a cuerpo del documento se pisaban entre sí y con la guía de la nota: van dentro de la nota.
    axA.annotate(dia(ENTREN["fecha"].min()), (mdates.date2num(ENTREN["fecha"].min()), -0.40),
                 ha="left", va="top", fontsize=CUERPO, color="#333")
    axA.annotate(dia(PRUEBA["fecha"].max()), (mdates.date2num(PRUEBA["fecha"].max()), -0.40),
                 ha="right", va="top", fontsize=CUERPO, color="#333")
    axA.set_ylim(-2.95, 1.25)
    axA.set_title("A. La partición: %s orígenes diarios en tres tramos contiguos"
                  % format(N, ",").replace(",", "."), fontsize=TITULO_PANEL, loc="left", pad=14)
    # La nota del embargo va DEBAJO y su guía sube hacia la barra: por encima, la guía cruzaba el
    # rótulo de la prueba, que ocupa toda la mitad derecha a cuerpo del documento.
    axA.annotate("EMBARGO: todo enero de 2023, %d orígenes que ni entrenan ni se evalúan.\n"
                 "El entrenamiento acaba el %s y la prueba empieza el %s."
                 % (len(EMBARGO), dia(FIN_TR), dia(INI_PR)),
                 xy=(mdates.date2num(EMBARGO["fecha"].min() + pd.Timedelta(days=15)), -0.34),
                 xytext=(mdates.date2num(pd.Timestamp("2010-09-01")), -2.35), fontsize=CUERPO,
                 ha="left", va="center", color="#444", multialignment="left",
                 arrowprops=dict(arrowstyle="->", color="#888", lw=0.9,
                                 connectionstyle="angle3,angleA=0,angleB=-70"))

    # ---------------------------------------------------------------- B. los cinco pliegues
    axB = fig.add_subplot(gs[1])
    for k, (fa, vi, vf) in enumerate(PLIEGUES):
        yc = len(PLIEGUES) - 1 - k
        fa, vi, vf = pd.Timestamp(fa), pd.Timestamp(vi), pd.Timestamp(vf)
        assert fa.is_month_end and vi.is_month_start and vf.is_month_end, \
            "el pliegue %d no tiene fronteras de mes completo" % (k + 1)
        fin_aj = fa - pd.Timedelta(days=H)
        aj = ENTREN[ENTREN["fecha"] <= fin_aj]
        pu = ENTREN[(ENTREN["fecha"] > fin_aj) & (ENTREN["fecha"] <= fa)]
        va = ENTREN[(ENTREN["fecha"] >= vi) & (ENTREN["fecha"] <= vf)]
        assert len(pu) == H, "la purga del pliegue %d mide %d dias" % (k + 1, len(pu))
        assert (aj["fecha_objetivo"] < vi).all(), \
            "un origen del ajuste del pliegue %d tiene su objetivo en la validacion" % (k + 1)
        barra(axB, yc, aj["fecha"].min(), fin_aj, C_AJ)
        barra(axB, yc, fin_aj + pd.Timedelta(days=1), fa, C_PU, trama="////")
        barra(axB, yc, vi, vf, C_VA)
        # A cuerpo del documento no cabe texto dentro de la barra de validación —son dos años
        # sobre un eje de trece— ni caben dos rótulos de fechas bajo la misma fila sin pisarse.
        # Todo el pliegue se describe en UNA línea bajo su barra, alineada a la izquierda, y los
        # colores los explica la leyenda.
        axB.text(mdates.date2num(aj["fecha"].min() + (fin_aj - aj["fecha"].min()) / 2), yc,
                 "ajuste · %s" % format(len(aj), ",").replace(",", "."),
                 ha="center", va="center", fontsize=CUERPO, color="white", fontweight="bold",
                 zorder=4)
        axB.annotate("hasta %s · validación %s: %s a %s"
                     % (dia(fin_aj), format(len(va), ",").replace(",", "."), dia(vi), dia(vf)),
                     (mdates.date2num(aj["fecha"].min()), yc - 0.32),
                     ha="left", va="top", fontsize=CUERPO, color="#555")
    axB.set_yticks(range(len(PLIEGUES)))
    axB.set_yticklabels(["Pliegue %d" % k for k in range(len(PLIEGUES), 0, -1)], fontsize=CUERPO)
    axB.set_ylim(-1.15, len(PLIEGUES) + 0.15)
    axB.set_title("B. Los cinco pliegues: el ajuste crece y la validación avanza",
                  fontsize=TITULO_PANEL, loc="left", pad=5)
    # La leyenda baja al pie del panel y se reparte en tres columnas: a cuerpo del documento, el
    # recuadro de la esquina tapaba la barra del primer pliegue.
    axB.legend(handles=[Patch(facecolor=C_AJ, label="ajuste"),
                        Patch(facecolor=C_PU, hatch="////", edgecolor="#555",
                              label="purga de %d días: ni ajuste ni validación" % H),
                        Patch(facecolor=C_VA, label="validación")],
               loc="upper center", fontsize=CUERPO, frameon=False, ncol=3,
               bbox_to_anchor=(0.5, -0.06), handlelength=1.6, columnspacing=1.6)

    # ---------------------------------------------------------------- C. un origen
    axC = fig.add_subplot(gs[2])
    t0 = pd.Timestamp("2024-08-31")                 # un origen cualquiera del bloque de prueba
    assert (s["fecha"] == t0).any(), "el origen del ejemplo no esta en el conjunto"
    ini_h, fin_h = t0 - pd.Timedelta(days=REZAGO_MAX), t0 + pd.Timedelta(days=H)
    barra(axC, 0, ini_h, t0, C_AJ, alto=0.46)
    barra(axC, 0, t0 + pd.Timedelta(days=1), fin_h, C_PR, alto=0.46)
    # La línea del origen se detiene antes de la fecha que va debajo: más larga la cruzaba.
    axC.plot([mdates.date2num(t0)] * 2, [-0.58, 0.42], color="black", lw=1.3, zorder=5)
    # LAS DOS BARRAS MIDEN LO MISMO, treinta dias, y esa simetria es el mensaje del panel: un origen
    # mira treinta dias atras y pregunta por treinta dias adelante.
    #
    # LOS ROTULOS VAN FUERA. Con el alcance corregido a treinta dias las barras miden la tercera
    # parte de lo que median con el 180, y ningun rotulo de esta longitud cabe dentro: encajarlo a
    # la fuerza reduciendo el cuerpo lo dejaria por debajo de lo legible en la pagina impresa. Van
    # encima, cada uno del color de su barra, y el detalle de las columnas derivadas baja al pie
    # porque es informacion de segundo orden.
    # Los dos rótulos se anclan a los EXTREMOS del eje, no al centro de su barra: a cuerpo del
    # documento cada uno mide casi tanto como su barra, y centrados se tocaban en el medio.
    # Anclados a los extremos del PANEL, no a los de las barras: las dos barras juntas ocupan algo
    # más de la mitad del ancho, y los dos rótulos a cuerpo del documento no caben sobre ellas.
    axC.annotate("lo que el origen CONOCE\n%d días de historia" % REZAGO_MAX,
                 (0.0, 0.82), xycoords=("axes fraction", "data"),
                 ha="left", va="center", fontsize=CUERPO, color="#1f5580", fontweight="bold",
                 multialignment="left")
    axC.annotate("lo que hay que PRONOSTICAR\nlos %d días siguientes" % H,
                 (1.0, 0.82), xycoords=("axes fraction", "data"),
                 ha="right", va="center", fontsize=CUERPO, color="#8d2f28", fontweight="bold",
                 multialignment="right")
    axC.annotate("origen t\n%s" % dia(t0), (mdates.date2num(t0), -0.30), ha="center", va="top",
                 fontsize=CUERPO, fontweight="bold")
    axC.annotate("t + %d\n%s" % (H, dia(fin_h)), (mdates.date2num(fin_h), -0.30),
                 ha="center", va="top", fontsize=CUERPO, color="#8d2f28")
    axC.annotate("t − %d\n%s" % (REZAGO_MAX, dia(ini_h)), (mdates.date2num(ini_h), -0.30),
                 ha="center", va="top", fontsize=CUERPO, color="#1f5580")
    # UNA sola nota al pie, centrada en el panel y partida en dos líneas: en una sola, a cuerpo
    # del documento, se salía por los dos costados.
    pie_c = "columnas derivadas de cada variable:\n%s" % ROTULO_DERIVADAS
    axC.text(mdates.date2num(t0), -1.78, pie_c, ha="center", va="center", fontsize=CUERPO,
             color="#555", linespacing=1.5)
    axC.set_ylim(-2.35, 1.32)
    axC.set_title("C. Anatomía de un origen: por eso hace falta la purga, y por eso mide %d días"
                  % H, fontsize=TITULO_PANEL, loc="left", pad=9)
    axC.set_xlim(mdates.date2num(ini_h - pd.Timedelta(days=20)),
                 mdates.date2num(fin_h + pd.Timedelta(days=20)))

    for ax in (axA, axB, axC):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.grid(axis="x", alpha=0.18, zorder=0)
    axC.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axC.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axA.set_yticks([])
    axC.set_yticks([])

    ruta = os.path.join(SCR, "fig16_particion.png")
    fig.savefig(ruta, dpi=PPP, facecolor="white")
    # Vectoriales: el texto sigue siendo texto y se puede editar fuera de Python.
    for ext in ("svg", "pdf"):
        fig.savefig(os.path.join(SCR, "fig16_particion.%s" % ext), facecolor="white")
    plt.close(fig)
    b = open(ruta, "rb").read()
    w, h = struct.unpack(">II", b[16:24])
    assert (w, h) == (ANCHO, ALTO), "salio %dx%d y debe ser %dx%d" % (w, h, ANCHO, ALTO)
    print("[ok] fig16_particion.png  %dx%d  %.1f KB  (+ .svg y .pdf editables)"
          % (w, h, len(b) / 1024))
    print("     alcance hacia atras DERIVADO del dataset: t-%d  |  %s"
          % (REZAGO_MAX, ROTULO_DERIVADAS))
    print("     %s origenes | entrenamiento %s | embargo %d | prueba %s"
          % (N, len(ENTREN), len(EMBARGO), len(PRUEBA)))
    for k, (fa, vi, vf) in enumerate(PLIEGUES, 1):
        fin_aj = pd.Timestamp(fa) - pd.Timedelta(days=H)
        print("     pliegue %d: ajuste hasta %s (%d) | purga %d | validacion %s a %s (%d)"
              % (k, fin_aj.date(), (ENTREN["fecha"] <= fin_aj).sum(), H, vi, vf,
                 ((ENTREN["fecha"] >= pd.Timestamp(vi)) & (ENTREN["fecha"] <= pd.Timestamp(vf))).sum()))


if __name__ == "__main__":
    main()
