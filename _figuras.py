# -*- coding: utf-8 -*-
u"""Las figuras que comparan familias, con un solo código para las cinco.

POR QUÉ UN MÓDULO Y NO UNA CELDA EN CADA CUADERNO. Lo que se ve de una familia tiene que poder
compararse con lo que se ve de otra, y eso no se consigue copiando treinta líneas en cuatro
cuadernos: a la primera corrección una de las copias se queda atrás y las figuras dejan de ser
comparables sin que nada avise. Aquí hay una definición y cuatro llamadas.

QUÉ DIBUJA `pronostico`, en dos paneles:
  · arriba, el precio observado, la referencia de persistencia y el pronóstico de la configuración
    elegida, situados por FECHA OBJETIVO —el día pronosticado— y no por la del origen, con el techo
    del entrenamiento marcado, que es el valor más alto del objetivo que el ajuste llegó a ver;
  · abajo, la dispersión entre las cinco semillas: la banda mínimo-máximo y la configuración
    elegida, que es su promedio.

QUÉ DIBUJA `comparativa_horizonte`: las cinco configuraciones y la persistencia frente al
horizonte, en MAE y en razón. Va en el último cuaderno de la cadena, que es el único que tiene a
mano los archivos de los otros cuatro.

LA ESCALA LA PONE QUIEN LLAMA. Cada cuaderno define en su celda de configuración `ANCHO_PAGINA`,
`alto_pagina()` y `a_escala()`, que son la escala de la página del documento —6,93 pulgadas a 186
ppp—, y se las pasa a estas funciones. Así no hay una segunda definición de la escala en el repo.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: el alto de la figura del pronóstico, en la proporción con que se inserta en el documento
PROPORCION_PRONOSTICO = (1289, 760)
PROPORCION_HORIZONTE = (1289, 390)

AZUL, GRIS, NEGRO, AMBAR = "#2c6fa8", "grey", "black", "#e8a33d"


def techo_entrenamiento(proc, fin_entrenamiento):
    u"""El valor más alto del objetivo que el ajuste llegó a ver.

    Se lee del dataset de selección y no de una variable en memoria, para que las cinco familias
    marquen exactamente la misma línea.
    """
    s = pd.read_csv(os.path.join(proc, "4_dataset_seleccion.csv"),
                    usecols=["fecha", "objetivo"], parse_dates=["fecha"])
    return float(s.loc[s["fecha"] <= pd.Timestamp(fin_entrenamiento), "objetivo"].max())


def pronostico(pred, columna, prefijo_semillas, etiqueta, techo, a_escala, ancho, alto):
    u"""El pronóstico de una familia sobre el periodo de prueba, en el formato común.

    `pred` es el mismo DataFrame que el cuaderno guarda como `N_predicciones_*.csv`: una fila por
    origen de prueba, con la fecha objetivo, el observado, la persistencia, la configuración
    elegida y las cinco semillas. No se lee ningún archivo: se dibuja lo que se acaba de guardar.
    """
    semillas = [c for c in pred.columns if c.startswith(prefijo_semillas)]
    assert len(semillas) == 5, u"se esperaban 5 semillas con prefijo %r y hay %d: %s" % (
        prefijo_semillas, len(semillas), semillas)
    assert columna in pred.columns, u"no está la columna %r" % columna
    x = pd.to_datetime(pred["fecha_objetivo"])
    banda = pred[semillas].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(ancho, alto), sharex=True)
    axes[0].plot(x, pred["real"], color=NEGRO, lw=0.9, label=u"Observado")
    axes[0].plot(x, pred["persistencia"], color=GRIS, lw=0.7, ls="--", label=u"Persistencia")
    axes[0].plot(x, pred[columna], color=AZUL, lw=1.0, label=etiqueta)
    axes[0].axhline(techo, color=AMBAR, lw=0.9, ls=":", label=u"Techo del entrenamiento")
    axes[0].set_ylabel(u"Precio [COP/kWh]")
    axes[0].set_title(u"Precio de bolsa a 30 días: observado y pronosticado")
    axes[0].legend(ncol=4, frameon=False, loc="upper left")

    axes[1].fill_between(x, banda.min(axis=1), banda.max(axis=1), color=AZUL, alpha=0.25,
                         label=u"Mínimo-máximo entre las cinco semillas")
    axes[1].plot(x, pred[columna], color=AZUL, lw=0.9, label=u"Configuración elegida")
    axes[1].set_ylabel(u"Precio [COP/kWh]")
    axes[1].set_xlabel(u"Día pronosticado")
    axes[1].set_title(u"Dispersión del pronóstico entre semillas")
    axes[1].legend(frameon=False, loc="upper left")

    for ax in axes:
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    a_escala(fig)
    fig.tight_layout(pad=0.6)

    amplitud = float(np.mean(banda.max(axis=1) - banda.min(axis=1)))
    n_sobre = int((pred[columna] > techo).sum())
    return fig, {"amplitud_banda": amplitud, "dias_sobre_techo": n_sobre,
                 "maximo_pronosticado": float(pred[columna].max()), "techo": techo}


#: (nombre, archivo de error por horizonte, color, trazo, grosor) — el orden es el de la memoria
FAMILIAS_HORIZONTE = [
    (u"SARIMAX", "5_error_por_horizonte.csv", "#6b6b6b", "-", 1.3),
    (u"XGBoost", "6_error_por_horizonte_xgb.csv", "#e8a33d", "--", 1.3),
    (u"LSTM", "7_error_por_horizonte_lstm.csv", "#b5453c", "--", 1.3),
    (u"GRU", "8_error_por_horizonte_gru.csv", "#2c6fa8", "-", 2.1),
    (u"Híbrido", "9_error_por_horizonte_hibrido.csv", "#3f7d58", "-", 1.5),
]


def _leer_horizonte(proc, archivo):
    u"""Las columnas tienen nombres distintos en cada cuaderno; se localizan, no se suponen."""
    d = pd.read_csv(os.path.join(proc, archivo))
    h = next(c for c in d.columns if "horizonte" in c.lower())
    mm = next(c for c in d.columns if c.lower().startswith("mae_mod"))
    mp = next(c for c in d.columns if c.lower().startswith("mae") and "pers" in c.lower())
    d = d.sort_values(h)
    razon = d["razon"] if "razon" in d.columns else d[mm] / d[mp]
    return d[h].to_numpy(), d[mm].to_numpy(), d[mp].to_numpy(), razon.to_numpy()


def comparativa_horizonte(proc, a_escala, ancho, alto):
    u"""Las cinco configuraciones frente al horizonte, en error y en razón."""
    fig, axes = plt.subplots(1, 2, figsize=(ancho, alto))
    h0, _, pers, _ = _leer_horizonte(proc, FAMILIAS_HORIZONTE[0][1])
    axes[0].plot(h0, pers, color=NEGRO, ls=":", lw=1.4, label=u"Persistencia")
    resumen = {}
    for nombre, archivo, color, trazo, grosor in FAMILIAS_HORIZONTE:
        h, mae, _, razon = _leer_horizonte(proc, archivo)
        axes[0].plot(h, mae, color=color, ls=trazo, lw=grosor, label=nombre)
        axes[1].plot(h, razon, color=color, ls=trazo, lw=grosor, label=nombre)
        resumen[nombre] = {"razon_h1": float(razon[0]), "razon_h30": float(razon[-1]),
                           "horizontes_que_pierde": int((razon > 1).sum()), "n": len(razon)}
    axes[0].set_title(u"Error según el horizonte")
    axes[0].set_ylabel(u"MAE [COP/kWh]")
    axes[1].axhline(1.0, color=GRIS, lw=0.9)
    axes[1].set_title(u"Ventaja sobre la persistencia")
    axes[1].set_ylabel(u"MAE del modelo / MAE de la persistencia")
    for ax in axes:
        ax.set_xlabel(u"Horizonte [días]")
        ax.legend(loc="upper left", ncol=2, frameon=False)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    a_escala(fig)
    fig.tight_layout(pad=0.6)
    return fig, resumen
