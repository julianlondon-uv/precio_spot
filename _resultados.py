# -*- coding: utf-8 -*-
"""Registro único de resultados de modelos.

POR QUE EXISTE
--------------
Hasta ahora cada notebook guardaba sus cifras en un archivo propio y con un esquema propio
(`4_metricas_modelo_base.csv` con cuatro columnas, `5_metricas_sarimax.csv` con nueve). Con los cinco
modelos del alcance —SARIMAX, XGBoost, LSTM, GRU y el BiLSTM con atención y descomposición— serían
cinco archivos y cinco esquemas, y la tabla comparativa del documento habría que escribirla a mano. Eso ya produjo una divergencia: las cifras de la Tabla 8 del documento no
coinciden con las que hay hoy en `4_metricas_modelo_base.csv`, porque salieron de corridas
distintas y nadie tenía cómo notarlo.

Este módulo deja UNA fila por corrida en UN archivo, `resultados_modelos.csv`, con las tres
métricas del trabajo y, junto a ellas, las condiciones experimentales que hacen comparable —o no—
una fila con otra: la partición, el embargo, la política de reajuste, la semilla y el bloque sobre
el que se midió. La tabla comparativa del documento se genera desde ahí.

NO SE TOCA `metrics.csv`: ese archivo es el catálogo de métricas de la API de XM (MetricId,
MetricName, MetricUnits...) y no tiene relación con los resultados de los modelos. Son dos
sentidos distintos de la palabra «métrica» y cada uno conserva su archivo.

USO
---
    from _resultados import metricas, registrar

    m = metricas(y_test, pred)                     # dict con MAE, RMSE, SMAPE
    registrar(notebook="6_XGBoost_modelo_base",
              familia="Machine Learning", modelo="XGBoost",
              objetivo="promedio_30d",             # OBLIGATORIO: ver OBJETIVOS
              configuracion="objetivo relativo",
              particion="65/15/20", embargo_dias=30, reajuste_dias=None,
              semilla=42, bloque="prueba", n_obs=len(y_test), **m)

EL CAMPO `objetivo`
-------------------
Se añadió el 22/10/2026, después de que el consolidado llegara a tener diecisiete filas de DOS
objetivos distintos sin nada que los separara: nueve del XGBoost contra el precio del día 30
(MAE ~230) y ocho del SARIMAX contra el promedio de la ventana (MAE ~137). Leída de corrido, esa
tabla decía que el modelo estadístico superaba al de árboles por un 40 %, y era falso: no medían
lo mismo. Es el mismo error que este trabajo documenta en la literatura que revisa, de modo que
no podía quedarse en una convención — tenía que ser imposible de cometer.
"""
import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd

ARCHIVO = "resultados_modelos.csv"

COLUMNAS = [
    "fecha_corrida",    # cuándo se produjo la fila
    "notebook",         # qué notebook la escribió
    "familia",          # Estadística | Machine Learning | Deep Learning | Híbrido | Referencia
    "modelo",           # SARIMAX, XGBoost, LSTM, GRU, persistencia...
    "objetivo",         # QUÉ se pronostica: ver OBJETIVOS. Sin esto dos filas no son comparables
    "configuracion",    # texto libre: la variante concreta
    "particion",        # 65/15/20
    "embargo_dias",     # días descartados entre bloques
    "reajuste_dias",    # None = un solo ajuste; 60 = reajuste cada 60 días
    "semilla",
    "bloque",           # entrenamiento (en muestra) | validacion | prueba
    "n_obs",            # número de orígenes evaluados
    "MAE", "RMSE", "SMAPE",
    "notas",
]

# Los objetivos que el trabajo ha evaluado. Un MAE solo significa algo junto al suyo: el promedio
# de la ventana es una cantidad más estable que el precio de un día suelto —los errores diarios se
# cancelan al agregarlos— de modo que TODAS las cifras bajan sin que ningún modelo haya mejorado.
# Compararlas entre sí es el error que este campo existe para impedir.
OBJETIVOS = {
    "pares_30d":    "cada uno de los 30 días, medido por separado (la métrica global del trabajo)",
    "promedio_30d": "precio medio de los 30 días siguientes al origen (lectura de negocio)",
    "dia_30":       "precio del día 30 desde el origen (el horizonte más difícil)",
}


def ruta(base=None):
    """Ruta del archivo consolidado, en la raíz del repositorio."""
    base = pathlib.Path(base or pathlib.Path.cwd())
    return os.path.join(base, ARCHIVO)


def smape(y_true, y_pred):
    """SMAPE en porcentaje.

    Se prefiere al MAPE porque no se dispara cuando el precio se acerca a cero, cosa que ocurre
    en los periodos de alta hidraulicidad. (O'Connor et al., 2025) confirman esa inestabilidad del
    MAPE, aunque advierten que la versión simétrica tiene dificultades de interpretación propias;
    por eso no se reporta sola sino junto al MAE, al RMSE y a la referencia de persistencia.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def metricas(y_true, y_pred):
    """Las tres métricas del trabajo, siempre en COP/kWh sobre el precio reconstruido."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(y_true - y_pred))),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "SMAPE": smape(y_true, y_pred),
    }


def lecturas(trayectoria, reales):
    """Las tres lecturas de UNA MISMA trayectoria de 30 días, en un solo sitio.

    Las tres salen del mismo pronóstico; lo único que cambia es qué se resume. Se calculan juntas
    y aquí —y no en cada notebook— para que no puedan divergir entre familias: una comparación
    entre modelos solo significa algo si los tres resúmenes se construyen igual para todos.

        trayectoria   matriz (n_orígenes × 30) con el precio pronosticado de cada día
        reales        matriz (n_orígenes × 30) con el precio observado de cada día

    - `pares_30d`    mide los 30 días por separado y promedia los n×30 errores. Es la métrica
                     global: califica el pronóstico completo, que es lo que el modelo entrega.
    - `promedio_30d` promedia los 30 días y compara los promedios. Da un error MENOR porque los
                     errores de signo opuesto se compensan antes del valor absoluto; responde a
                     cuánto costará la energía en el mes, que es la pregunta del negocio.
    - `dia_30`       descarta 29 de cada 30 valores y conserva el horizonte más lejano.

    Sus cifras NO son comparables entre sí. Por eso cada una se registra con su propio `objetivo`.
    """
    t = np.asarray(trayectoria, dtype=float)
    r = np.asarray(reales, dtype=float)
    if t.shape != r.shape:
        raise ValueError("trayectoria %s y reales %s no coinciden" % (t.shape, r.shape))
    if t.ndim != 2:
        raise ValueError("se esperan matrices (orígenes × horizontes), no %dD" % t.ndim)
    ok = ~np.isnan(t).any(axis=1) & ~np.isnan(r).any(axis=1)
    t, r = t[ok], r[ok]
    return {
        "pares_30d": metricas(r.ravel(), t.ravel()),
        "promedio_30d": metricas(r.mean(axis=1), t.mean(axis=1)),
        "dia_30": metricas(r[:, -1], t[:, -1]),
    }


def olvidar(notebook, base=None, verbose=True):
    """Borra del consolidado TODAS las filas de `notebook`. Se llama una vez, antes de su primer
    `registrar()`.

    POR QUE EXISTE. `registrar()` reemplaza una fila anterior solo cuando coincide en todo lo que
    la identifica, y la configuración forma parte de esa identidad. Es lo correcto —dos
    configuraciones distintas son dos resultados distintos—, pero tiene un efecto colateral: si
    cambia la selección de variables o los hiperparámetros, las filas de la corrida anterior
    quedan huérfanas y son indistinguibles de filas buenas.

    Pasó el 26/10/2026. El archivo conservaba nueve filas del 19/10 medidas sobre el dataset de
    seis variables y 42 columnas, con la configuración `profundidad 2, 100 árboles`. Ninguna estaba
    mal etiquetada: todas medían el precio del día 30, su `objetivo` era correcto. Pero al ordenar
    por MAE, `tabla(objetivo="dia_30")` devolvía el 224,47 de esa corrida superada por delante del
    232,74 vigente — un ranking falso construido con filas individualmente correctas. El campo
    `objetivo` impide mezclar lecturas; no impedía mezclar épocas.

    Con esto, cada notebook es DUEÑO de sus filas: una reejecución deja exactamente las de su
    corrida actual. El precio es que una corrida interrumpida a mitad deja el archivo sin las filas
    de ese notebook, pero son regenerables reejecutándolo, que es la operación normal del trabajo.
    """
    destino = ruta(base)
    if not os.path.exists(destino):
        return 0
    d = pd.read_csv(destino)
    if "notebook" not in d.columns:
        return 0
    quitar = (d["notebook"] == notebook).sum()
    if quitar:
        d[d["notebook"] != notebook].to_csv(destino, index=False)
    if verbose:
        print("  [resultados_modelos.csv] %s: se retiran %d filas de corridas anteriores "
              "de este notebook" % (notebook, quitar))
    return int(quitar)


def registrar(notebook, familia, modelo, objetivo, configuracion, particion, semilla, bloque,
              n_obs, MAE, RMSE, SMAPE, embargo_dias=30, reajuste_dias=None, notas="",
              base=None, verbose=True):
    """Añade una fila al consolidado. Si ya existe una fila idéntica en todo salvo la fecha y
    las métricas, la reemplaza: así volver a correr un notebook actualiza su resultado en vez de
    acumular duplicados que después nadie sabe cuál es el bueno.

    `objetivo` NO tiene valor por omisión a propósito: una cifra sin saber qué pronostica no es
    comparable con ninguna otra, y olvidarlo debe fallar en voz alta y no pasar inadvertido."""
    if objetivo not in OBJETIVOS:
        raise ValueError(
            "objetivo=%r no está en OBJETIVOS. Debe ser uno de: %s.\n"
            "Este campo es obligatorio porque dos filas con objetivos distintos NO son "
            "comparables: el promedio de la ventana es más fácil de acertar que el precio de un "
            "día suelto, de modo que sus MAE no se pueden poner en la misma tabla."
            % (objetivo, ", ".join(repr(k) for k in OBJETIVOS)))
    fila = {
        "fecha_corrida": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "notebook": notebook, "familia": familia, "modelo": modelo,
        "objetivo": objetivo,
        "configuracion": configuracion, "particion": particion,
        "embargo_dias": embargo_dias, "reajuste_dias": reajuste_dias,
        "semilla": semilla, "bloque": bloque, "n_obs": int(n_obs),
        "MAE": round(float(MAE), 2), "RMSE": round(float(RMSE), 2),
        "SMAPE": round(float(SMAPE), 2), "notas": notas,
    }
    destino = ruta(base)
    if os.path.exists(destino):
        d = pd.read_csv(destino)
        for c in COLUMNAS:
            if c not in d.columns:
                d[c] = pd.NA
        d = d[COLUMNAS]
    else:
        d = pd.DataFrame(columns=COLUMNAS)

    # la identidad de una corrida: todo lo que la hace comparable, menos el resultado
    clave = ["notebook", "familia", "modelo", "objetivo", "configuracion", "particion",
             "embargo_dias", "reajuste_dias", "semilla", "bloque"]
    if len(d):
        igual = np.ones(len(d), dtype=bool)
        for c in clave:
            izq = d[c].astype(object).where(d[c].notna(), None)
            igual &= np.array([a == fila[c] for a in izq])
        if igual.any():
            d = d.loc[~igual]
            if verbose:
                print("  (se reemplaza la fila anterior de esta misma configuración)")

    nueva = pd.DataFrame([fila])[COLUMNAS]
    d = nueva if not len(d) else pd.concat([d, nueva], ignore_index=True)
    d.to_csv(destino, index=False)
    if verbose:
        # sin flechas ni guiones tipográficos: la consola de Windows usa cp1252 y no los admite
        print("  [resultados_modelos.csv] %s | %s | %s | bloque %s -> MAE %.1f  RMSE %.1f  SMAPE %.1f"
              % (familia, modelo, configuracion, bloque, fila["MAE"], fila["RMSE"], fila["SMAPE"]))
    return fila


def tabla(base=None, bloque="prueba", objetivo="promedio_30d"):
    """El consolidado listo para leer, ordenado por MAE. Es lo que debe alimentar la tabla
    comparativa del documento, en vez de transcribir cifras a mano.

    Filtra por `objetivo` por omisión, y no por comodidad: una tabla que mezcle objetivos ordena
    por MAE cifras que miden cosas distintas y produce un ranking falso. Para verlo todo hay que
    pedirlo explícitamente con objetivo=None, y entonces avisa.

    LAS REFERENCIAS APARECEN REPETIDAS, y es intencional. Cada notebook de modelo recalcula la
    persistencia sobre su propio conjunto de orígenes en vez de leerla de otro sitio, de modo que
    si dos notebooks hubieran construido la partición de forma distinta sus referencias no
    coincidirían y se vería aquí. Que aparezcan dos filas con el MISMO MAE es la comprobación de
    que el protocolo es idéntico; dos filas con MAE distinto serían un fallo grave. Al armar la
    tabla comparativa del documento se conserva una sola, y la columna `notebook` dice de dónde
    viene cada una.
    """
    destino = ruta(base)
    if not os.path.exists(destino):
        return pd.DataFrame(columns=COLUMNAS)
    d = pd.read_csv(destino)
    if "objetivo" not in d.columns:
        raise ValueError(
            "%s no tiene la columna `objetivo`: es de antes de que fuera obligatoria. "
            "Hay que marcarla antes de usar la tabla, o las filas no son comparables." % ARCHIVO)
    if bloque:
        d = d[d["bloque"] == bloque]
    if objetivo is None:
        presentes = sorted(d["objetivo"].dropna().unique())
        if len(presentes) > 1:
            print("  AVISO: la tabla mezcla %d objetivos (%s). Sus MAE NO son comparables "
                  "entre sí: ordenarla por MAE produce un ranking sin sentido."
                  % (len(presentes), ", ".join(presentes)))
        return d.sort_values(["objetivo", "MAE"]).reset_index(drop=True)
    d = d[d["objetivo"] == objetivo]
    return d.sort_values("MAE").reset_index(drop=True)
