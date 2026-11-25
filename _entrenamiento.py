# -*- coding: utf-8 -*-
"""Las métricas del entrenamiento junto a las de la prueba, medidas con el mismo rasero.

POR QUÉ EXISTE
--------------
El trabajo informaba del error sobre el conjunto de prueba y, en la etapa de validación, de la
razón frente a la persistencia. Faltaba la otra mitad de la pregunta: **cuánto acierta el modelo
sobre los datos con los que se ajustó**. Sin esa cifra, el control del sobreajuste de cada familia
se podía argumentar —pocos parámetros, árboles de un solo corte, parar a tiempo— pero no medir, y
la distancia entre lo que el modelo sabe y lo que generaliza quedaba fuera del documento.

CÓMO SE MIDE, Y POR QUÉ ASÍ
---------------------------
La tentación es informar del error «en muestra» que cada librería da de serie: los residuos de un
paso del SARIMAX, la pérdida de entrenamiento de Keras. No sirve, y mezclarlo con el error de
prueba induce a error de la misma forma que lo hace citar un MAE sin decir qué pronostica. El
residuo de un paso mide el error a un día; la prueba de este trabajo mide el error a treinta. Son
cantidades de orden distinto, y ponerlas en la misma tabla diría que el modelo empeora diez veces
al salir de la muestra cuando buena parte de esa diferencia es solo el horizonte.

Aquí las dos etapas se miden **igual**: trayectorias de treinta días desde cada origen, el precio
reconstruido en COP/kWh, y las tres lecturas de `_resultados.lecturas`. La única diferencia entre
una fila y otra es el conjunto de orígenes sobre el que se evalúa y si el ajuste los vio o no.

LA PERSISTENCIA VIAJA CON CADA BLOQUE
-------------------------------------
Un MAE de entrenamiento y uno de prueba tampoco son comparables tal cual, porque los dos periodos
tienen niveles de precio muy distintos: el objetivo medio pasa de 192,0 COP/kWh en el
entrenamiento a 466,6 en la prueba. Por eso cada bloque lleva su propia referencia de persistencia
y su razón; la razón sí es comparable entre bloques, y es la que dice si la brecha es sobreajuste
o simplemente un periodo más difícil.

USO
---
    from _entrenamiento import tabla_brecha, registrar_brecha

    BLOQUES = {"entrenamiento": (TRAY_EN, REALES_EN, PERS_EN),
               "prueba":        (TRAY,    REALES_PR, PERS_PR)}
    brecha = tabla_brecha(BLOQUES)          # una fila por bloque y lectura
    registrar_brecha(brecha, notebook=..., familia=..., modelo=..., configuracion=..., ...)

    # y, con las fechas de cada bloque, el error de CADA ventana de treinta días:
    detalle = error_por_origen({"entrenamiento": (TRAY_EN, REALES_EN, PERS_EN, FECHAS_EN),
                                "prueba":        (TRAY,    REALES_PR, PERS_PR, FECHAS_PR)})
"""
import numpy as np
import pandas as pd

from _resultados import lecturas, registrar

#: el orden en que se leen los bloques, de lo que el modelo vio a lo que no
ORDEN = ["entrenamiento", "validacion", "prueba"]


def _clave_orden(nombre):
    for i, n in enumerate(ORDEN):
        if nombre.startswith(n):
            return i
    return len(ORDEN)


def error_por_origen(bloques):
    """El error de CADA ventana de treinta días, una fila por origen.

        bloques   {nombre: (trayectoria, reales, persistencia, fechas)}, las tres matrices
                  (n_orígenes x 30) del bloque y las fechas de sus orígenes.

    Devuelve un DataFrame con bloque, fecha_origen, MAE, MAE_persistencia y razon.

    PARA QUÉ, SI YA ESTÁ `tabla_brecha`. Esa resume cada bloque en una cifra, y una cifra no dice
    si el modelo acierta de forma pareja o alterna tramos buenos y malos. Sobre la prueba, la
    razón de las redes se mueve entre 0,5 y 1,6 según el tramo: ganan mucho en los periodos
    tranquilos y pierden contra la persistencia en los picos. Eso no se ve en el promedio, y es
    justo lo que decide si un pronóstico sirve para operar.

    EL MAE DE UNA VENTANA es la media del error absoluto de sus treinta días, que es la lectura
    `pares_30d` restringida a un origen. No es el `promedio_30d`, que compara el precio medio del
    mes con el pronosticado y cancela errores de signo contrario dentro de la ventana.
    """
    filas = []
    for nombre, (tray, reales, pers, fechas) in bloques.items():
        tray, reales, pers = (np.asarray(x, dtype=float) for x in (tray, reales, pers))
        if not (tray.shape == reales.shape == pers.shape):
            raise ValueError("en el bloque %r las matrices no coinciden: %s, %s y %s"
                             % (nombre, tray.shape, reales.shape, pers.shape))
        if len(fechas) != len(tray):
            raise ValueError("en el bloque %r hay %d fechas para %d orígenes"
                             % (nombre, len(fechas), len(tray)))
        mae = np.abs(tray - reales).mean(axis=1)
        mae_p = np.abs(pers - reales).mean(axis=1)
        filas.append(pd.DataFrame({"bloque": nombre,
                                   "fecha_origen": pd.DatetimeIndex(fechas),
                                   "MAE": mae, "MAE_persistencia": mae_p,
                                   "razon": mae / mae_p}))
    t = pd.concat(filas, ignore_index=True)
    return t.sort_values("fecha_origen").reset_index(drop=True)


def tabla_brecha(bloques):
    """Una fila por bloque y lectura, con el modelo, su persistencia y la razón entre los dos.

        bloques   {nombre: (trayectoria, reales, persistencia)}, las tres matrices
                  (n_orígenes x 30) del mismo bloque. `persistencia` es el precio del día de
                  origen repetido los treinta días, que es la referencia del trabajo.

    Devuelve un DataFrame con bloque, n_origenes, objetivo, MAE, RMSE, SMAPE, MAE_persistencia y
    razon. Las filas de distinto `objetivo` NO son comparables entre sí; las de distinto bloque
    dentro del mismo objetivo, sí, y la columna `razon` es la que permite compararlas pese a que
    los dos periodos tengan niveles de precio distintos.
    """
    filas = []
    for nombre, (tray, reales, pers) in bloques.items():
        tray, reales, pers = (np.asarray(x, dtype=float) for x in (tray, reales, pers))
        if not (tray.shape == reales.shape == pers.shape):
            raise ValueError("en el bloque %r las matrices no coinciden: %s, %s y %s"
                             % (nombre, tray.shape, reales.shape, pers.shape))
        lm, lp = lecturas(tray, reales), lecturas(pers, reales)
        for clave in ("pares_30d", "promedio_30d", "dia_30"):
            filas.append({"bloque": nombre, "n_origenes": len(tray), "objetivo": clave,
                          **lm[clave],
                          "MAE_persistencia": lp[clave]["MAE"],
                          "razon": lm[clave]["MAE"] / lp[clave]["MAE"]})
    t = pd.DataFrame(filas)
    t["_o"] = t["bloque"].map(_clave_orden)
    t["_l"] = t["objetivo"].map({"pares_30d": 0, "promedio_30d": 1, "dia_30": 2})
    return t.sort_values(["_l", "_o"]).drop(columns=["_o", "_l"]).reset_index(drop=True)


def registrar_brecha(brecha, notebook, familia, modelo, configuracion, particion, embargo_dias,
                     reajuste_dias, semilla=None, solo=("entrenamiento",), verbose=False):
    """Lleva al consolidado las filas de `brecha` cuyo bloque esté en `solo`.

    Por omisión registra solo el entrenamiento: las de prueba las escribe el propio notebook con
    su nombre de configuración completo, y duplicarlas aquí crearía dos filas de la misma corrida
    que después nadie sabría cuál es la buena.
    """
    n = 0
    for _, f in brecha.iterrows():
        if f["bloque"] not in solo:
            continue
        registrar(notebook=notebook, familia=familia, modelo=modelo, objetivo=f["objetivo"],
                  configuracion=configuracion, particion=particion, embargo_dias=embargo_dias,
                  reajuste_dias=reajuste_dias, semilla=semilla, bloque=f["bloque"],
                  n_obs=int(f["n_origenes"]) * (30 if f["objetivo"] == "pares_30d" else 1),
                  MAE=f["MAE"], RMSE=f["RMSE"], SMAPE=f["SMAPE"],
                  notas="en muestra: el ajuste vio estos orígenes"
                        if f["bloque"] == "entrenamiento" else "",
                  verbose=verbose)
        n += 1
    return n


def leer_brecha(brecha, objetivo="promedio_30d"):
    """Una frase con lo que dice la tabla para una lectura, para no escribirla a mano debajo.

    La escribe el módulo y no el notebook porque es la conclusión que más fácil se queda sin
    actualizar cuando las cifras cambian al reejecutar.
    """
    d = brecha[brecha["objetivo"] == objetivo].set_index("bloque")
    if not {"entrenamiento", "prueba"} <= set(d.index):
        return ""
    en, pr = d.loc["entrenamiento"], d.loc["prueba"]
    salto = pr["MAE"] / en["MAE"]
    sube_pers = pr["MAE_persistencia"] / en["MAE_persistencia"]
    razon = pr["razon"] / en["razon"]
    # El veredicto lo decide la cifra, no quien escribe la celda: cuánto del salto del MAE queda
    # después de descontar lo que también sube la persistencia sobre los mismos orígenes.
    if razon < 1.05:
        veredicto = ("La razón frente a la persistencia casi no se mueve, de modo que el salto "
                     "del MAE\nes dificultad del periodo y no distancia entre lo que el modelo "
                     "sabe y lo que generaliza.")
    elif razon < 1.25:
        veredicto = ("Descontado lo que sube la persistencia, queda un deterioro de %.0f %% en la "
                     "razón:\nhay brecha, pero la mayor parte del salto del MAE es el periodo."
                     % (100 * (razon - 1)))
    else:
        veredicto = ("La razón se deteriora un %.0f %% aun descontando la persistencia: el salto "
                     "NO se\nexplica solo por el periodo y hay que leerlo como sobreajuste."
                     % (100 * (razon - 1)))
    return (
        "El MAE pasa de %.2f en el entrenamiento a %.2f en la prueba, %.2f veces mayor. La\n"
        "persistencia pasa de %.2f a %.2f sobre los mismos orígenes, %.2f veces, y la razón\n"
        "frente a ella pasa de %.3f a %.3f.\n%s"
        % (en["MAE"], pr["MAE"], salto, en["MAE_persistencia"], pr["MAE_persistencia"],
           sube_pers, en["razon"], pr["razon"], veredicto))
