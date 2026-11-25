# Pronóstico del precio de bolsa de la energía eléctrica en Colombia a 30 días

**Trabajo Integrador — Maestría en Analítica e Inteligencia de Negocios**
Universidad del Valle

---

## La pregunta

¿Qué familia de modelos predictivos ofrece el mejor desempeño para pronosticar el precio spot de la
energía eléctrica en Colombia a **30 días**, considerando variables operativas, hidrológicas,
climáticas y de mercado?

El trabajo no busca un modelo: busca una **comparación**. Varias familias se entrenan y evalúan bajo
condiciones idénticas —el mismo conjunto de variables, la misma partición, el mismo embargo, las
mismas métricas— porque solo así la diferencia entre ellas responde a la pregunta y no a una
diferencia de montaje experimental.

## Resultados

Cada modelo pronostica **los 30 días** siguientes a cada origen, y esa trayectoria se puntúa de tres
maneras. Las tres salen del mismo pronóstico y **no son comparables entre sí**; lo que importa es que
el orden entre modelos coincida en las tres. Coincide.

| Familia | Modelo | `pares_30d` | `promedio_30d` | `dia_30` |
|---|---|---:|---:|---:|
| Deep Learning | **GRU de 32 unidades, 15 épocas** | **163,17** | **132,47** | **215,56** |
| Estadística | SARIMAX (2,1,1)(1,0,0,7) sobre `log(precio)` | 170,49 | 136,70 | 219,52 |
| Híbrida | BiLSTM con atención y descomposición | 172,30 | 140,17 | 226,45 |
| Machine Learning | XGBoost, objetivo relativo | 175,59 | 143,17 | 232,74 |
| Deep Learning | LSTM de 32 unidades, 3 épocas | 176,93 | 144,83 | 233,48 |
| Referencia | Persistencia (el precio de hoy) | 178,37 | 145,66 | 239,27 |
| Referencia | Media del entrenamiento | 300,84 | 292,57 | 298,97 |

MAE en COP/kWh sobre 1.155 orígenes de prueba (2023-02-01 a 2026-03-31), que son 34.650 pares
origen-horizonte. `promedio_30d` compara el precio medio del mes y es la lectura que **reporta el documento**,
porque es la que responde la pregunta del trabajo; `pares_30d` mide los 30 días por separado y
`dia_30` conserva solo el horizonte más lejano, y las dos se conservan como comprobación de que el
orden entre modelos no depende de cómo se puntúe. El consolidado completo, con una fila por corrida y por lectura, está en
[`resultados_modelos.csv`](resultados_modelos.csv), que contiene además variantes exploratorias: no
se ordena por MAE sin filtrar antes por la configuración declarada.

Los hallazgos que conviene adelantar:

- **Gana el GRU, pero gana el conjunto de las cinco semillas, no una corrida.** Su desviación entre
  semillas es de 6,05 COP/kWh y **dos de las cinco quedan por detrás del SARIMAX** (139,62 y
  142,07). Los 4,24 COP/kWh que lo separan de la familia estadística equivalen a una vez y media el
  error típico de esa media. Ninguna cifra del GRU significa nada sin esa dispersión al lado.
- **El modelo más complejo no es el mejor.** El híbrido —descomposición, bidireccionalidad y
  atención— queda tercero, y con **menos parámetros que la LSTM**. La diferencia no es de tamaño
  sino de mecanismo. Dentro de la misma familia, el GRU y la LSTM comparten ventana, variables,
  rejilla y semillas, y se diferencian solo en la celda recurrente: esa única diferencia vale
  **12,37 COP/kWh**.
- **Los márgenes sobre la persistencia no siguen el orden de la complejidad**: 9,1 % el GRU, 6,2 %
  el SARIMAX, 3,8 % el híbrido, 1,7 % el XGBoost y 0,6 % la LSTM. El SARIMAX resuelve su pronóstico
  en segundos y queda segundo.
- **Solo dos modelos superan a la persistencia en los treinta horizontes**: el GRU, con holgura
  —su peor razón es 0,969—, y el híbrido, que lo hace por un margen inapreciable a quince días
  (razón 0,99999, una diferencia de 0,0014 COP/kWh). El SARIMAX gana en 29 de 30, el XGBoost en 28
  y la LSTM solo en 20. La ventaja del SARIMAX **crece** con el horizonte (1,002 a un día, 0,917 a
  treinta); el XGBoost y la LSTM hacen lo contrario.
- **Ningún modelo gana en todas partes.** Por fase climática y por año del día pronosticado, el
  SARIMAX es el mejor durante El Niño y en 2024; el GRU, en condiciones neutrales, durante La Niña y
  en 2025 y 2026; el híbrido, en 2023. Esto es lo que sostiene el análisis por régimen ENSO como
  aporte del trabajo.
- **En aprendizaje profundo, la duración del entrenamiento lo decide todo.** El error de validación
  tiene forma de U con el mínimo entre tres y quince épocas; a sesenta épocas la red queda muy por
  detrás de repetir el precio de hoy. Una rejilla que solo hubiera probado 30 y 60 épocas habría
  concluido que la familia es inservible, y es la familia que gana.
- **Las variables exógenas aportan muy poco.** En las dos familias donde la ablación se midió, más
  del 90 % de la mejora sobre la persistencia viene de la dinámica del propio precio. Quitarle al
  SARIMAX todas las exógenas cuesta 0,75 COP/kWh de MAE; al XGBoost, 0,16 — por debajo de su
  dispersión entre semillas. A este horizonte el precio spot colombiano es abrumadoramente
  autorregresivo. Pero el GRU recibe exactamente la misma entrada que la LSTM y la supera en 12,37:
  el límite de fondo está en la información disponible, y dentro de ese límite la arquitectura
  todavía decide una parte apreciable del error.
- **El error segmentado por fase ENSO engaña en términos absolutos.** El MAE varía en razón 2,2
  entre fases, pero el coeficiente de variación de la serie es casi el mismo en las tres (0,20 /
  0,19 / 0,16): medido en relativo, el modelo es **mejor** durante El Niño (25,5 % frente a 31,9 %
  en condiciones neutras). La evaluación por régimen reporta siempre una medida relativa junto a la
  absoluta.

## Cómo está organizado

El flujo completo —qué lee y qué escribe cada notebook, el protocolo común y el orden de
reejecución— está en **[PIPELINE.md](PIPELINE.md)**, con diagrama.

```
1_DataExtraction        API de XM, NOAA y DANE   ->  datasets/
2_DataProcessing        ETL y calendario         ->  1_dataset_unificado.csv
3_EDA_Consolidado       exploración, 21 figuras  ->  2_dataset_procesado.csv
4_SeleccionVariables    permutación y SHAP       ->  4_dataset_seleccion.csv
5_SARIMAX_modelo_base   familia estadística      ->  predicciones y métricas
6_XGBoost_modelo_base   machine learning         ->  predicciones y métricas
7_LSTM_modelo_base      aprendizaje profundo     ->  predicciones y métricas
8_GRU_modelo_base       aprendizaje profundo     ->  predicciones y métricas
9_Hibrido_BiLSTM_atencion_modelo_base
                        familia híbrida          ->  predicciones y métricas
```

Dos módulos propios: `_periods.py` (trocea el rango de fechas para la API de XM) y `_resultados.py`
(deja una fila por corrida en el consolidado, reemplazando la anterior si la configuración se
repite).

## Datos

**Los datos derivados sí viajan en este repositorio**, y son los que permiten reproducir los
resultados sin rehacer la extracción. En `datasets/procesado/` están los tres conjuntos de la
cadena —`1_dataset_unificado.csv`, `2_dataset_procesado.csv` y `4_dataset_seleccion.csv`— junto con
las predicciones, las métricas y el error por origen de los cinco modelos. En `datasets/` están las
series crudas de XM, NOAA y DANE que caben.

**Tres archivos crudos no caben** y hay que descargarlos con `1_DataExtraction.ipynb`:
`Gene.csv` (188 MB), `DispoDeclarada.csv` (157 MB) y `PrecOferDesp.csv` (65 MB). El límite de
GitHub es de 100 MB por archivo; el tercero cabría, pero no sirve de nada sin los otros dos, porque
el ETL necesita las tres. La extracción es **incremental**: toma solo lo que falta y repite los
últimos 90 días, porque XM ajusta las cifras recientes.

| Fuente | Qué aporta |
|---|---|
| **XM S.A. E.S.P.** (API pública, vía `pydataxm`) | 13 series del Mercado de Energía Mayorista: precio de bolsa, demanda, generación, aportes hídricos, volumen útil, vertimientos, precio de oferta, precio de escasez, compras y ventas |
| **NOAA** | Índice ONI (Oceanic Niño Index), en su versión **publicada** |
| **DANE** | IPC e IPP, anexos históricos |

Sobre el ONI: el índice centrado contiene información de hasta dos meses después de cada fecha. Se
usa la **versión publicada** —la que estaba disponible en la fecha del pronóstico— para modelar y
evaluar. Con la centrada, un modelo de árboles no superaba a la persistencia en ninguna semilla.

## Reproducir

```bash
conda env create -f requirements_py_3_10_14.yml
conda activate tesis_xm
```

Python **3.10.14**. Los notebooks se ejecutan en orden, **del 1 al 9**; cada uno consume lo que
produjo el anterior y, reejecutados, producen conjuntos idénticos. La cadena completa cuesta algo
más de **dos horas** de cómputo.

**Qué corre recién clonado el repositorio**, sin descargar nada:

| Notebook | ¿Corre desde el clon? | Qué le falta |
|---|---|---|
| `1_DataExtraction` | No, y no hace falta | Rehace la descarga desde la API de XM. Solo si se quiere ampliar el periodo |
| `2_DataProcessing` | **No** | Las tres series crudas que no caben (ver *Datos*). Su salida, `1_dataset_unificado.csv`, ya viene hecha |
| `3_EDA_Consolidado` | **Sí** | — |
| `4_SeleccionVariables` | **No** | `DispoDeclarada.csv`, de donde saca el reparto de la disponibilidad por tipo de recurso. Sus salidas ya vienen hechas |
| `5_SARIMAX` a `9_Hibrido` | **Sí** | — |

Es decir: **las cinco familias de modelos, que son lo que responde la pregunta del trabajo, se
reproducen desde el clon**. Los dos cuadernos que no corren son los que leen las series crudas
pesadas, y sus salidas se publican para que la cadena no se interrumpa.

El cuaderno 9 incluye además la comparación entre las cinco familias y la prueba de
Diebold-Mariano, de modo que la tabla de resultados de arriba se vuelve a obtener corriendo el 5,
el 6, el 7, el 8 y el 9 en ese orden.

Los nueve resuelven sus rutas contra el **directorio de trabajo**, que tiene que ser la raíz del
repositorio. [PIPELINE.md](PIPELINE.md) explica paso a paso cómo ejecutarlos desde Visual Studio
Code, con los tiempos de cada uno y las reglas que no conviene saltarse.

## Metodología

CRISP-DM. Partición **cronológica**, fijada por fecha y alineada a periodos completos —no como
fracción de filas—, porque varias variables son series mensuales propagadas a diario y la
evaluación por fase ENSO también se define sobre meses. Entre entrenamiento y prueba queda **todo
enero de 2023 como embargo**, un mes completo que cubre con holgura el horizonte de 30 días y que no
se emplea en ninguno de los dos bloques. El ajuste de hiperparámetros usa validación progresiva
sobre cinco pliegues dentro del entrenamiento, con purga de 30 días.

## Antecedente

La organización inicial de este repositorio se adaptó del trabajo de **Jusquini Tinoco, C. A.
(2024)**, *Predicción del precio en bolsa de la energía en Colombia en un horizonte de mediano
plazo* (Especialización en Analítica y Ciencia de Datos, Universidad de Antioquia), que comparó
SARIMAX con modelos de Deep Learning sobre el mismo mercado.

El alcance, el protocolo experimental, el conjunto de variables y los resultados de este trabajo son
propios y difieren de los suyos: aquí la partición es cronológica con embargo, la evaluación se
segmenta por régimen climático, y la comparación abarca familias que aquel trabajo no incluía.

## Licencia

El **código** de este repositorio se publica bajo licencia [MIT](LICENSE): se puede usar, copiar
y modificar citando la autoría.

**Los datos no son nuestros y no los cubre esa licencia.** Las series del mercado eléctrico son de
**XM S.A. E.S.P.**, el índice ONI es de la **NOAA** y el IPC y el IPP son del **DANE**; aquí viajan
derivados suyos para que los resultados se puedan reproducir, y cada fuente mantiene sus propios
términos de uso. Quien los reutilice debe acudir a la fuente original.

