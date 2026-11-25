# Todo explicado desde cero

**Guía conceptual del Trabajo Integrador: por qué se hace cada cosa, qué significa cada término y
qué hace cada línea de código.**

Este documento no asume que sabes nada. Define cada palabra antes de usarla, explica la razón de
cada decisión y recorre el código línea por línea. Si en algún punto aparece un término sin
explicar, es un error de este documento — está en el [glosario](#glosario) o no debería estar.

---

## Cómo leer este libro

| Capítulo | De qué trata | Cuándo volver a él |
|---|---|---|
| [1](#1-el-problema) | Qué es el precio de bolsa y por qué pronosticarlo | Al explicar el trabajo a alguien |
| [2](#2-los-datos) | De dónde sale cada número | Cuando dudes de una variable |
| [3](#3-el-vocabulario) | Estacionariedad, ACF, diferenciación, volatilidad | Antes de leer el notebook 3 o 5 |
| [4](#4-el-montaje) | Orígenes, partición, embargo, purga, fuga | **El más importante.** Antes de tocar cualquier modelo |
| [5](#5-cómo-se-mide) | MAE, RMSE, SMAPE, **las tres lecturas** y las referencias | Al interpretar cualquier resultado |
| [6](#6-elegir-las-variables) | Permutación, SHAP, VIF | Antes de leer el notebook 4 |
| [7](#7-el-sarimax) | El modelo estadístico, línea por línea | Antes de leer el notebook 5 |
| [8](#8-el-xgboost) | El modelo de árboles, línea por línea | Antes de leer el notebook 6 |
| [9](#9-la-lstm) | Qué es una red recurrente y qué hace la LSTM | Antes de leer el notebook 7 |
| [10](#10-el-gru-y-el-híbrido) | El GRU y el híbrido: atención y descomposición | Antes de leer los notebooks 8 y 9 |

---

# 1. El problema

## 1.1 Qué es el precio de bolsa

En Colombia la electricidad se compra y se vende en un mercado. Hay dos formas de hacerlo:

- **Por contrato.** Un generador y un comercializador acuerdan hoy un precio para entregar energía
  dentro de meses o años. El precio queda fijo.
- **En bolsa.** Cada hora se decide quién genera y a qué precio, según lo que se necesite ese
  momento. Ese precio es el **precio de bolsa** o **precio spot**.

El precio de bolsa de cada hora se fija así: todos los generadores dicen a qué precio están
dispuestos a producir. El operador los ordena de más barato a más caro y los va llamando hasta
cubrir toda la demanda. **El precio que cobran todos es el de la última planta que entró** — la
«planta marginal».

Eso explica el comportamiento de la serie. Cuando llueve, las hidroeléctricas —baratas— cubren casi
todo y el precio es bajo. Cuando hay sequía, los embalses bajan, hay que encender térmicas que
queman gas o carbón, y la última planta en entrar es mucho más cara. El precio se dispara.

> **Por qué importa.** Un comercializador que compra en bolsa sin cubrirse queda expuesto a esos
> picos. Un generador que vende todo por contrato se pierde las subidas. Saber hacia dónde va el
> precio es la diferencia entre gestionar el riesgo y sufrirlo.

## 1.2 Qué significa «pronosticar a 30 días»

Cada día del periodo de prueba es un **origen**, y para cada origen el modelo produce **los treinta
valores** de la ventana que empieza al día siguiente: no un número, sino una **trayectoria**.

```
origen = 1 de febrero
    │
    ├─ día  1 ──▶  2 de febrero   ┐
    ├─ día  2 ──▶  3 de febrero   │
    ├─  ...                        ├─ los 30 valores que el modelo entrega
    ├─ día 29 ──▶  1 de marzo     │
    └─ día 30 ──▶  2 de marzo     ┘
```

Con 1.155 orígenes de prueba y 30 días cada uno, el trabajo produce **34.650 pronósticos**
—a estos pares (origen, día) se les llama **pares origen-horizonte**—, todos hechos solo con la
información disponible en su origen.

> **¿Y por qué no pronosticar solo el día 30?** Porque el día 30 aislado responde a menos de lo que
> el problema pregunta. Buena parte del interés está en el **precio medio del mes**, que es lo que
> un agente necesita para decidir si se cubre, y ese promedio exige los días intermedios. Entregar
> la trayectoria completa permite además las tres lecturas del capítulo siguiente, y el día 30
> sigue estando ahí: es una de ellas.

**Por qué 30 días y no 1 o 365.** Es el horizonte de las decisiones comerciales del mercado: con
30 días de anticipación un agente todavía puede contratar, cubrirse o ajustar su posición. A un día
no hay margen de maniobra; a un año, el pronóstico pierde todo valor práctico.

**La misma trayectoria se puede puntuar de tres maneras**, y las tres responden preguntas
distintas. Es un punto central y tiene su propia sección: [§5.2](#52-las-tres-lecturas-de-un-mismo-pronóstico).

**Un dato que ilustra lo difícil que es.** La «persistencia» —decir simplemente «el precio seguirá
siendo el de hoy»— se equivoca en promedio en **178 COP/kWh** sobre los 34.650 pares. Mirando solo
el día 30 se equivoca en **239**; mirando solo el promedio del mes, en **146**. Las tres cifras
describen el mismo pronóstico trivial: lo único que cambia es qué se resume.

## 1.3 Qué pregunta responde el trabajo

> ¿Qué **familia** de modelos ofrece el mejor desempeño para pronosticar el precio spot a 30 días?

La palabra clave es **familia**. El trabajo no busca «el mejor modelo»: busca saber si para este
problema conviene más la estadística clásica, el machine learning de árboles, las redes neuronales
o los híbridos. Por eso todos se entrenan y evalúan bajo **condiciones idénticas**: mismos datos,
misma partición, mismas métricas. Si cada familia usara su propio montaje, la comparación mediría
los montajes y no las familias.

---

# 2. Los datos

## 2.1 Qué es una serie de tiempo

Una **serie de tiempo** es una sucesión de valores ordenados en el tiempo, uno por periodo. Aquí:
un precio por día, desde el 1 de enero de 2010 hasta el 30 de abril de 2026. Son **5.964 días**.

Lo que la distingue de una tabla normal es que **el orden importa**. En una tabla de clientes puedes
barajar las filas y no pasa nada. En una serie de tiempo, barajar destruye la información: el precio
de hoy depende del de ayer. Esa dependencia se llama **autocorrelación** y es, literalmente, la
materia prima del pronóstico.

## 2.2 De dónde sale cada número

| Fuente | Qué entrega | Cómo se obtiene |
|---|---|---|
| **XM S.A. E.S.P.** | 13 series del mercado eléctrico | API pública, vía la librería `pydataxm` |
| **NOAA** | Índice ONI (temperatura del Pacífico) | Un archivo de texto en su sitio |
| **DANE** | IPC e IPP (inflación) | Anexos en Excel |

XM es el operador del mercado eléctrico colombiano: la entidad que ejecuta el despacho que
describimos arriba. Su API es pública y no requiere credenciales.

## 2.3 Las variables, una por una

**La que se quiere pronosticar:**

- `precio_bolsa` — el precio spot promedio del día, en COP/kWh.

**Las hidrológicas** (el agua, que es lo que gobierna el precio en Colombia):

- `aportes_energia` — cuánta energía trae el agua que llega a los embalses. Es la *entrada*.
- `vol_util` — cuánta energía hay almacenada en los embalses. Es el *stock*.
- `vertimientos` — agua que se deja salir sin generar, porque el embalse está lleno. Es señal de
  abundancia: si estás botando agua, el precio no va a subir.

**Las de operación:**

- `demanda` — cuánta energía consume el país.
- `generacion` y sus componentes (`gen_hidraulica`, `gen_termica`, `gen_no_termica`) — cuánta
  energía produce cada tecnología.
- `disponibilidad_declarada` y sus tipos — cuánta capacidad dicen los generadores que pueden
  entregar.
- `precio_oferta` — a qué precio ofrecen las plantas.
- `precio_escasez` — un tope regulatorio que la CREG calcula y que sube cuando suben los
  combustibles. **Funciona como representante de los precios de los combustibles**, que no entraron
  al trabajo por cobertura parcial.

**La climática:**

- `indice_ONI` — el Oceanic Niño Index, que mide la temperatura del océano Pacífico. Positivo = El
  Niño (sequía en Colombia), negativo = La Niña (lluvia).

**Las de calendario:**

- `mes`, `dia_semana`, `tipo_dia` (ordinario, sábado, domingo, festivo), `dia_mes`.

> **Por qué el calendario es especial.** Todas las demás variables hay que *pronosticarlas* para
> usarlas en el futuro. El calendario **se conoce de antemano**: sabes con certeza que el 25 de
> diciembre de 2027 es festivo. Es la única información del futuro que se puede usar sin trampa.

**Esta lista es de candidatas, no de las que acabaron en los modelos.** De todas ellas la selección
del capítulo [6](#6-elegir-las-variables) deja **cinco**: `precio_bolsa`, `aportes_energia`,
`vertimientos`, `gen_hidraulica` y `dia_semana`. Las demás se descartan, cada una con su motivo
registrado en `4_importancia_por_bloque.csv`. Que `vol_util`, `precio_oferta` y el ONI queden fuera
es contraintuitivo y tiene su explicación en [§6.3](#63-importancia-por-permutación).

## 2.4 El ONI y una trampa que casi arruina el trabajo

El ONI se publica de dos maneras y la diferencia es crítica.

- **La versión centrada** promedia tres meses *alrededor* de cada mes. El valor de marzo usa datos
  de febrero, marzo y abril.
- **La versión publicada** es la que estaba disponible en cada fecha. El valor de marzo se conoce
  hacia el 6 de mayo.

Si usas la centrada para pronosticar, **el modelo ve el futuro**: el ONI de marzo contiene
información de abril, y tú estás tratando de pronosticar abril.

Esto no es teórico. Está medido, y el registro está en `4_comparacion_oni_centrado_publicado.csv`:
con la versión centrada, un modelo de árboles **no superaba a la persistencia en ninguna de diez
semillas** (MAE medio 279,9 frente a 228,0 con la publicada), las variables del índice acaparaban el
**67 %** de la importancia en vez del 33 %, y el error durante El Niño se disparaba a **406** COP/kWh
frente a **256**. El modelo se había apoyado en un índice que, en el entrenamiento, anticipaba el mes
siguiente — y esa muleta desapareció en la prueba.

> **De dónde salen esas cifras.** De un diagnóstico hecho con el XGBoost **en nivel** y el objetivo
> del día 30, que era el montaje del momento. No son cifras del protocolo actual y no hay que
> compararlas con las de los capítulos 7 y 8; su función es medir **la diferencia entre las dos
> versiones del índice** bajo un mismo montaje, y para eso siguen valiendo. Repetir el diagnóstico
> con el montaje de hoy daría otros números y la misma conclusión.

**Regla:** la versión centrada sirve para *describir* el pasado (el EDA). La publicada es la única
que puede *pronosticar* y *evaluar*.

**Y hay una segunda regla, que este trabajo aprendió después:** que el ONI publicado sea el único
admisible **no lo convierte en predictor útil**. No pasó la regla de selección
([§6.3](#63-importancia-por-permutación)). Se conserva en el dataset, fuera de las predictoras,
porque sirve para **etiquetar** la fase ENSO y reportar el error por régimen
([§5.5](#55-absoluto-frente-a-relativo-la-trampa-de-la-segmentación)).

---

# 3. El vocabulario

Este capítulo define los términos de series de tiempo que aparecen en el notebook 3 y en el
documento. Cada uno viene con la cifra real del trabajo.

## 3.1 Estacionariedad

Una serie es **estacionaria** si sus propiedades no cambian con el tiempo: la media se mantiene
alrededor del mismo valor, la variabilidad es parecida, y la relación entre un día y el siguiente es
siempre la misma.

**Por qué importa.** Casi todos los modelos estadísticos asumen que lo que aprendieron del pasado
sigue valiendo. Si la serie deriva —si la media de 2024 no tiene nada que ver con la de 2012—, lo
aprendido no se transfiere.

El precio de bolsa **no** es estacionario, y se nota a simple vista: la tendencia promedia 107
COP/kWh en 2010 y 617 en 2024.

## 3.2 Raíz unitaria, y las dos pruebas

Una **raíz unitaria** es la forma matemática de decir «esta serie no tiene un ancla». Formalmente,
si el precio de mañana es `precio de hoy × φ + ruido`:

- Si **φ < 1**, la serie revierte: tras un golpe, vuelve poco a poco a su media.
- Si **φ = 1**, la serie **no revierte**: cada golpe queda incorporado para siempre. Eso es una raíz
  unitaria, y se llama *caminata aleatoria*.

Se contrasta con dos pruebas que tienen **hipótesis nulas opuestas** — y esto es lo que confunde a
todo el mundo:

| | Su hipótesis nula (H₀) | Rechazarla significa |
|---|---|---|
| **ADF** (Dickey-Fuller aumentada) | «hay raíz unitaria» | «es estacionaria» |
| **KPSS** | «es estacionaria» | «hay raíz unitaria» |

**Qué es una «hipótesis nula».** Es la afirmación por defecto, la que se mantiene salvo que los
datos la contradigan con fuerza. El **p-valor** mide cuán raros serían los datos observados *si la
hipótesis nula fuera cierta*. Un p-valor menor a 0,05 se toma como suficiente para rechazarla.

**Lo que pasó en este trabajo: las dos rechazaron.**

```
ADF       estadístico −4,9627   p = 0,0000   →  "es estacionaria"
KPSS      estadístico  2,2942   p = 0,0100   →  "NO es estacionaria"
          crítico al 1 %: 0,7390
```

El estadístico KPSS es **más de tres veces** el valor crítico más exigente. El p-valor de 0,0100
está truncado: `statsmodels` avisa que el verdadero es menor, porque se salió de la tabla.

Se hizo una tercera prueba, el KPSS «alrededor de una tendencia» (`regression="ct"`), que sería la
explicación benigna del conflicto: quizá la serie es estacionaria una vez le quitas una tendencia
lineal. También se rechazó (0,2804 frente a 0,2160), aunque por mucho menos margen.

**Se le cree a KPSS.** Razón: el ADF pierde potencia en series muy persistentes — cuando φ está
cerca de 1 pero no es exactamente 1, el ADF tiende a confundirse. Y hay una tercera evidencia
independiente que lo respalda: la ACF (siguiente sección).

## 3.3 Autocorrelación: ACF y PACF

La **ACF** (función de autocorrelación) responde: *¿cuánto se parece el precio de hoy al de hace
k días?* Se calcula para k = 1, 2, 3, …

La **PACF** (autocorrelación parcial) responde algo más fino: *¿cuánto se parecen, descontando lo
que se transmite a través de los días intermedios?* Si hoy se parece a ayer, y ayer a anteayer,
entonces hoy se parece a anteayer **por cadena**. La PACF quita ese efecto indirecto.

**Lo que muestran en este trabajo:**

```
ACF    rezago 1: 0,971    rezago 7: 0,872    rezago 30: 0,652    rezago 60: 0,507
PACF   rezago 1: 0,972    rezago 2: 0,143    rezago 3:  0,114
```

**Cómo se lee esto:**

- La ACF decae **despacio y casi en línea recta**. Una serie estacionaria pierde memoria de forma
  *exponencial* (rápido al principio, luego cada vez menos). Un decaimiento lineal es la firma de
  una raíz unitaria. **Tercera evidencia a favor de KPSS.**
- La PACF tiene **un único pico dominante** y luego casi nada. Eso describe un proceso
  autorregresivo de **primer orden**: lo que importa es el día anterior, y lo demás llega por
  cadena.

**Una cautela que hay que declarar.** Con 5.964 observaciones, la banda de confianza al 95 % mide
apenas **±0,025**. Decenas de rezagos salen «estadísticamente significativos» con magnitudes sin
ninguna relevancia práctica. **El criterio de lectura es el tamaño del coeficiente, no que
sobrepase la banda.**

## 3.4 Diferenciar

**Diferenciar** es trabajar con el *cambio* en vez del *nivel*: en lugar de modelar el precio, se
modela `precio(t) − precio(t−1)`.

Es la receta estándar contra la raíz unitaria: una serie que no revierte, convertida en sus cambios,
casi siempre sí revierte. El número de veces que se diferencia se llama **`d`**.

**Qué pasó al diferenciar una vez:**

```
                        ADF              KPSS         (crítico 5 %: 0,463)
diff(precio)          −15,9             0,007
diff(log(precio))     −15,3             0,012
```

Las dos pruebas **dejan de contradecirse**: ambas dicen «estacionaria». Con una sola diferencia
basta, de modo que `d = 1`.

**Y aparece lo que el nivel ocultaba.** Sobre la serie diferenciada se ven picos en los rezagos
**7, 14, 21, 28 y 35** (entre 0,15 y 0,18): el patrón semanal. Y **ningún pico en el rezago 365**:
no hay eco anual que justifique un componente estacional de un año.

> **Por qué el nivel lo ocultaba.** Cuando la ACF de la serie en nivel vale 0,971 en el primer
> rezago, todo lo demás queda aplastado contra ese valor. Diferenciar quita la persistencia y deja
> ver la estructura fina.

## 3.5 Descomposición: tendencia, estacionalidad y residuo

**Descomponer** es partir la serie en tres piezas que suman el original:

```
serie = tendencia + estacionalidad + residuo
```

- **Tendencia**: hacia dónde va a largo plazo. Aquí es una media móvil centrada de 365 días.
- **Estacionalidad**: el patrón que se repite cada año.
- **Residuo**: lo que queda.

**El reparto de la varianza en este trabajo:**

```
tendencia    42,9 %
estacional    7,5 %
residuo      42,1 %
```

**Cómo se lee.** Un esquema de «tendencia + estacionalidad» deja sin explicar casi la mitad de la
variación — y lo que deja fuera son precisamente los episodios extremos, que es lo que se quiere
pronosticar.

### El artefacto de octubre: una lección sobre estimadores

La componente estacional del método clásico sitúa su máximo anual en **octubre**. Eso contradice la
hidrología colombiana, donde la temporada seca es entre diciembre y marzo.

**Por qué pasa.** El método estima la estacionalidad **promediando**, para cada posición del año, la
serie sin tendencia. Hay solo diecisiete ciclos disponibles, y **dos** episodios de El Niño que
culminaron entre septiembre y octubre. Con tan pocos ciclos, dos valores enormes arrastran el
promedio.

**Qué pasa al usar la mediana**, que es insensible a valores extremos:

| | Con media | Con mediana |
|---|---|---|
| Máximo anual | octubre | **febrero** |
| Amplitud del ciclo | 217,5 COP/kWh | **87,6 COP/kWh** |

Y el perfil robusto correlaciona **−0,868** con el de los aportes hídricos: más agua, menos precio.
Exactamente lo que predice el mecanismo físico.

> **La lección, que vale más allá de este caso:** con pocos ciclos y regímenes extremos, un
> estimador que promedia no estima la estacionalidad — estima los episodios.

**Consecuencia para el trabajo:** existe un ciclo anual real y de origen hidrológico, pero su
amplitud es modesta frente a la variación que introducen los episodios climáticos. Por eso la
especificación estadística **no** incorpora componente estacional anual y **sí** uno semanal.

## 3.6 Volatilidad: por qué el logaritmo

**Volatilidad** es cuánto se mueve la serie. Se mide con la desviación estándar en una ventana
móvil: para cada día, la dispersión de los últimos 30 días.

La pregunta que decide la escala del modelo es: **¿la dispersión es constante, o crece con el
nivel?**

```
Correlación entre el nivel local y la dispersión local: 0,813
```

Crece con el nivel. Cuando el precio está en 150 se mueve unos pocos pesos; cuando está en 800, se
mueve en decenas. Eso se llama comportamiento **multiplicativo** (frente a *aditivo*), o
**heterocedasticidad** (varianza no constante).

**Qué se hace al respecto: modelar el logaritmo.**

El logaritmo convierte multiplicaciones en sumas. Sobre esa escala, una variación del 10 % pesa lo
mismo con el precio en 100 que en 2.000 — que es el supuesto que la serie **sí** cumple.

No es una convención heredada; está medido. En la rejilla de validación del SARIMAX, la misma
especificación da razón **0,975** en logaritmo y **1,023** en nivel. La versión en nivel es peor que
la persistencia.

### La consecuencia que no es sobre modelar, sino sobre evaluar

El **coeficiente de variación** es la dispersión dividida por el nivel: cuánto se mueve la serie *en
proporción a lo que vale*.

```
            nivel mediano   dispersión   coef. de variación
El Niño          261,27        52,93            0,20
Neutral          155,70        28,81            0,19
La Niña          121,96        19,70            0,16
```

Es casi el mismo en las tres fases. **El Niño no es un régimen intrínsecamente más difícil de
pronosticar**: su mayor dispersión absoluta es *proporcional* a su mayor nivel.

Y de ahí sale una regla de reporte que el trabajo adopta: **la evaluación segmentada por fase
reporta siempre una medida relativa junto a la absoluta.** Capítulo 5 explica por qué es
imprescindible.

---

# 4. El montaje

**Este es el capítulo más importante del libro.** Un montaje mal hecho produce resultados
excelentes y falsos, y nada en el código avisa.

## 4.1 Fuga de información

**Fuga de información** (*data leakage*) es cuando el modelo usa, para entrenarse, información que
en la práctica no habría estado disponible. El resultado es un modelo que parece brillante en el
laboratorio y fracasa en producción.

En series de tiempo la fuga es especialmente fácil de cometer, porque basta con mezclar el orden.
Todo lo que sigue en este capítulo existe para impedirla.

## 4.2 Un «origen» y cómo se arma la tabla

El problema se convierte en una tabla donde **cada fila es un origen**:

| fecha (origen) | variables conocidas en esa fecha | `precio_hoy` | `fecha_objetivo` | `objetivo` | `objetivo_dia30` |
|---|---|---|---|---|---|
| 2023-02-01 | aportes, vertimientos, demanda, … | 769,0 | 2023-03-03 | 506,7 | 435,5 |

- **`precio_hoy`** es el precio del día del origen. Sirve de punto de partida y de referencia.
- **`objetivo`** es el **precio medio de los 30 días** del 2 de febrero al 3 de marzo: 506,7.
- **`objetivo_dia30`** es el precio del **último** día de esa ventana, el 3 de marzo: 435,5.
- **`fecha_objetivo`** es la fecha de ese último día, y es la columna que gobierna la purga
  ([§4.5](#45-la-purga-que-es-distinta-del-embargo)): marca cuándo se conoce la ventana completa.

En el ejemplo se ve por qué el horizonte no es un detalle: el precio pasó de 769 a 435, y el
promedio del camino —506,7— no es ninguno de los dos.

**Dos columnas de objetivo, y ninguna sobra.** El modelo que pronostica el promedio del mes se
entrena contra `objetivo`; cada uno de los treinta modelos que arman la trayectoria se entrena
contra el precio de *su* día, reconstruido sobre la serie diaria. `objetivo_dia30` existe además
porque es la única columna que lleva el precio de días posteriores al último origen, y con ella se
completa la serie hasta el 30 de abril de 2026 —el día 30 del último origen—; sin eso, los
horizontes largos de los últimos orígenes se quedarían sin valor observado contra el que medirse.

> **Cuidado al añadir columnas.** `objetivo_dia30` es el precio futuro: si entra como predictora, el
> modelo lee la respuesta. Los notebooks la excluyen por nombre y lo comprueban con un `assert`, no
> por diferencia de conjuntos — el capítulo [6.4](#64-el-vehículo-de-medición) cuenta por qué esa
> distinción importó.

## 4.3 La partición: entrenamiento, embargo y prueba

**Entrenar** es dejar que el modelo aprenda de unos datos. **Probar** es medirlo en datos que nunca
vio. Si se mide sobre lo mismo con lo que aprendió, el resultado no significa nada: es como
calificar un examen con las respuestas a la vista.

```
┌──────────── ENTRENAMIENTO ────────────┐ ┌ EMBARGO ┐ ┌──── PRUEBA ────┐
  2010-06-30  ────────────  2022-12-31     enero 2023   2023-02-01 ── 2026-03-31
       4.568 orígenes                       31 días        1.155 orígenes (20,1 %)
```

**Por qué la partición es cronológica y no aleatoria.** Si se repartieran las filas al azar, el
modelo entrenaría con días de 2025 para pronosticar días de 2015. En la vida real eso es imposible:
no se puede usar el futuro. La partición tiene que respetar el tiempo.

**Por qué las fronteras se fijan por fecha y no como «el 80 % de las filas».** Varias variables (el
ONI, los índices de precios, el precio de escasez) son series **mensuales** propagadas a diario, y
la evaluación por fase ENSO también se define sobre meses. Un corte a mitad de mes dejaría días
sueltos con el valor de un mes que quedó del otro lado de la frontera.

## 4.4 El embargo: el concepto más sutil

**El problema.** El último origen del entrenamiento es el 31 de diciembre de 2022. Su objetivo es el
precio del 30 de enero de 2023. Si la prueba empezara el 1 de enero de 2023, los primeros orígenes
de prueba estarían pronosticando días que el entrenamiento **ya usó como respuesta correcta**.

**La solución.** Se deja un hueco entre los dos bloques que nadie usa. Aquí: **todo enero de 2023**,
31 días, que cubre con holgura el horizonte de 30.

**Por qué un mes completo y no exactamente 30 días.** Por la misma razón que las fronteras van por
fecha: para no partir un mes por la mitad.

## 4.5 La purga, que es distinta del embargo

El embargo separa el entrenamiento de la prueba. La **purga** resuelve el mismo problema *dentro*
del entrenamiento, cuando se valida.

En el código:

```python
def conjunto_ajuste(corte):
    """Lo observable el día `corte`: purga de 30 días y embargo permanente de enero de 2023."""
    return s[(s["fecha_objetivo"] < corte) & ~EN_EMBARGO]
```

**Léelo con cuidado, porque el detalle está en qué columna filtra.** No filtra por `fecha` (el
origen) sino por **`fecha_objetivo`**. La condición dice: *quédate solo con las filas cuyo resultado
ya se conocía en la fecha de corte*.

Si filtrara por `fecha`, entrarían orígenes de hace 10 días cuyo objetivo todavía no ha ocurrido —
y el modelo estaría aprendiendo de respuestas que aún no existen. Esa sola línea es la que impide la
fuga, y es fácil escribirla mal sin que nada falle.

## 4.6 Validación progresiva (walk-forward)

**El problema.** Para elegir los hiperparámetros hay que probar configuraciones y comparar. Pero no
se puede usar la prueba para eso: si eliges tus parámetros mirando la prueba, la prueba deja de ser
una prueba.

**La solución.** Se parte el entrenamiento en **pliegues** sucesivos, cada uno ajustado con todo lo
anterior y validado con lo siguiente:

```
pliegue 1:  ajusta hasta 2012-12  →  valida 2013-2014   (886 orígenes de ajuste)
pliegue 2:  ajusta hasta 2014-12  →  valida 2015-2016   (1.616)
pliegue 3:  ajusta hasta 2016-12  →  valida 2017-2018   (2.347)
pliegue 4:  ajusta hasta 2018-12  →  valida 2019-2020   (3.077)
pliegue 5:  ajusta hasta 2020-12  →  valida 2021-2022   (3.808)
```

Esto imita cómo se usaría el modelo de verdad: entrenar con lo que hay, pronosticar lo que viene,
volver a entrenar cuando llegan más datos.

**Una trampa importante: los pliegues no son comparables entre sí.** El precio medio de validación
va de 118 COP/kWh (2017-18) a 336 (2015-16, El Niño). El MAE de la persistencia varía de 35,5 a
150,7 según el pliegue. Promediar MAE entre pliegues mezclaría peras con manzanas.

**La solución que adopta el trabajo:** comparar configuraciones por la **razón** entre el error del
modelo y el de la persistencia en cada pliegue. Una razón de 0,92 significa «este modelo comete el
92 % del error que cometería no hacer nada», y eso **sí** es comparable entre periodos.

## 4.7 Reajuste cada 60 días

Una vez elegida la configuración, al pronosticar la prueba hay una decisión más: ¿se reestiman los
parámetros a medida que llegan datos nuevos?

El trabajo reajusta **cada 60 días** y, entre reajustes, actualiza solo el *estado* del modelo día a
día. Es un equilibrio: reestimar cada día sería costosísimo y no estimar nunca desperdiciaría la
información nueva.

---

# 5. Cómo se mide

## 5.1 Las tres métricas

Las tres comparan el pronóstico con lo que de verdad pasó, y las tres se calculan **sobre el precio
reconstruido en COP/kWh**, nunca sobre el logaritmo — porque un error en logaritmos no se puede
interpretar.

### MAE — Mean Absolute Error

Para cada valor pronosticado, |real − pronóstico|. Se promedian todos.

Su virtud es que **está en las unidades de la variable**. Un MAE de 170,5 significa: «el pronóstico
se equivoca, en promedio, en 170,5 COP/kWh».

**Cuántos valores se promedian depende de la lectura**, y es lo primero que hay que decir al dar un
MAE: 34.650 si se cuentan los 30 días de cada origen por separado, 1.155 si antes se promedia cada
ventana. Es el tema de la sección siguiente.

### RMSE — Root Mean Squared Error

Igual, pero elevando los errores al cuadrado antes de promediar, y sacando la raíz al final.
Al elevar al cuadrado, **castiga mucho más los errores grandes**.

### SMAPE — Symmetric Mean Absolute Percentage Error

El error como porcentaje, usando el promedio entre real y pronóstico como denominador. Al ser
porcentual, es comparable entre series de distinta escala.

> **Por qué SMAPE y no MAPE.** El MAPE divide por el valor real, de modo que explota cuando el real
> se acerca a cero y penaliza asimétricamente: subestimar nunca puede costar más del 100 %,
> sobrestimar no tiene techo. El SMAPE reparte el denominador entre los dos y evita ambos problemas.

## 5.2 Las tres lecturas de un mismo pronóstico

Cada origen produce **una trayectoria de 30 valores**. Esa trayectoria se puede resumir de tres
maneras, y las tres responden preguntas distintas:

| Lectura | Qué hace | Cuántos valores | La pregunta que responde |
|---|---|---:|---|
| **`pares_30d`** | mide los 30 días **por separado** | 34.650 | ¿qué tan bueno es el pronóstico completo? |
| **`promedio_30d`** | promedia los 30 días y **compara los promedios** | 1.155 | ¿cuánto costará la energía este mes? |
| **`dia_30`** | conserva solo el **último** día | 1.155 | ¿qué tan lejos puede ver el modelo? |

Con el mismo pronóstico del SARIMAX:

| | `pares_30d` | `promedio_30d` | `dia_30` |
|---|---:|---:|---:|
| Persistencia | 178,4 | 145,7 | 239,3 |
| SARIMAX | **170,5** | **136,7** | **219,5** |
| razón | 0,96 | 0,94 | 0,92 |

**Las tres cifras describen el mismo pronóstico.** No es que un modelo mejore al cambiar de lectura:
es que se está resumiendo otra cosa.

### Por qué el promedio del mes da un error menor

Esta es la parte que hay que entender, porque parece una trampa y no lo es.

Los errores diarios tienen signo. Si el modelo se queda corto 60 el día 7 y se pasa 60 el día 20,
sobre los **pares** eso cuenta como 120 de error; sobre el **promedio** de la ventana los dos se
cancelan y el error del promedio es cero. Matemáticamente:

```
| promedio de los errores |  ≤  promedio de los | errores |
```

La desigualdad es el **valor absoluto de una suma frente a la suma de los valores absolutos**, y es
siempre cierta. Por eso `promedio_30d` nunca puede dar un MAE mayor que `pares_30d`. Medido aquí,
para la persistencia: 178,4 → 145,7, una reducción del **18 %** que no viene de pronosticar mejor
sino de que los errores se compensan.

> **La consecuencia práctica.** Un MAE de 136,7 y uno de 170,5 pueden ser el mismo modelo. Comparar
> dos cifras de lecturas distintas produce un ranking falso, y es exactamente el error que este
> trabajo le encuentra a parte de la literatura que revisa ([§5.3](#53-un-mae-solo-no-significa-nada)).
> Por eso `resultados_modelos.csv` tiene una columna obligatoria `objetivo`, y `registrar()`
> **falla** si no se la pasan: no es una convención, es un impedimento.

> **Y el `objetivo` no basta.** Impide mezclar lecturas, pero no impide mezclar **épocas**. En
> octubre de 2026 el consolidado conservaba nueve filas de una selección de variables anterior,
> cada una con su `objetivo` correcto, y al ordenar por MAE ponían una configuración superada por
> delante de la vigente. Sobrevivían porque la clave que decide si una fila reemplaza a otra
> incluye la configuración —lo correcto: dos configuraciones son dos resultados—, de modo que al
> cambiar los hiperparámetros las viejas quedaban huérfanas e indistinguibles de las buenas.
>
> La solución es un modelo de **propiedad**: `olvidar(notebook)` retira todas las filas de ese
> notebook y se llama una vez, antes de su primer `registrar()`. Una reejecución deja entonces
> exactamente las filas de su corrida actual. La lección general: un archivo que acumula resultados
> necesita saber **quién es dueño de qué**, o acumula también los que ya no valen.

### Cuál es «la» métrica del trabajo

**`pares_30d`**, porque califica lo que el modelo realmente entrega: los treinta valores. Las otras
dos se reportan al lado porque responden preguntas legítimas —el promedio del mes es la pregunta
comercial, y el día 30 es el caso más difícil— y porque **el orden entre modelos debe coincidir en
las tres**. Si coincide, la conclusión no depende de cómo se decidió puntuar. En el SARIMAX
coincide: la misma configuración queda primera en las tres lecturas y la de contraste (`d = 0`)
queda última en las tres.

Las tres se calculan en **un solo sitio**, la función `lecturas()` de `_resultados.py`, compartida
por todos los notebooks:

```python
def lecturas(trayectoria, reales):
    ok = ~np.isnan(trayectoria).any(axis=1) & ~np.isnan(reales).any(axis=1)
    t, r = trayectoria[ok], reales[ok]
    return {
        "pares_30d":    metricas(r.ravel(), t.ravel()),        # los 30 días por separado
        "promedio_30d": metricas(r.mean(axis=1), t.mean(axis=1)),
        "dia_30":       metricas(r[:, -1], t[:, -1]),
    }
```

Está ahí y no en cada notebook por una razón concreta: si cada familia construyera sus propios
resúmenes, la comparación entre familias podría estar midiendo las diferencias entre los resúmenes.

> **El `ok` de la primera línea no es cosmético.** Descarta todo origen que tenga un hueco en
> cualquiera de sus 30 columnas. Si la trayectoria de un modelo pierde los horizontes largos de sus
> últimos orígenes, ese modelo se acaba midiendo sobre menos orígenes que los demás **y nada lo
> avisa**. Pasó en el XGBoost y hubo que corregirlo; el notebook ahora lo comprueba con un `assert`.

## 5.3 Un MAE solo no significa nada

170,5 no es «bueno» ni «malo» en el vacío. Por eso el protocolo fija dos **referencias ingenuas**,
y las tres lecturas se reportan siempre contra ellas:

| | `pares_30d` | Qué representa |
|---|---:|---|
| Media del entrenamiento | 300,8 | «no sé nada, digo el promedio histórico» |
| **Persistencia** (el precio de hoy) | **178,4** | «el mejor pronóstico es que nada cambie» |
| **SARIMAX** | **170,5** | **−4,4 % frente a la persistencia** |

**La persistencia es la vara dura.** En series financieras y de energía es notoriamente difícil de
batir: si el precio es casi una caminata aleatoria, el mejor pronóstico *es* el valor actual. Que la
media del entrenamiento se equivoque un 69 % más que la persistencia da la medida del asunto.
**Ganarle es el resultado** — no el 170,5 en sí.

## 5.4 El MAE es una media, no «el mes típico»

Esta es la trampa que más confunde, y en estos datos es grande. Sobre la lectura
`promedio_30d` del mejor SARIMAX, los 1.155 errores se reparten así:

| | COP/kWh |
|---|---:|
| Percentil 25 del \|error\| | 37,5 |
| **Mediana** | **85,1** |
| **MAE** (la media) | **136,7** |
| Percentil 75 | 177,3 |
| Percentil 90 | 307,1 |
| Máximo | 1.147,6 |

**La mitad de los orígenes el error está por debajo de 85, no de 137.** El MAE está arrastrado por
la cola:

- el **10 % peor** de los orígenes aporta el **38,0 %** de todo el error;
- el **5 % peor**, el 23,8 %;
- el **1 % peor** (12 orígenes), el 6,3 %.

Casi cuatro décimas del error las producen 116 orígenes de 1.155. El modelo acierta razonablemente
casi siempre y se desploma en unos pocos episodios — que son, precisamente, los picos de El Niño.

**El cociente RMSE/MAE lo confirma numéricamente:** vale **1,51**. Si los errores siguieran una
distribución normal valdría ≈ 1,25. Que sea más alto es la firma de una **cola pesada**.

> Esto justifica, de paso, que el XGBoost se ajuste con pérdida absoluta y no cuadrática: una
> pérdida cuadrática concentraría todo el entrenamiento en unas pocas decenas de episodios extremos.

**Por qué conviene decir «la mediana es 85».** Un lector que oye «se equivoca en 137» imagina que
cada mes falla por ahí. No es así: falla bastante menos la mayoría de las veces, y mucho más unas
pocas. Las dos cifras juntas describen el modelo; el MAE solo, no.

## 5.5 Absoluto frente a relativo: la trampa de la segmentación

El MAE hereda la escala de la serie. Sobre un precio medio de prueba de 466,6:

```
MAE 136,7 / 466,6 = 29,3 % de error relativo
```

Por eso comparar MAE entre periodos de distinto nivel de precio **compara los precios, no los
modelos**. Es lo que ocurre al segmentar por fase ENSO:

| Fase | n | precio medio | MAE absoluto | MAE relativo | SMAPE |
|---|---:|---:|---:|---:|---:|
| El Niño | 335 | 599,3 | 152,5 | **25,5 %** | 25,1 |
| Neutral | 666 | 453,5 | 144,5 | 31,9 % | 30,8 |
| La Niña | 154 | 234,9 | 68,6 | 29,2 % | 27,4 |
| **razón máx/mín** | | | **2,2** | **1,3** | **1,2** |

**El orden se invierte.** En absoluto, El Niño parece el régimen difícil (152 frente a 69). En
relativo, es donde el modelo **mejor** funciona (25,5 % frente a 31,9 %). Y la razón entre fases cae
de 2,2 a 1,3: casi toda la diferencia aparente era efecto del nivel del precio.

> **Regla:** el MAE solo es comparable entre cosas medidas sobre el mismo conjunto. Para comparar
> entre fases, años o mercados, hace falta el relativo.

**Esta segmentación se hace aunque el ONI no sea predictora.** Son dos preguntas distintas y es
fácil confundirlas: «¿sirve el ONI para pronosticar?» —no, lo dice el capítulo
[6](#6-elegir-las-variables)— y «¿funciona el modelo igual en todos los regímenes hidrológicos?»,
que es una pregunta sobre la **generalización** del modelo y se responde partiendo el periodo de
prueba. Para lo segundo el ONI se usa como **etiqueta** del régimen, no como entrada del modelo.

## 5.6 El sesgo

El **sesgo** (*bias*) es el promedio de (pronóstico − real), **sin** valor absoluto. Mide si el
modelo se equivoca sistemáticamente hacia un lado.

El SARIMAX tiene sesgo **−22,3**: subestima en promedio. Parte de la causa es técnica: al modelar el
logaritmo y volver con `exp(·)`, lo que se obtiene es una **mediana**, no una media. No se corrigió,
para que las predicciones sean directamente comparables con las del resto de familias.

**El sesgo y el MAE pueden discrepar, y entonces el sesgo decide.** El MAE no distingue entre un
modelo que falla a los dos lados por igual y uno que falla siempre hacia abajo; el segundo es mucho
peor para decidir, porque un usuario que lo aplique se quedará corto sistemáticamente. Por eso toda
tabla de resultados del trabajo lleva el sesgo al lado, y no solo las tres métricas.

**El XGBoost en nivel lo demuestra.** Esa variante arrastra un sesgo de **−146,8** —casi siete veces
el del modelo base— y su pronóstico máximo es **557 COP/kWh** cuando el precio observado llega a
1.730. Hoy también tiene peor MAE (179,2 frente a 143,1), de modo que el MAE basta para rechazarla.

> **Pero el MAE no siempre basta para detectarlo, y esa es la lección.** Medida contra un objetivo
> **puntual** —el precio de un solo día— esta misma variante puede dar **mejor** MAE que la
> relativa (medido: 226,1 frente a 230,8) y ser igual de inservible. Mirando solo el MAE habrías
> elegido el modelo roto.
>
> El mecanismo: con pérdida absoluta el modelo busca la **mediana** condicional y, en una
> distribución con cola derecha, quedarse corto sale barato en MAE. Barato sobre un día suelto;
> menos barato sobre un promedio, porque el promedio de la ventana es una cantidad más estable
> ([§5.2](#52-las-tres-lecturas-de-un-mismo-pronóstico)) y castiga más un sesgo sistemático. De ahí
> que la dirección de esta comparación **dependa de la lectura** con la que se mire.
>
> La regla que queda: **nunca elegir por MAE solo.** Lo que delata el techo en los dos casos es el
> **sesgo** y el **máximo del pronóstico**. Y el notebook 6 comprueba con un `assert` las premisas
> del párrafo que explica la comparación, para que no pueda quedarse describiendo una dirección que
> los datos han dejado de sostener.

---

# 6. Elegir las variables

## 6.1 Por qué no basta con ordenar por correlación

La **correlación** mide si dos series se mueven juntas. Lo intuitivo sería quedarse con las que más
correlacionan con el precio. No funciona, por tres razones que el EDA documentó:

1. **Tendencia compartida.** Dos series que crecen durante dieciséis años correlacionan alto aunque
   no tengan nada que ver. El bloque macroeconómico (IPC, IPP) es el caso típico.
2. **Redundancia.** La generación total es la **suma exacta** de sus componentes. Incluir ambas no
   agrega información y rompe los modelos lineales.
3. **Persistencia.** El precio es tan predecible a partir de sí mismo que la pregunta correcta no es
   «¿esta variable se parece al precio?» sino «**¿esta variable aporta algo sobre el precio que ya
   conozco?**».

## 6.2 VIF: detectar redundancia que la correlación no ve

El **VIF** (Factor de Inflación de Varianza) pregunta, para cada variable: *¿cuán bien puedo
reconstruirla a partir de todas las demás?* Un VIF alto significa que la variable es casi una
combinación de otras.

**Por qué hace falta además de la correlación.** La correlación mira **pares**. El VIF mira el
**conjunto**. La demanda no correlaciona especialmente alto con ninguna generación por separado
(la mayor es 0,73), pero es casi la *suma* de todas — y eso solo lo ve el VIF, que le da 108.

Casos extremos del trabajo: `generacion` y `disponibilidad_declarada` tienen VIF **infinito**,
porque son combinaciones lineales **exactas** de sus componentes. Tras podar, el VIF máximo bajó de
**218 a 5,70**.

## 6.3 Importancia por permutación

**La idea, en una frase:** desordena al azar una variable y mira cuánto empeora el modelo. Lo que
empeora es lo que esa variable aportaba.

**El procedimiento exacto:**

1. Se entrena un modelo y se mide su error en datos que no usó para ajustarse.
2. Se barajan al azar los valores de una variable, **sin reentrenar**.
3. Se vuelve a medir. La diferencia es la importancia.

**Cuatro decisiones del montaje, cada una con su razón:**

- **No se mide sobre la prueba.** Eso convertiría la selección en parte del ajuste y el error
  reportado después sería optimista. Se mide sobre los cinco pliegues de validación.
- **Se permuta por bloques.** Los rezagos y las medias móviles de una misma variable son casi
  idénticos entre sí; si se permutaran uno a uno, cada uno parecería irrelevante porque sus gemelos
  cubrirían el hueco. Cada variable de origen se desordena **con todas sus derivadas a la vez**.
- **Se repite con cinco semillas y diez permutaciones.** El resultado depende del azar, y hay que
  saber cuánto.
- **Se añaden cinco variables de ruido puro** como control. Su mayor importancia media —**0,000003
  COP/kWh**, es decir cero a efectos prácticos— fija el piso por debajo del cual un aporte no se
  distingue del azar.

**La regla de entrada**, tal como está en el código, tiene tres condiciones y hay que superar
**las tres**:

```python
MIN_PLIEGUES_POSITIVOS = 4      # de 5
MIN_FRACCION_SEMILLAS  = 0.8    # al menos 4 de 5 semillas

supera_ruido = media > piso_ruido                       # 1. aporta algo
consistente  = pliegues_positivos >= 4                  # 2. en casi todos los periodos
estable      = fraccion_semillas  >= 0.8                # 3. casi independientemente del azar
```

La tercera es la más exigente y conviene entender en qué se diferencia de las otras: **no** pregunta
si la importancia media es alta, sino **cuántas semillas, por separado, habrían seleccionado esa
variable**. Una variable con buena media puede deber esa media a una o dos semillas afortunadas.

**El resultado — cinco variables de origen, 38 columnas derivadas:**

```
                    media   pliegues  semillas
precio_bolsa       3,3250     5/5       5/5     ✓
aportes_energia    0,3766     5/5       5/5     ✓
vertimientos       0,1258     4/5       4/5     ✓  (límite en las dos)
gen_hidraulica     0,0860     5/5       4/5     ✓  (límite en semillas)
dia_semana         0,0186     5/5       5/5     ✓
───────────────────────────────────────────────────
indice_ONI_publ.   0,1396     4/5       3/5     ✗  estabilidad
precio_oferta      0,0947     3/5       1/5     ✗  consistencia y estabilidad
festivos_ventana   0,0604     4/5       2/5     ✗  estabilidad
gen_termica       −0,0341     4/5       0/5     ✗  no supera al ruido
vol_util          −0,1102     4/5       0/5     ✗  no supera al ruido
```

**Cuatro observaciones que conviene tener en la cabeza:**

- **El precio domina abrumadoramente** (3,3 frente a 0,38 del segundo). Es lo esperable con un AR
  casi integrado: casi toda la información está en el precio y su dinámica reciente.
- **El ONI queda fuera por un pelo, y por la condición de estabilidad.** Su media (0,1396) supera la
  de `vertimientos` (0,1258), que sí entra; lo que lo descarta es que solo 3 de las 5 semillas lo
  habrían elegido. Dicho de otro modo: cuando aporta, aporta más que `vertimientos`, pero no se
  puede confiar en que aporte. Esto **no** contradice que El Niño mueva el precio —lo mueve, y el
  EDA lo documenta—: dice que el ONI **ya publicado** no añade nada sobre lo que el propio precio y
  los aportes de los últimos 30 días ya dejaron ver. La hidrología llega al modelo por la vía
  directa, no por el índice oceánico.
- **`vol_util` tiene importancia negativa.** Desordenarla *mejora* el error en promedio, y su
  desviación entre pliegues es **0,51** — casi cinco veces su propia media en valor absoluto. No
  dice que el nivel de los embalses sea irrelevante: dice que el modelo no le encuentra una relación
  que se sostenga entre periodos. El mecanismo está en el **cambio de signo**: un embalse lleno
  abarata hoy pero, si viene de vertimientos, anticipa un verano seco.
- **`gen_termica` y `precio_oferta` se quedan fuera** pese a ser las más correlacionadas con el
  precio en el EDA. Están ligadas *mecánicamente* a él —el precio se fija con la oferta de la planta
  marginal, que suele ser térmica—, así que sirven para **entender** el precio de hoy, no para
  anticipar su cambio.

> **La selección cambió al cambiar el objetivo**, y no por poco: con el objetivo anterior —el precio
> del día 30— entraban `indice_ONI_publicado`, `tipo_dia` y `mes`, y quedaban fuera `gen_hidraulica`
> y `dia_semana`. Ahora es al contrario.
>
> No es una inconsistencia: la importancia es una propiedad del **trío** (variable, modelo,
> objetivo), y al cambiar el tercero cambia el resultado. El promedio de treinta días suaviza el
> ruido diario, de modo que las variables que ayudaban a acertar un día suelto —el calendario, que
> dice si ese día concreto era festivo— pierden su papel, y las que describen el **estado
> hidrológico del mes** lo ganan. Que `mes` salga y `dia_semana` entre parece raro visto así, y es la
> misma lógica: lo que el promedio de la ventana todavía necesita del calendario es la composición
> de días de semana, no en qué mes del año cae.

## 6.4 El vehículo de medición

La importancia por permutación **no es una propiedad de la variable en abstracto**, sino de la
pareja formada por la variable y el modelo con el que se mide. Por eso hay que declarar cuál es ese
modelo y por qué se eligió.

Se usó potenciación del gradiente sobre árboles, con pérdida absoluta, profundidad máxima 2 y cien
árboles. Y la configuración **no se fijó por convención**: se eligió con una rejilla sobre los
mismos cinco pliegues, bajo una **condición de utilidad** —el vehículo debe superar a la
persistencia en los cinco pliegues, no solo en promedio—. Porque un modelo que no mejora a la regla
trivial no ha aprendido ningún mecanismo, y sus importancias no describirían nada.

```
                                           P1    P2    P3    P4    P5   media
absoluta, profundidad 2, 100 árboles     0,959 0,931 0,942 0,903 0,908  0,929  ← elegida
absoluta, profundidad 3, 100 árboles     0,951 0,947 0,942 0,917 0,911  0,933
absoluta, profundidad 4, 300 árboles     0,971 1,105 0,979 1,012 0,926  0,999  ✗ falla P2 y P4
cuadrática, profundidad 6, 400 árboles   0,976 1,124 1,052 1,164 0,966  1,057  ✗ falla tres
```

La configuración adoptada cumple con razón máxima **0,959**. **La configuración más extendida en la
literatura —pérdida cuadrática, profundidad 6, cuatrocientos árboles— no la cumple**: es peor que la
persistencia en tres de los cinco pliegues y su media (1,057) la deja por debajo de no hacer nada.

> **Y el orden importa:** primero se filtra por la condición de utilidad, después se elige la mejor
> media entre las que la cumplen. Hacerlo al revés —elegir la mejor media y luego comprobar— deja
> pasar configuraciones que ganan de promedio porque brillan en un pliegue y fallan en otro. Es un
> error fácil de cometer escribiendo el código en el orden natural, y pasó en el notebook 6.

> **Limitación que hay que declarar:** el conjunto se selecciona con un modelo de árboles, de modo
> que podría favorecer a esa familia en la comparación posterior. Está dicho en el documento.
>
> **Lo que mitiga esa limitación en este caso concreto.** De las seis etapas que descartan
> variables, **cuatro no usan ningún modelo**:
>
> | Etapa | Qué descarta | Cuántas | ¿Usa modelo? |
> |---|---|---:|---|
> | Identidades contables | los agregados que son suma exacta de sus componentes: `generacion` y `disponibilidad_declarada` | 2 | no |
> | Tendencia compartida | los cuatro índices de precios **en nivel**, cuya correlación se desvanece al pasar a cambios a 30 días | 4 | no |
> | Información futura | el ONI **centrado**, que promedia tres meses alrededor de cada fecha y por tanto usa datos posteriores a ella | 1 | no |
> | VIF | las reconstruibles a partir de las demás: la demanda y un subíndice del IPP (el VIF máximo baja de 218 a 5,70) | 2 | no |
> | Permutación + regla | las que no superan al ruido, o no son consistentes entre pliegues, o no son estables entre semillas | 16 | **sí** |
> | Ablación de dominio | las dudosas cuya retirada no empeora el pronóstico | — | **sí** |
>
> Las dos primeras etapas y la cuarta se aplican sobre las variables de origen antes de derivar
> nada; la tercera es la trampa del [§2.4](#24-el-oni-y-una-trampa-que-casi-arruina-el-trabajo).
> La comprobación de calidad de datos del notebook 4 **no descarta**: marca `disp_dec_hidraulica`
> con una advertencia y la deja seguir, y acaba cayendo en la etapa de permutación.
>
> Y el resultado final deja fuera precisamente a las variables que un modelo de árboles tendería a
> premiar por su relación mecánica con el precio (`gen_termica`, `precio_oferta`), que es lo
> contrario de lo que el sesgo temido predeciría. El notebook 5 lo confirma desde la otra familia:
> añadirle esas dos al SARIMAX **empeora** su MAE, de 136,70 a 139,82.

## 6.5 SHAP

**SHAP** (SHapley Additive exPlanations) responde una pregunta distinta de la permutación:

| | Qué mide |
|---|---|
| **Permutación** | cuánto **empeora el error** al perder la variable |
| **SHAP** | cuánto **mueve la predicción** la variable |

SHAP reparte **cada pronóstico individual** entre las variables que lo produjeron, usando un
resultado de teoría de juegos (el valor de Shapley) que garantiza un reparto justo y aditivo. La
media de sus valores absolutos mide la influencia de cada variable.

Se calcula sobre las **5 variables seleccionadas** (38 columnas) y 3.652 predicciones fuera de
muestra. La desviación de la aditividad es 7,45 × 10⁻⁸, que es la comprobación de que el reparto está
bien hecho:

```
precio_bolsa       0,0657
aportes_energia    0,0233
gen_hidraulica     0,0157
vertimientos       0,0076
dia_semana         0,0024
```

Están en la escala del **cambio logarítmico a 30 días**: 0,10 ≈ empujar la predicción un 10 %.

**Las dos medidas coinciden en el orden**, y eso es tranquilizador: permutación y SHAP responden
preguntas distintas y aquí dan el mismo ranking, de modo que ninguna conclusión depende de haber
elegido una de las dos.

**Lo instructivo está en cómo cambia el reparto por régimen hidrológico:**

| | El Niño (n=1.096) | Neutral (n=1.711) | La Niña (n=845) |
|---|---:|---:|---:|
| `precio_bolsa` | **0,0819** | 0,0627 | 0,0506 |
| `aportes_energia` | 0,0171 | 0,0237 | **0,0306** |
| `gen_hidraulica` | 0,0107 | 0,0106 | **0,0324** |

**En El Niño el modelo se apoya en el precio; en La Niña, en la hidrología.** Tiene sentido: durante
el estiaje el precio alto ya resume la escasez y su propia inercia manda, mientras que en época de
abundancia lo que decide es cuánta agua está entrando. El peso de `gen_hidraulica` se **triplica**
de El Niño a La Niña.

> **El ONI no aparece en esta tabla**, y no por omisión: no es predictora, de modo que no tiene
> aporte SHAP que repartir. La fase ENSO aquí es la **etiqueta** con la que se parten las
> predicciones, no una entrada del modelo. Es la distinción de
> [§5.5](#55-absoluto-frente-a-relativo-la-trampa-de-la-segmentación), vista desde el otro lado.

### Cómo leer los signos de SHAP

Hay una trampa de interpretación que hay que tener presente: **el modelo predice el cambio respecto
al precio de hoy, no el nivel**. Un SHAP positivo significa «se espera que el precio **suba** frente
al de hoy», no «el precio es alto».

La correlación de Spearman entre el valor de cada variable y su aporte SHAP:

- `precio_bolsa` (**−0,77**): cuanto más alto está hoy, menor el cambio esperado → **reversión a la
  media**. Esperable, y es el mecanismo dominante.
- `aportes_energia` (**−0,37**): más agua entrando, menor el cambio esperado. Signo físicamente
  correcto y la razón por la que esta variable entra en la selección.
- `gen_hidraulica` (**+0,47**) y `vertimientos` (**+0,44**): **cuidado**. No significa que generar con
  agua o verterla suban el precio. Con mucha generación hidráulica o muchos vertimientos el precio de
  hoy suele ser bajo, y el cambio esperado *desde un precio bajo* es al alza. Es un efecto sobre el
  **cambio**, no sobre el **nivel**, y no debe citarse como relación causal.
- `dia_semana` (**−0,65**): el número de día crece hacia el fin de semana, cuando el precio baja. Es
  el mismo perfil que recogen las *dummies* del SARIMAX ([§7.4](#74-qué-aprendió-el-modelo)), lo que
  da cierta confianza en que las dos familias están viendo lo mismo.

---

# 7. El SARIMAX

## 7.1 Qué significa cada letra

**SARIMAX** = **S**easonal **A**uto**R**egressive **I**ntegrated **M**oving **A**verage with
e**X**ogenous regressors. Se desarma así:

| Letra | Qué es | En palabras simples |
|---|---|---|
| **AR** | AutoRegresivo | «el precio de hoy depende del de días anteriores» |
| **I** | Integrado | «se trabaja con los cambios, no con el nivel» |
| **MA** | Media Móvil | «el precio de hoy depende de los errores recientes» |
| **S** | Estacional | «además hay un patrón que se repite cada semana» |
| **X** | Exógenas | «y además entran variables externas» |

**La parte MA merece una explicación**, porque es la menos intuitiva. No es un promedio móvil de los
precios. Es una dependencia de los **errores** pasados: si ayer el modelo se equivocó mucho hacia
arriba, eso contiene información sobre hoy. Es la forma de capturar que los choques se absorben de
forma gradual y no instantánea.

## 7.2 La notación (p,d,q)(P,D,Q,s)

La especificación ganadora es `(2,1,1)(1,0,0,7)`:

```
   (2, 1, 1)                      (1, 0, 0, 7)
    │  │  │                        │  │  │  └─ s = 7:  el ciclo dura 7 días (semanal)
    │  │  └─ q = 1: un término MA  │  │  └──── Q = 0:  sin MA estacional
    │  └──── d = 1: diferenciar    │  └─────── D = 0:  sin diferencia estacional
    └─────── p = 2: dos rezagos AR └────────── P = 1:  un AR estacional (el precio de hace 7 días)
```

**De dónde sale cada número:**

- **`d = 1`** ← del KPSS (capítulo 3.2) y confirmado por la validación (0,922 frente a 0,947).
- **`p = 2`** ← de la PACF, que muestra un pico dominante y una contribución menor en el segundo.
- **`q = 1`** ← del primer rezago negativo de la serie diferenciada (−0,16), que indica reversión
  parcial tras los saltos y sugiere un MA de orden bajo.
- **`s = 7` con `P = 1`** ← de los picos en los rezagos 7, 14, 21, 28 y 35 de la serie diferenciada.
- **Sin componente anual** ← porque no hay pico en el rezago 365, y la descomposición robusta
  mostró que el ciclo anual tiene amplitud modesta.

> Nada de esto se fijó por intuición: cada número viene de una medición del EDA y se verificó
> después en la rejilla de validación.

## 7.3 El código, línea por línea

```python
def matriz_exog(d, columnas):
    """Regresores de una especificación. Con d = 0 se agrega la constante explícita 'const'."""
    M = Z[columnas].copy() if columnas else pd.DataFrame(index=Z.index)
    if d == 0:
        M["const"] = 1.0
    return None if M.shape[1] == 0 else M
```

**Qué hace y por qué.** Arma la tabla de variables externas. El detalle interesante es el `if d == 0`:
cuando **no** se diferencia, el modelo necesita una constante que fije el nivel medio; cuando se
diferencia (`d = 1`), esa constante se cancela sola al tomar diferencias y añadirla sería un error.

```python
def ajustar(serie, M, orden, estacional, inicio=None, invertible=False):
    modelo = SARIMAX(serie, exog=None if M is None else M.loc[serie.index],
                     order=orden, seasonal_order=estacional, trend="n",
                     enforce_stationarity=False, enforce_invertibility=invertible)
    return modelo.fit(start_params=inicio, disp=False, maxiter=300)
```

**Argumento por argumento:**

| Argumento | Qué hace | Por qué así |
|---|---|---|
| `serie` | la variable a modelar | aquí se le pasa `log(precio)`, por el capítulo 3.6 |
| `exog` | las variables externas | `.loc[serie.index]` garantiza que estén alineadas por fecha con la serie; desalinearlas es una fuente clásica de errores silenciosos |
| `order` | `(p,d,q)` | la parte no estacional |
| `seasonal_order` | `(P,D,Q,s)` | la parte semanal |
| `trend="n"` | sin tendencia determinística | **ninguna**: la tendencia ya la absorbe `d = 1`. Añadirla además sería modelar lo mismo dos veces |
| `enforce_stationarity=False` | no obliga a las raíces AR a ser estacionarias | permite al optimizador explorar libremente; forzarlo puede hacerlo converger a un óptimo peor |
| `enforce_invertibility` | ídem para las raíces MA | por defecto `False`; la sección 7.2 del notebook examina qué pasa al activarlo |
| `disp=False` | silencia el optimizador | evita cientos de líneas de salida |
| `maxiter=300` | tope de iteraciones | sin él, un ajuste difícil puede no terminar nunca |

**Qué significa «ajustar».** Estimar los parámetros por **máxima verosimilitud**: buscar los valores
que hacen más probables los datos observados. Es un problema de optimización numérica, y por eso
puede «no converger» — que es justo lo que `maxiter` acota.

## 7.4 Qué aprendió el modelo

```
                          coef   error_est   p_valor   efecto_%
aportes_energia_dif30   0,0016      0,0024    0,5048       0,16
vertimientos_ma30       0,0180      0,0368    0,6251       1,82
gen_hidraulica_ma30    -0,0222      0,0631    0,7246      -2,20
dow1 (lunes)            0,0275      0,0050    0,0000       2,79
dow2                    0,0338      0,0068    0,0000       3,43
dow3                    0,0286      0,0072    0,0001       2,91
dow4                    0,0087      0,0065    0,1847       0,87
dow5 (sábado)          -0,0296      0,0057    0,0000      -2,91
dow6 (domingo)         -0,0908      0,0046    0,0000      -8,68
ar.L1                   0,8443      0,0140    0,0000
ar.L2                   0,0775      0,0119    0,0000
ma.L1                  -1,0317      0,0107    0,0000
ar.S.L7                 0,0903      0,0131    0,0000
sigma2                  0,0128      0,0003    0,0000

Convergió: True   |   AIC: −6.587,6   |   log-verosimilitud: 3.307,8
```

**Cómo se leen los coeficientes.** Están en la escala del **logaritmo** del precio. Para una variable
indicadora como `dow6`, el efecto porcentual es `(e^β − 1) × 100`: un coeficiente de −0,0908
significa que el domingo el precio es un **8,68 %** más bajo que el día de referencia.

**Qué dice cada bloque:**

- **Ninguna de las tres variables del sistema es significativa** (p = 0,50, 0,63 y 0,72). El modelo
  lineal no les encuentra efecto una vez descontada la dinámica propia del precio. Es coherente con
  lo que mide la comparación de configuraciones: quitar las exógenas empeora el MAE de 136,70 a
  137,45, menos de un COP/kWh.
- **El día de la semana sí lo es, y fuerte.** El perfil tiene sentido físico: el precio sube de lunes
  a miércoles, se aplana el jueves y cae el fin de semana, con el domingo claramente más barato. Es
  el reflejo de la demanda industrial.
- **`ar.L1` + `ar.L2` = 0,922**: alta persistencia de lo que queda tras la diferencia.
- **`ar.S.L7` = 0,090**, mucho menor que en versiones anteriores del modelo (0,228). La razón es
  interesante y no es un deterioro: antes el eco semanal **solo** podía entrar por ese término,
  porque el modelo no tenía indicadores de día de la semana. Ahora `dia_semana` pasó la selección,
  las *dummies* están en las exógenas y capturan el patrón de forma explícita, de modo que al AR
  estacional le queda poco por explicar. Dos vías para el mismo fenómeno, y la explícita es
  preferible porque es interpretable.
- **`sigma2` = 0,0128** → desviación de las innovaciones de **~11,3 % diario** (√0,0128 = 0,113): la
  incertidumbre que el modelo ve en un día cualquiera.

**Qué es un «p-valor» aquí.** La probabilidad de observar un coeficiente así de grande si la
variable no tuviera ningún efecto. Por debajo de 0,05 se considera que el efecto es real.

> **Un p-valor alto no es prueba de que la variable no sirva.** Dice que *este* modelo, *en esta
> escritura*, no le encuentra un efecto lineal distinguible del ruido. Los aportes importan en el
> mercado eléctrico colombiano; lo que estos coeficientes dicen es que, sabiendo el precio de hoy y
> su dinámica reciente, no añaden información lineal nueva sobre el mes que viene.

## 7.5 Dos diagnósticos que hay que saber leer

**Las raíces.** Un modelo AR es *estacionario* si las raíces de su polinomio tienen módulo mayor que
1; un MA es *invertible* bajo la misma condición. Aquí: AR **1,078** (admisible, con margen
estrecho) y MA **0,969** — **por debajo de 1**, es decir, el modelo estimado **no es invertible**.

Esto no es un fallo del ajuste: es la señal clásica de **sobrediferenciación**. Con `d = 1` y un MA
cercano a −1, los dos efectos casi se cancelan. La interpretación útil es que la parte integrada se
comporta como un **promedio exponencial de los últimos ~30 días**: el modelo no parte de «el precio
de ayer» sino de una media que se adapta despacio.

Un MA con raíz `r` y otro con la raíz reflejada `1/r` producen **las mismas autocovarianzas**: son
dos escrituras del mismo modelo. Por eso la variante con invertibilidad impuesta da MAE 136,53
frente a 136,70 — una diferencia sin importancia, y la prueba empírica de que las dos escrituras son
el mismo modelo.

> **Y sin embargo `d = 1` se conserva.** La alternativa `d = 0` quedó última en las tres lecturas
> (145,28 / 176,24 / 226,02) y con un sesgo de −83,7, casi cuatro veces el del modelo adoptado. La
> sobrediferenciación aparente cuesta menos que no diferenciar.

**El Ljung-Box.** Contrasta si los residuos conservan autocorrelación (un modelo bien especificado
no debería dejar estructura sin capturar).

```
     lb_stat   lb_pvalue
 7    3,1459     0,8712     no se rechaza
14   36,0124     0,0010     se rechaza
30   68,4309     0,0001     se rechaza
```

**A 7 rezagos no se rechaza con holgura** (p = 0,87): la estructura semanal queda bien capturada —lo
que es de esperar, con las *dummies* de día de la semana dentro—. A 14 y 30 **sí** se rechaza: queda
dinámica de plazo más largo sin modelar. Esto es la limitación honesta del modelo, y conviene no
disimularla: el horizonte del trabajo es 30 días, y es precisamente a 30 rezagos donde el residuo
conserva más estructura.

**Qué le pasa a los intervalos de predicción.** Cubren el **89 %** y el **98 %** de los valores
reales cuando su nivel nominal es 80 % y 95 %. Es decir, son **demasiado anchos**: el modelo declara
más incertidumbre de la que tiene. No es el error más peligroso —un intervalo estrecho de más daría
una falsa seguridad—, pero sigue siendo un intervalo mal calibrado, y por eso el documento declara
que **no deben reportarse como intervalos calibrados**.

> La causa es la misma estructura residual del Ljung-Box, con un efecto añadido del cambio de
> objetivo: el intervalo lo calcula `statsmodels` para cada día por separado y aquí se aplica al
> **promedio de la ventana**, que es una cantidad menos variable porque los errores diarios se
> compensan ([§5.2](#52-las-tres-lecturas-de-un-mismo-pronóstico)). Un intervalo calculado para la
> variabilidad diaria queda necesariamente ancho cuando se usa sobre un promedio.

---

# 8. El XGBoost

## 8.1 Qué es un árbol, y qué es potenciar

Un **árbol de decisión** hace preguntas encadenadas sobre las variables: «¿los aportes de los
últimos 30 días cayeron más de X? sí → ¿los vertimientos están por debajo de Y? sí → predigo un
cambio de +8 %». Cada hoja del árbol contiene un número: el **promedio** de los casos de
entrenamiento que llegaron ahí.

**Gradient boosting** (potenciación del gradiente) construye muchos árboles **en secuencia**: cada
uno corrige los errores que dejaron los anteriores. El primero da una aproximación burda; el
segundo aprende sobre lo que falló el primero; y así. XGBoost es una implementación eficiente de
esa idea.

**La diferencia fundamental con el SARIMAX:** el SARIMAX es un modelo *de serie de tiempo* — sabe
que existe un orden. El árbol **no sabe nada del tiempo**: ve cada fila como un caso independiente.
Por eso el tiempo tiene que entrar a mano, en forma de rezagos, medias móviles y diferencias.

## 8.2 La limitación que lo define: no extrapola

**Un árbol predice promedios de hojas.** Por tanto **no puede predecir un valor fuera del rango que
vio al entrenarse**. Si el precio más alto del entrenamiento fue 1.200, el árbol jamás predecirá
1.300.

En este trabajo eso no es un detalle teórico, es el problema central:

El objetivo del modelo es el **promedio de la ventana**, de modo que es su rango el que importa:

```
promedio de los 30 días   entrenamiento   prueba
mínimo                             46,1    111,4
media                             192,0    466,6     (×2,43)
percentil 99                      833,0  1.624,8
máximo                          1.147,9  1.730,0

orígenes de prueba por encima del MÁXIMO del entrenamiento: 46 de 1.155
```

> **Dos máximos que no hay que confundir.** 1.730 es el promedio de ventana más alto; el **precio de
> un solo día** llegó a **2.498,8** el 30 de septiembre de 2024. El primero es la vara para un
> modelo que pronostica el promedio; el segundo, para uno que pronostica un día.

**Qué pasa si se ignora.** Ajustado directamente sobre el precio, el XGBoost se queda con un techo:

```
                   objetivo relativo   objetivo en nivel   persistencia   observado
máximo                       2.257,5               556,7        2.498,8     1.730,0
sesgo medio                    −22,1              −146,8            4,6         0,0
% días por debajo               59,0                73,2           55,5         —
```

**El modelo en nivel no pasa de 557 COP/kWh** y se queda, en media, 147 por debajo de lo observado.
No es que le cueste acertar: **no puede representar** el nivel al que llegó el precio. El relativo
alcanza 2.258 porque el nivel entra multiplicando — y la persistencia, que simplemente repite el
precio de hoy, alcanza 2.499 sin esfuerzo. Esa es la comparación que deja clara la limitación: la
regla trivial puede expresar valores que el modelo entrenado no.

**La solución: modelar el cambio, no el nivel.**

```python
if relativo:
    z = np.log(ajuste["objetivo"].values / ajuste["precio_hoy"].values)
    modelo.fit(ajuste[columnas], z)
    return evalua["precio_hoy"].values * np.exp(modelo.predict(evalua[columnas]))
```

**Línea por línea:**

1. `z = log(objetivo / precio_hoy)` — el objetivo deja de ser el precio y pasa a ser el **cambio
   logarítmico** respecto del día de origen. Ese cambio sí vive en un rango parecido entre
   entrenamiento y prueba.
2. `modelo.fit(X, z)` — el árbol aprende a predecir el cambio.
3. `precio_hoy × exp(predicción)` — se reconstruye el precio multiplicando el precio del origen por
   el cambio predicho.

Así el pronóstico **no tiene techo**: si el precio de hoy es 2.000 y el modelo predice un cambio de
+10 %, el resultado es 2.200 aunque jamás haya visto ese valor.

> **Regla para la familia de árboles:** el objetivo relativo es **obligatorio** en esta serie. Y no
> basta mirar el MAE — hay que mirar el **sesgo** y el **máximo del pronóstico**, porque un modelo
> con techo puede tener buen MAE promedio mientras falla donde importa.

## 8.3 Los hiperparámetros, uno por uno

La configuración elegida por la rejilla:

```python
dict(objective="reg:absoluteerror",
     max_depth=1, n_estimators=200, learning_rate=0.02, min_child_weight=100,
     subsample=0.7, colsample_bytree=0.5, random_state=semilla, n_jobs=-1)
```

Un **hiperparámetro** es una decisión que se toma *antes* de entrenar y que el modelo no aprende de
los datos. Aquí ninguno se fijó por convención: todos salieron de una rejilla evaluada sobre los
cinco pliegues, con la misma condición de utilidad de
[§6.4](#64-el-vehículo-de-medición) —superar a la persistencia en los **cinco**—.

| Parámetro | Qué controla | Por qué este valor |
|---|---|---|
| `objective="reg:absoluteerror"` | qué error se minimiza | **Pérdida absoluta, no cuadrática.** Con cola pesada (RMSE/MAE = 1,51), una pérdida cuadrática concentraría el ajuste en unas decenas de episodios extremos. Además alinea el entrenamiento con la métrica principal, que es el MAE |
| `max_depth=1` | profundidad de cada árbol | **Un solo corte por árbol** (un *stump*). Es el extremo de lo poco flexible, y lo eligió la rejilla: con 38 predictoras y una señal muy débil, cualquier capacidad adicional memoriza ruido |
| `n_estimators=200` | cuántos árboles | más que antes, pero cada uno aporta mucho menos: 200 árboles de un corte tienen menos capacidad que 100 de dos |
| `learning_rate=0.02` | cuánto corrige cada árbol | muy bajo: el conjunto aprende despacio y con menos riesgo de sobreajustar |
| `subsample=0.7` | fracción de **filas** por árbol | cada árbol ve el 70 % de los datos. Introduce variedad entre árboles y actúa como regularización |
| `colsample_bytree=0.5` | fracción de **columnas** por árbol | cada árbol ve la mitad de las variables. Impide que todos se apoyen en las mismas |
| `min_child_weight=100` | mínimo de observaciones por hoja | **muy alto a propósito**: ninguna hoja puede basarse en menos de 100 casos, lo que impide reglas construidas sobre un puñado de días raros |
| `random_state=semilla` | la semilla del azar | `subsample` y `colsample` usan azar; fijar la semilla hace el resultado reproducible |
| `n_jobs=-1` | núcleos del procesador | usa todos los disponibles |

**El hilo que une a casi todos: regularización.** Profundidad mínima, tasa de aprendizaje baja,
submuestreo de filas y columnas, hojas muy grandes. Todo apunta en la misma dirección porque la
señal disponible por encima de la persistencia es pequeña, y lo fácil aquí es construir un modelo
que memorice el pasado y no sirva para el futuro.

> **Y la rejilla lo confirma por contraste.** De 13 configuraciones, 10 superan a la persistencia en
> los cinco pliegues. La de **mejor razón media** (profundidad 4, 300 árboles: 0,936) **no se elige**
> porque falla en un pliegue (1,025); la elegida queda en 0,937 de media con 0,970 en el peor. Y la
> configuración habitual de la literatura —profundidad 6, 400 árboles, pérdida cuadrática— queda
> **última** y por debajo de la persistencia.
>
> **El objetivo de ventana es lo que empuja hacia tan poca flexibilidad.** Promediar treinta días
> suaviza la serie y debilita lo que queda por explicar por encima de la persistencia; con menos
> señal, menos capacidad.

## 8.4 Por qué cinco semillas

`subsample` y `colsample_bytree` eligen filas y columnas **al azar**. Dos corridas con semillas
distintas dan modelos distintos. Reportar una sola semilla sería reportar un golpe de suerte.

El trabajo corre **cinco** y reporta el promedio. La dispersión resultó de **0,12 COP/kWh** de MAE
(143,11 de media sobre las cinco), de modo que la cifra es estable — pero eso **se comprueba, no se
asume**. La variante en nivel, por contraste, dispersa 0,73.

## 8.5 El resultado y cómo se lee

Sobre la lectura `promedio_30d`, con la persistencia en **145,66**:

| Variante | MAE | sesgo | Lectura |
|---|---:|---:|---|
| Objetivo relativo, reajuste 60 d (base) | 143,10 | −22,1 | el del protocolo |
| Objetivo relativo, un solo ajuste | 141,03 | −30,8 | reajustar **no** ayuda |
| Objetivo relativo, solo el precio | 142,94 | −38,6 | las exógenas **no aportan** |
| Objetivo en nivel | 179,17 | −146,8 | el techo, en cifras |

**Tres resultados negativos, y los tres son resultados.** Reajustar cada 60 días empeora el MAE en
2,07 frente a un solo ajuste; quitar todas las exógenas y dejar solo el precio lo mejora en 0,16,
que está dentro del ruido entre semillas. El modelo base conserva el reajuste **porque es el
protocolo** —así se usaría de verdad, reentrenando cuando llegan datos— y conserva las exógenas
porque retirarlas no mejora nada demostrable. Cambiar el protocolo porque da mejor número en esta
prueba sería ajustar el método al resultado.

**Las tres lecturas de la trayectoria** (un modelo por horizonte, promediado sobre cinco semillas):

| | `pares_30d` | `promedio_30d` | `dia_30` |
|---|---:|---:|---:|
| Persistencia | 178,37 | 145,66 | 239,27 |
| XGBoost | **175,59** | **143,17** | **232,74** |
| razón | 0,98 | 0,98 | 0,97 |

> **Una comprobación que conviene notar.** El 143,17 de la trayectoria y el 143,10 del modelo
> entrenado **directamente** sobre el promedio difieren en 0,07 COP/kWh. Son dos estimadores
> distintos —treinta modelos promediados frente a uno solo— y coinciden. Si no coincidieran, uno de
> los dos estaría mal montado.

**El error por horizonte dice algo incómodo:**

```
 horizonte    modelo   persistencia   razón
         1    52,08        53,08       0,981
         2    73,38        76,11       0,964
         3    87,89        90,96       0,966
         7   131,69       133,39       0,987
        14   181,05       180,93       1,001
        21   209,10       211,56       0,988
        30   232,75       239,27       0,973
```

La ventaja **no crece** con el horizonte. Debería: la persistencia pierde valor a medida que t+h se
aleja, de modo que un modelo que explique algo real debería distanciarse. Lo que se ve es que gana a
dos y tres días, **empata hacia la mitad** del horizonte (1,001 en h = 14) y recupera algo al final.
Dicho sin adornos: **el modelo gana donde la persistencia ya es buena.**

**Y el desglose por régimen lo confirma:**

| Fase | n | precio medio | Persistencia | XGBoost | ventaja |
|---|---:|---:|---:|---:|---:|
| El Niño | 335 | 599,3 | 163,3 | 161,5 | 1,1 % |
| Neutral | 666 | 453,5 | 149,9 | 149,4 | 0,4 % |
| La Niña | 154 | 234,9 | 88,8 | 76,0 | **14,4 %** |

Casi toda la ventaja del XGBoost viene de **La Niña**, que son 154 de 1.155 orígenes. En los días
neutrales —más de la mitad del periodo— aporta un 0,4 %, que es indistinguible de nada.

**Frente al SARIMAX, el XGBoost pierde en todo el tablero:**

| | Persistencia | SARIMAX | XGBoost |
|---|---:|---:|---:|
| MAE global | 145,66 | **136,53** | 143,10 |
| El Niño | 163,3 | **152,3** | 161,5 |
| Neutral | 149,9 | **144,4** | 149,4 |
| La Niña | 88,8 | **68,2** | 76,0 |
| 2023 | 163,5 | **150,2** | 163,5 |
| 2024 | 213,6 | **205,8** | 209,8 |
| 2025 | 88,1 | **82,3** | 85,4 |
| 2026 | 68,3 | **55,6** | 63,4 |

**Fases en que el XGBoost supera al SARIMAX: 0 de 3. Años: 0 de 4.** Eso es lo que hace sólida la
conclusión: no gana el SARIMAX por el promedio de un periodo favorable, gana en todos los cortes.
Y las dos familias superan a la persistencia en las tres fases y en los cuatro años, de modo que
ninguna debe su ventaja a un tramo concreto.

Es un resultado legítimo y reproduce, con datos propios y condiciones idénticas, el patrón que la
literatura colombiana ya había documentado para este mercado: en esta serie, con este horizonte, la
estadística clásica le gana a los árboles. Que una familia pierda no es un fallo del trabajo — es
la respuesta a la pregunta que el trabajo hace.

---

# 9. La LSTM

Es la familia de Deep Learning del trabajo, y el notebook 7 la construye. Este capítulo explica qué
es una red neuronal, qué le añade la palabra «recurrente», qué problema resuelve la LSTM en
concreto, y por qué cada decisión del montaje se tomó como se tomó.

## 9.1 Qué es una neurona, y qué es una red

Una **neurona** hace tres cosas, en este orden:

1. Multiplica cada entrada por un número propio, su **peso**.
2. Suma los resultados y añade otro número, el **sesgo** (aquí la palabra significa otra cosa que en
   [§5.6](#56-el-sesgo); es un término de desplazamiento).
3. Pasa el total por una **función de activación**, que lo achata. Sin ese achatamiento la neurona
   sería una suma ponderada y nada más, y apilar muchas daría otra suma ponderada.

```
entradas         pesos        suma       activación     salida
  x₁ ──────────── w₁ ──┐
  x₂ ──────────── w₂ ──┼──▶ Σ + b ──▶   tanh(·)    ──▶  h
  x₃ ──────────── w₃ ──┘
```

Una **red** es un montón de neuronas organizadas en capas, donde la salida de una capa es la entrada
de la siguiente. **Entrenar** la red es buscar los pesos y los sesgos que hacen más pequeño el
error. Se hace así: se le pasa un caso, se compara su salida con la respuesta correcta, se calcula
cuánto habría que mover cada peso para que el error bajara un poco, y se mueve. Se repite con todos
los casos, muchas veces. Cada pasada completa por los datos es una **época**.

La LSTM de este trabajo tiene **5.966 parámetros** en su configuración de 32 unidades. Ninguno se
elige a mano: todos salen de ese proceso de empujar y medir.

## 9.2 Qué añade la palabra «recurrente»

Una red normal ve cada caso por separado y no sabe que existe un antes y un después. Una red
**recurrente** procesa una secuencia paso a paso y, en cada paso, se pasa a sí misma un resumen de
lo que lleva visto. Ese resumen es el **estado**.

```
precio del día 1 ──▶ [red] ──▶ estado₁ ──┐
                                          │
precio del día 2 ──▶ [red] ◀──────────────┘  ──▶ estado₂ ──┐
                                                            │
precio del día 3 ──▶ [red] ◀────────────────────────────────┘  ──▶ ...
```

La misma red, con los mismos pesos, se aplica en los 31 pasos. Lo único que cambia entre un paso y
el siguiente es el estado. Al llegar al último día, el estado contiene lo que la red decidió
conservar de toda la ventana.

**Esa es la diferencia de fondo con un árbol.** Un árbol no tiene noción de orden: para que vea
historia hay que dársela como columnas, y de ahí los rezagos y las medias móviles del capítulo
[6](#6-elegir-las-variables). Una red recurrente recorre el tiempo por construcción.

## 9.3 El problema que la LSTM resuelve

Una red recurrente simple **olvida**, y la razón es aritmética. Para ajustar los pesos hay que
calcular cuánto influyó cada paso en el error final, y eso implica multiplicar una cadena de
factores, uno por paso. Si esos factores son menores que uno, el producto se va a cero: con treinta
pasos y factores de 0,8, el primer día influye 0,8³⁰ ≈ 0,001 en la corrección. La red no aprende
nada de los días lejanos, no porque sean irrelevantes sino porque la señal para aprenderlos se
desvanece por el camino. Se llama **gradiente que se desvanece**.

La LSTM —*Long Short-Term Memory*, memoria larga de corto plazo— resuelve eso con dos añadidos: una
**celda de memoria** que atraviesa todos los pasos casi sin modificarse, y tres **compuertas** que
deciden qué entra, qué sale y qué se borra de ella.

## 9.4 Las tres compuertas

Una compuerta es una neurona cuya activación da un número entre 0 y 1, y ese número se usa para
multiplicar: 0 cierra el paso, 1 lo abre, y los valores intermedios dejan pasar una parte. La imagen
útil es una cinta transportadora con tres grifos.

| Compuerta | Qué decide | Si vale 0 | Si vale 1 |
|---|---|---|---|
| **De olvido** | qué parte de la memoria se descarta | borra la memoria anterior | la conserva íntegra |
| **De entrada** | qué parte de la observación nueva se guarda | ignora el día actual | lo incorpora entero |
| **De salida** | qué parte de la memoria se expone al resto de la red | no muestra nada | muestra todo |

```
   memoria anterior ──▶⊗───────────────▶⊕──────────────▶ memoria nueva
                       ▲                ▲                      │
                  [olvido]         [entrada]⊗                  ▼
                                          ▲               [salida]⊗ ──▶ estado
   día actual ────────────────────────────┴──────────────────────▲
```

Lo importante de ese dibujo es la línea de arriba: la memoria viaja de un paso al siguiente **por
una suma, no por una multiplicación encadenada**. Ahí está la solución al desvanecimiento. Y las
compuertas no están programadas: son neuronas, y sus pesos se aprenden. La red decide por sí misma
cuánta historia retener en cada momento.

## 9.5 Por qué la red lee una ventana y no las 38 columnas

Esta es la decisión de diseño más importante del notebook 7, y conviene entenderla bien.

Los notebooks 5 y 6 leen una fila por origen con 38 columnas ya derivadas: rezagos de 1, 7, 14 y 30
días, medias móviles de 7 y 30, y diferencias de 1 y 30. Si a la LSTM se le diera lo mismo, recibiría
**un solo paso temporal**. Y con un solo paso:

- la compuerta de olvido no tiene memoria anterior que olvidar,
- la celda no acumula nada, porque no hay un paso siguiente,
- y la red queda reducida a una capa densa, con el coste en parámetros de una LSTM y ninguna de sus
  propiedades.

Sería una LSTM de nombre. Hay un segundo argumento, y es el más fuerte: **las 38 columnas ya deciden
qué resúmenes de la historia importan**. Que la media relevante sea de 7 y de 30 días, y los rezagos
de 1, 7, 14 y 30, lo eligió una persona. La afirmación central de una red recurrente es que ella
*aprende* qué peso dar a cada punto de la historia. Darle los agregados le quita exactamente la
facultad que se está poniendo a prueba.

Por eso la red lee una **ventana de 31 días × 4 variables**: los últimos treinta días más el día del
origen, de `precio_bolsa`, `aportes_energia`, `vertimientos` y `gen_hidraulica`.

### Y sin embargo no son datos distintos

Esto es lo que hace legítima la comparación, y el notebook lo comprueba antes de entrenar nada. La
ventana **contiene** las 38 columnas: cada rezago, cada media móvil y cada diferencia es una función
de ella.

```
precio_bolsa_lag7   = el valor del día 24 de la ventana
precio_bolsa_ma30   = la media de los días 1 a 30
precio_bolsa_dif30  = día 31 menos día 1
```

El notebook reconstruye esas columnas desde la ventana para 300 orígenes al azar y las compara con
las del dataset. **Reconstruye 36 de las 38 con un error relativo máximo de 10⁻¹⁰**, que es
aritmética de coma flotante. Las dos que no da son el día de la semana y el indicador de
vertimientos, que entran aparte como variables **estáticas** —no son series que haya que recorrer—.
Si la comprobación fallara, el notebook se detiene.

> **El detalle que delató la alineación correcta.** Al reconstruir las medias móviles no cuadraban.
> La causa está en el notebook 4: se calculan con `shift(1)` antes del `rolling`, de modo que **la
> ventana de la media termina el día anterior** al origen, no en el origen. Es deliberado y está
> comentado en el código. Reconstruirlas sin ese desplazamiento da otro número, y por eso una
> comprobación numérica vale más que leer el nombre de la columna.

### Lo que sí cambia, y hay que declararlo

El **grano**. Las 38 columnas son un resumen con pérdida: una media de treinta valores tira los
treinta valores. La ventana los conserva. La red ve, por tanto, **más detalle**, no distinto
detalle.

Eso deja una limitación honesta: si la LSTM gana, no se podrá separar cuánto de la ganancia viene de
la arquitectura recurrente y cuánto del grano más fino de la entrada. Separarlo exigiría una variante
que leyera los agregados en un solo paso, y se decidió no correrla para no multiplicar el trabajo.
Queda declarada en las conclusiones del notebook y en el documento.

## 9.6 La arquitectura, pieza por pieza

```
ventana (31 × 4) ──▶ LSTM(32 unidades) ──▶ estado (32 números) ──┐
                                                                  ├──▶ Dense(30) ──▶ 30 salidas
estáticas (8) ───────────────────────────────────────────────────┘
```

- **`LSTM(32)`**: treinta y dos celdas en paralelo, cada una con sus tres compuertas. Cada una
  aprende a seguir un aspecto distinto de la ventana. El resultado es un estado de 32 números.
- **Las estáticas**: siete indicadoras del día de la semana del origen más el indicador de
  vertimientos. Van directas al final, sin pasar por la parte recurrente, porque no son historia.
- **`Dense(30)`**: una capa que toma esos 40 números y produce los treinta valores de la
  trayectoria.

**Por qué siete indicadoras y no el número del día.** Si el lunes fuera 0 y el domingo 6, la red
leería que el domingo está «seis veces más lejos» del lunes que el martes, y eso no significa nada.
Con siete columnas de ceros y un uno, cada día es una categoría sin orden impuesto. Se llama
**codificación one-hot**.

**Y por qué basta el día de la semana del origen**, cuando lo que importa es el de cada uno de los
treinta días pronosticados. Porque la salida tiene una neurona por horizonte: la que produce el día
7 siempre produce el día 7, de modo que puede aprender por sí sola el desplazamiento. El día de la
semana de t+h se deduce del de t y de h.

## 9.7 Por qué treinta salidas de una vez

Es la ventaja de forma de esta familia, y conviene verla al lado de lo que costó a las otras dos.

| Familia | Cómo produce los 30 valores | Coste |
|---|---|---|
| SARIMAX | una sola proyección de 30 pasos, que es su forma nativa | segundos |
| XGBoost | un modelo por horizonte: 30 modelos × 5 semillas | 1.569 s |
| LSTM | una capa densa de 30 salidas, en un solo ajuste | un ajuste |

El tiempo del XGBoost es el que imprime su celda en el equipo donde se corrió, y cambia de una
máquina a otra; lo que no cambia es el orden de magnitud frente a las otras dos filas.

Que la red entregue los treinta valores juntos tiene además una consecuencia que no es solo de
coste: los treinta comparten el mismo estado, de modo que la red puede aprender que la trayectoria
es una curva y no treinta números sueltos.

## 9.8 El escalador, donde se esconde una fuga

Una red necesita entradas en un rango parecido. Si una variable vale cientos de millones y otra
vale decenas, la primera domina los gradientes y la segunda no se aprende. La solución estándar es
**estandarizar**: a cada variable se le resta su media y se la divide por su desviación estándar, de
modo que todas queden centradas en cero con dispersión uno.

Aquí está la trampa. **La media y la desviación se calculan sobre unos datos, y esos datos no pueden
incluir el bloque de prueba.** Si se calcularan sobre todo el conjunto, el modelo habría visto —a
través del escalador— la media del periodo que todavía no ha ocurrido. Es una fuga de información
sutil y, por lo tanto, fácil de cometer: nada falla, el modelo simplemente sale mejor de lo que es.

En el notebook, el escalador se ajusta **solo con el conjunto de ajuste de cada bloque** y se
reajusta en cada bloque de la validación progresiva, igual que el modelo. Es el equivalente, para
esta familia, del filtro `fecha_objetivo < corte` que protege a las otras dos
([§4.5](#45-la-purga-que-es-distinta-del-embargo)).

Antes de estandarizar, las cuatro variables pasan a **escala logarítmica**, por la razón del
capítulo [3.6](#36-volatilidad-por-qué-el-logaritmo): la serie es multiplicativa, con una
correlación de 0,813 entre el nivel local y la dispersión local. En `vertimientos` se usa `log1p`
—es decir log(1+x)— porque tiene 2.313 días en cero y el logaritmo de cero no existe.

## 9.9 El objetivo relativo, también aquí

El capítulo [8.2](#82-la-limitación-que-lo-define-no-extrapola) explicó por qué un árbol **tiene
que** pronosticar la razón y no el precio: no puede salir del rango que vio. Una red no tiene ese
techo, porque su salida es una combinación continua de sus entradas. El argumento que hacía
obligatorio el objetivo relativo en los árboles no se aplica, entonces, a esta familia.

Y sin embargo se usa el mismo objetivo relativo, por otra razón. El nivel del objetivo se multiplica
por **2,43** entre entrenamiento y prueba, y la red trabaja sobre entradas estandarizadas con un
escalador ajustado solo en entrenamiento: en la prueba recibiría valores muy desplazados respecto de
aquellos con los que se normalizó. Que la arquitectura *pueda* extrapolar en principio no significa
que lo haga bien a partir de entradas fuera de rango. El objetivo relativo elimina el problema en el
origen y deja, además, **una sola formulación en las tres familias**, que es más fácil de defender
que tres decisiones distintas.

Así que la red aprende, para cada uno de los treinta días:

```
z(h) = log( precio del día t+h / precio del día t )
```

y el precio se reconstruye después multiplicando por el precio del origen. Es exactamente la
formulación del notebook 6, extendida de una salida a treinta.

## 9.10 Lo que esta familia no arregla

Conviene decirlo antes de ver los resultados, para no leerlos con expectativas que el montaje no
puede cumplir.

- **Sigue siendo pronóstico puntual.** La red entrega treinta números, no treinta distribuciones. La
  dispersión entre semillas mide cuánto varía el procedimiento, no la incertidumbre del pronóstico.
- **Depende del azar más que un árbol.** Los pesos se inicializan al azar y el orden de los lotes
  también, de modo que dos corridas con semillas distintas dan modelos distintos. De ahí las cinco
  semillas del protocolo, que aquí no son una formalidad.
- **Cuesta mucho más.** Sin GPU, un ajuste de 60 épocas sobre las 4.568 filas de entrenamiento tarda
  **45,6 segundos**. La validación progresiva son veinte bloques por cinco semillas, cien ajustes, y
  la rejilla otros treinta. El notebook completo ronda los noventa y cinco minutos, frente a los
  segundos del SARIMAX.
- **Y no hace falta que gane.** Si no baja de la marca del SARIMAX, eso responde la pregunta del
  trabajo —qué familia funciona mejor en este problema— y una respuesta negativa sobre una familia
  costosa también es un resultado.

## 9.11 El resultado, y las dos cosas que enseña

La LSTM supera a la persistencia en las tres lecturas, y la supera por muy poco:

| | `pares_30d` | `promedio_30d` | `dia_30` |
|---|---:|---:|---:|
| Persistencia | 178,37 | 145,66 | 239,27 |
| LSTM | 176,93 | 144,83 | 233,48 |
| razón | 0,99 | 0,99 | 0,98 |

Puesta al lado de las demás configuraciones, sobre el promedio del mes, queda **última de las
cinco**:

| Modelo | MAE | Mejora sobre la persistencia |
|---|---:|---:|
| Persistencia | 145,66 | — |
| **GRU** | **132,47** | **9,1 %** |
| SARIMAX | 136,70 | 6,2 % |
| Híbrido | 140,17 | 3,8 % |
| XGBoost | 143,17 | 1,7 % |
| LSTM | 144,83 | 0,6 % |

Una mejora del 0,57 % no se distingue de nada en una serie con esta cola. Pero el camino hasta esa
cifra enseña dos cosas que importan más que ella.

### La mejor red es la que casi no se entrena

De las dieciocho configuraciones de la rejilla, **cuatro** superan a la persistencia en los cinco
pliegues, y la elegida es una LSTM de 32 unidades entrenada **tres épocas**: tres pasadas por los
datos. La curva de aprendizaje explica por qué, y conviene verla:

```
                      3 ép.  5 ép.  10 ép.  15 ép.  30 ép.  60 ép.
LSTM 16               0,989  0,960   0,954   0,956   1,287   1,359
LSTM 32               0,955  0,956   0,993   1,157   1,389   1,550
LSTM 32 dropout 0,2   0,955  0,958   0,963   1,069   1,322   1,461
```

El error de validación baja hasta algún punto entre las tres y las quince épocas, y a partir de ahí
**sube sin parar**. A sesenta épocas la red llega a quedar un 55 % peor que repetir el precio de
hoy. No es un fallo del montaje: es memorización. Casi seis mil parámetros contra conjuntos de
ajuste que arrancan en 886 orígenes dan de sobra para aprenderse los casos uno por uno, y lo que se
aprende así no sirve para los siguientes.

> **La lección general.** En el XGBoost la regularización venía de la forma del modelo: árboles de
> uno o dos cortes, hojas de cien observaciones como mínimo, submuestreo de filas y columnas. Aquí
> ninguna de esas palancas existe, y la que queda es **parar a tiempo**. Las unidades y el *dropout*
> apenas mueven el resultado —0,955 con 32 unidades y 0,955 con 32 unidades y *dropout*— mientras
> que las épocas lo deciden todo.
>
> Esto también es una advertencia metodológica. Una rejilla que solo hubiera probado 30 y 60 épocas
> —valores perfectamente razonables a priori— habría concluido que esta familia es inservible, con
> razones de 1,26 a 1,56. La primera versión de este notebook probó exactamente eso y se detuvo en
> su propia condición de utilidad. Que un modelo falle puede significar que el modelo no sirve o que
> la rejilla se buscó en el sitio equivocado, y distinguir las dos cosas exige mirar la curva.

### Gana y pierde en los extremos, mucho más que las demás

Aquí está el hallazgo de fondo. El error por régimen climático:

| Fase | n | Persistencia | SARIMAX | XGBoost | LSTM | GRU | Híbrido |
|---|---:|---:|---:|---:|---:|---:|---:|
| El Niño | 335 | 162,6 | **152,2** | 162,3 | 185,6 | 175,2 | 157,9 |
| Neutral | 666 | 147,7 | 142,0 | 145,7 | 140,6 | **129,8** | 144,5 |
| La Niña | 154 | 88,4 | 65,2 | 76,7 | 58,4 | **32,0** | 67,6 |

La LSTM se **polariza**: durante La Niña mejora a la persistencia en un 34 % y durante El Niño
queda un 14 % *por debajo* de ella, que es el único caso de los cinco. El GRU lleva la ventaja en
La Niña mucho más lejos —32,0, menos de la mitad del siguiente— pero también cede durante El Niño,
donde el mejor es el SARIMAX.

Por año repite el patrón: la LSTM pierde contra la persistencia en 2023 (180,4 frente a 163,5) y
queda por detrás del GRU en 2025 y 2026.

**Qué significa.** El régimen húmedo es el tramo en que el precio es bajo y su dinámica más regular,
y ahí la red encuentra estructura que los otros dos no aprovechan. El régimen seco es el de los
picos, y ahí la red falla más que no hacer nada. Para el horizonte que el trabajo pregunta eso es
malo: el error del periodo se decide en El Niño, que es donde la LSTM se desploma.

### Y depende del azar quince veces más que los árboles

Las cinco semillas dan MAE de 142,07, 144,68, 145,49, 146,41 y 147,46 — una desviación de **2,05
COP/kWh**, frente a los **0,12** del XGBoost. Es la razón por la que el protocolo exige varias
semillas, y aquí no es una formalidad: reportando una sola semilla se podría haber escrito «la LSTM
mejora al XGBoost» o «la LSTM no supera a la persistencia», y las dos frases habrían sido ciertas
sobre su corrida.

> **Ojo con la convención.** La desviación entre semillas se puede calcular de dos maneras, y la
> diferencia no es despreciable con cinco valores: la muestral (n−1) da 2,05 aquí y la poblacional
> (n) da 1,83. El notebook 8 recalcula la de la LSTM con el mismo criterio que la del GRU para que
> las dos sean comparables, y es su par —**GRU 6,05 | LSTM 2,05**— el que debe citarse. Mezclar las
> dos convenciones en una misma comparación es un error fácil de cometer y difícil de ver.

### Cuánto costó

**976 segundos** el notebook completo: 411 de rejilla —quince entrenamientos que dan dieciocho
configuraciones, porque se evalúa en cortes de un mismo entrenamiento— y 536 de validación
progresiva con las cinco semillas. Frente a los segundos que tarda el SARIMAX, por un resultado
peor.

---

# 10. El GRU y el híbrido

Los dos últimos modelos del trabajo: el **GRU** y un híbrido **BiLSTM con atención y
descomposición**. Entre los dos cierran la comparación, y los dos dan una respuesta que no era la
esperada.

## 10.1 El GRU, y por qué se compara con la LSTM

**GRU** significa *Gated Recurrent Unit*, unidad recurrente con compuertas. Resuelve el mismo
problema del desvanecimiento que la LSTM, con **dos** compuertas en vez de tres: fusiona la de
olvido y la de entrada en una sola, que decide a la vez cuánto conservar y cuánto incorporar, y
prescinde de la de salida. Menos compuertas significan menos parámetros y un entrenamiento más
rápido.

La pregunta empírica es si la complejidad adicional de la LSTM se paga. Para que la comparación la
responda, el montaje tiene que ser **idéntico**: misma ventana, mismas estáticas, mismo objetivo
relativo, misma rejilla y mismas semillas. La única diferencia debe ser la celda recurrente.

**La respuesta es que no se paga.** Con 4878 parámetros frente a 5966 —un 18,2 % menos—, el GRU da
un MAE de 132,47 sobre el promedio del mes frente a los 144,83 de la LSTM. Esa única diferencia de
diseño vale **12,37 COP/kWh**, y convierte a la familia de aprendizaje profundo, que con la LSTM
era la peor de las cinco configuraciones, en la ganadora.

Hay un matiz que no se puede separar de esa cifra. El óptimo del GRU está en **15 épocas** y el de
la LSTM en **3**: la red con menos capacidad aguanta cinco veces más entrenamiento antes de
memorizar, que es lo que la hipótesis predecía. Pero la misma hipótesis predecía que sería más
estable, y ahí falla: su desviación entre semillas es de **6,05 COP/kWh** frente a **2,05** de la
LSTM, medidas las dos con el mismo criterio, y **dos de sus cinco semillas quedan por detrás del
SARIMAX**. El GRU gana como procedimiento de cinco semillas, no como una corrida cualquiera.

## 10.2 Bidireccional, atención y descomposición

El híbrido combina tres ideas:

- **BiLSTM** (bidireccional): procesa la secuencia hacia adelante **y** hacia atrás, de modo que el
  estado de cada paso conoce también lo que viene después. Solo se puede aplicar sobre la **ventana
  histórica de entrada**, nunca sobre el horizonte a pronosticar; hacerlo sobre el horizonte sería
  leer el futuro, y es un error que aparece en la literatura con cierta frecuencia.
- **Atención**: en vez de resumir toda la ventana en un único estado, la red aprende **a qué días
  mirar** para cada predicción. Resuelve el cuello de botella del estado único y, de paso, es
  interpretable: se puede ver dónde miró.
- **Descomposición**: separar la serie en tendencia y residuo *antes* de la red, y modelarlos por
  separado. La red se concentra en lo difícil en vez de gastar capacidad en reaprender la tendencia.

**Las tres juntas quedan terceras**, con 140,17 sobre el promedio del mes: por delante del XGBoost
y de la LSTM, por detrás del SARIMAX y del GRU. Y lo hacen con 5486 parámetros, **menos que los
5966 de la LSTM**, de modo que no es cuestión de tamaño sino de mecanismo: encadenar más técnicas
no mejora el pronóstico a este horizonte. Es un hallazgo del trabajo, no una decepción, y es la
clase de resultado que solo aparece cuando se compara bajo condiciones idénticas.

## 10.3 Lo que **no** cambia

Da igual lo sofisticado que sea el modelo: **el protocolo del capítulo 4 es innegociable**. Mismo
`4_dataset_seleccion.csv`, misma partición por fecha, mismo embargo de enero de 2023, mismos 1.155
orígenes de prueba, mismas tres métricas en COP/kWh, varias semillas, y cierre con `registrar()`.

Y una exigencia que el cambio de alcance añadió: **el modelo tiene que entregar los treinta valores**,
no uno. Las tres lecturas se calculan después con `lecturas()`, que es compartida; ningún notebook
construye sus propios resúmenes.

Si un modelo nuevo cambiara cualquiera de esas cosas, su cifra dejaría de ser comparable y el
trabajo entero perdería su razón de ser.

## 10.4 La comparación, cerrada

Con las cuatro familias corridas bajo el mismo protocolo, el orden es el mismo en las tres lecturas:

| | `pares_30d` | `promedio_30d` | `dia_30` |
|---|---:|---:|---:|
| **GRU** | **163,17** | **132,47** | **215,56** |
| SARIMAX | 170,49 | 136,70 | 219,52 |
| Híbrido | 172,30 | 140,17 | 226,45 |
| XGBoost | 175,59 | 143,17 | 232,74 |
| LSTM | 176,93 | 144,83 | 233,48 |
| Persistencia | 178,37 | 145,66 | 239,27 |

Que el orden coincida en las tres es lo que permite afirmar algo: la conclusión no depende de cómo
se decidió puntuar.

Tres cosas hay que leer juntas, porque por separado inducen a error:

1. **El GRU gana, pero gana el conjunto de sus cinco semillas.** Dos de ellas pierden contra el
   SARIMAX. Ninguna cifra del GRU debe citarse sin su dispersión al lado.
2. **El modelo más complejo no es el mejor.** El híbrido queda tercero y con menos parámetros que
   la LSTM: el coste no compra precisión a este horizonte.
3. **Ningún modelo gana en todas partes.** El SARIMAX es el mejor durante El Niño y en 2024; el
   GRU, en condiciones neutrales, durante La Niña y en 2025 y 2026; el híbrido, en 2023. Por eso la
   evaluación se reporta segmentada por régimen y no solo en agregado.

Una respuesta negativa sobre una familia costosa también es un resultado: contesta la pregunta del
trabajo, que es **qué familia funciona mejor aquí**. Lo que no sería admisible es presentar a una
familia como ganadora cambiando la lectura con la que se la mide, o eligiendo la semilla que mejor
le sienta — y con 6,05 COP/kWh de dispersión en la familia que gana, esa tentación es real.

## 10.5 ¿Y si la diferencia fuera casualidad?

Queda una pregunta que la tabla de arriba no contesta. El GRU acertó más que el SARIMAX **en este
periodo de prueba**. ¿Habría acertado más en otro? Dicho de otro modo: ¿son 4,24 COP/kWh una
ventaja real o el tipo de diferencia que aparece sola al cambiar las fechas?

Eso es lo que mide la **prueba de Diebold-Mariano**, en la celda 9.4 del cuaderno 9. La idea es
simple: en cada uno de los 1.155 orígenes se mira cuánto erró cada modelo, se resta, y se pregunta
si ese montón de restas tiene un promedio lo bastante lejos de cero como para no atribuirlo al
azar. Si las restas son unas positivas y otras negativas y solo *en promedio* salen a favor de
uno, la ventaja es frágil.

**Aquí hay una trampa que decide el resultado.** Un origen pronostica del 1 al 30 de marzo; el
siguiente, del 2 al 31. Comparten veintinueve de sus treinta días. Si se tratara a los 1.155
orígenes como si fueran 1.155 pruebas independientes, se estaría contando casi mil veces la misma
información. La prueba corrige eso —con lo que en estadística se llama una varianza de largo
plazo— y el efecto es grande: **los 1.155 orígenes valen como poco más de un centenar de pruebas
independientes**.

El resultado, con la corrección puesta: de los quince pares que se pueden formar con las cinco
configuraciones y la persistencia, **ninguno alcanza significancia estadística**. Ni el GRU contra
el SARIMAX, ni el GRU contra la persistencia. Sin corregir el solapamiento, ocho de los quince
parecerían significativos, y esa es exactamente la lectura equivocada que la corrección evita.

Conviene no sacar de ahí más de lo que dice, ni menos:

- **No dice que los modelos sean iguales.** El GRU tiene el menor error, el menor SMAPE, y mejora
  a la persistencia en los treinta horizontes. Eso ocurrió, y está medido.
- **Sí dice que tres años de prueba no bastan para certificarlo.** Con errores del orden de 130
  COP/kWh y ventanas que se solapan, una diferencia de 4 a 13 COP/kWh no se distingue del ruido.
- **Por eso el trabajo afirma que el GRU fue el mejor sobre este periodo**, y no que sea el mejor
  en general. La diferencia entre esas dos frases es el contenido entero de esta sección.

---

# Glosario

| Término | Definición en una línea |
|---|---|
| **ACF** | Cuánto se parece el valor de hoy al de hace *k* días |
| **ADF** | Prueba cuya hipótesis nula es que hay raíz unitaria |
| **Atención** | Mecanismo que deja a la red elegir qué partes de la entrada mirar |
| **Autocorrelación** | Correlación de una serie consigo misma desplazada en el tiempo |
| **Embargo** | Hueco entre entrenamiento y prueba que nadie usa |
| **Estacionariedad** | Que las propiedades de la serie no cambien con el tiempo |
| **Exógena** | Variable externa que entra al modelo como regresor |
| **Fuga de información** | Usar para entrenar algo que no habría estado disponible |
| **Heterocedasticidad** | Varianza que no es constante en el tiempo |
| **Hiperparámetro** | Decisión tomada antes de entrenar, que el modelo no aprende |
| **Horizonte** | A cuántos periodos se pronostica (aquí, de 1 a 30 días) |
| **KPSS** | Prueba cuya hipótesis nula es que la serie es estacionaria |
| **Lectura** | Una de las tres formas de resumir una trayectoria: `pares_30d`, `promedio_30d`, `dia_30` |
| **Ljung-Box** | Prueba de si los residuos conservan autocorrelación |
| **MAE** | Error absoluto medio, en las unidades de la variable |
| **Origen** | Día desde el que se hace un pronóstico |
| **Par origen-horizonte** | Un valor pronosticado: el día *h* visto desde el origen *t*. Hay 1.155 × 30 = 34.650 |
| **p-valor** | Probabilidad de ver estos datos si la hipótesis nula fuera cierta |
| **PACF** | Autocorrelación descontando los rezagos intermedios |
| **Permutación (importancia por)** | Cuánto empeora el modelo al desordenar una variable |
| **Persistencia** | Pronosticar que el valor futuro será igual al actual |
| **Purga** | Excluir del ajuste los orígenes cuyo objetivo aún no se conocía |
| **Raíz unitaria** | Serie sin ancla: los choques quedan incorporados para siempre |
| **Regularización** | Restringir el modelo para que no memorice ruido |
| **RMSE** | Error cuadrático medio; castiga más los errores grandes |
| **Semilla** | Número que fija el azar y hace reproducible un resultado |
| **Sesgo** | Error promedio **con** signo: hacia qué lado se equivoca |
| **SHAP** | Reparto de cada predicción entre las variables que la produjeron |
| **SMAPE** | Error porcentual simétrico |
| **Trayectoria** | Los 30 valores que un modelo pronostica desde un origen |
| **VIF** | Cuán reconstruible es una variable a partir de las demás |
| **Walk-forward** | Validar avanzando en el tiempo, como se usaría de verdad |

---

*Las cifras de este documento provienen de las salidas ejecutadas de los notebooks del repositorio.
Si una corrida nueva las cambia, este documento hay que actualizarlo: las celdas de markdown no se
revalidan solas.*
