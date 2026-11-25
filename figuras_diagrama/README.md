# Los cinco diagramas del documento

Aquí viven los scripts que dibujan las **figuras conceptuales** del documento: las que explican un
procedimiento en vez de mostrar un resultado medido. No son salida de ningún cuaderno, y por eso
están aparte.

| Script | Produce | Figura del documento |
|---|---|---|
| `fig_crispdm.py` | `fig_crispdm.png` | 1. Metodología CRISP-DM aplicada al trabajo |
| `fig_proceso_datos.py` | `fig_proceso_datos.png` | 2. Recorrido de los datos desde las fuentes hasta la evaluación |
| `fig_extraccion.py` | `fig_extraccion.png` | 3. Extracción de los datos |
| `fig_particion_v2.py` | `fig16_particion.png` | 20. Partición, pliegues y anatomía de un origen |
| `fig_seleccion.py` | `fig_seleccion.png` | 24. Recorrido de cada variable por los tres filtros |

**Esta tabla es la única fuente del emparejamiento.** El instalador que sustituye las imágenes del
documento la lee para saber qué PNG va en qué figura. Las figuras de análisis se emparejan por su
contenido, comparando la imagen del documento con la salida ejecutada de cada celda; con estas
cinco no se puede, porque al rehacerlas el dibujo cambia lo bastante como para que la comparación
no sea fiable. Si se añade o se renumera un diagrama, hay que corregirlo **aquí**.

## Por qué están aquí y no en una carpeta temporal

Hasta noviembre de 2026 estos scripts vivían solo en una carpeta temporal de trabajo. El documento
se apoyaba en cinco figuras que, si esa carpeta se borraba, dejaban de ser reproducibles. Las
figuras de **análisis** no tienen ese problema porque son la salida ejecutada de una celda y viajan
dentro del `.ipynb`; estas no, porque no las dibuja ningún cuaderno.

## La escala, que es la misma que la de los cuadernos

Los cinco dibujan a **1.289 píxeles de ancho y 186 puntos por pulgada**, es decir a las 6,93
pulgadas del ancho útil de la página, y lo comprueban con un `assert` sobre el tamaño del PNG que
acaban de escribir. Así cada punto tipográfico del dibujo es un punto sobre el papel, siempre que
la figura se inserte en el documento al ancho de la columna y no más estrecha.

## La letra: el cuerpo del documento, y la geometría deducida de él

Los cinco están a **10 puntos**, el cuerpo del documento, con 11 para el nombre de una caja o el
título de un panel. No se llegó ahí subiendo la fuente sobre la disposición anterior: a 10 puntos
el texto de una caja invadía la de al lado, y el pie se salía por los costados. Lo que se hizo fue

1. **recortar el texto** —en los diagramas de cajas ninguna línea pasa de veinticuatro caracteres,
   y se quitaron las aclaraciones entre paréntesis y los títulos interiores que repetían la
   leyenda de Word—, y
2. **deducir la geometría de los tamaños en puntos** en vez de fijar la geometría y ver qué letra
   cabía. En `fig_crispdm.py`, `fig_proceso_datos.py` y `fig_extraccion.py` toda medida se expresa
   en puntos tipográficos y se convierte a coordenadas con `ux()` y `uy()`; si cambia el cuerpo,
   las cajas cambian con él.

Los tres diagramas de cajas llevan además un **guard que mide el texto ya dibujado** y comprueba
que cabe dentro de su caja. Un retoque que alargue una línea falla al generar la figura, no en la
página impresa.

## Qué hay que mirar si se retoca uno

Lo que rompió la primera versión a 10 puntos, por si vuelve a aparecer:

- **rótulos anclados al centro de su barra**: cuando el rótulo mide casi tanto como la barra, dos
  rótulos contiguos se tocan en el medio. Se anclan a los extremos del panel.
- **dos fechas próximas sobre un eje largo**: el fin del entrenamiento y el principio de la prueba
  distan 38 meses sobre dieciséis años, y a cuerpo del documento se pisan. Van dentro de una nota.
- **guías de anotación que cruzan otro rótulo**: la nota del embargo pasó a ir debajo de la barra.
- **encabezados de etapa más anchos que su columna**: en el embudo de la selección van escalonados
  en dos alturas y con el nombre en una sola línea.
- **márgenes de la figura**: el nombre de variable más largo del embudo y las etiquetas «Pliegue N»
  necesitan bastante más margen izquierdo que con la letra pequeña.
