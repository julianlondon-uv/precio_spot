# De dónde sale cada figura del documento

Las figuras del documento se sustituyen por la salida ejecutada del cuaderno que las dibuja, y la
correspondencia se resuelve **comparando las imágenes**: la del documento contra las de todas las
celdas, por su contenido. Funciona para la mayoría y es lo que conviene, porque no hay que
mantener ninguna lista.

**No funciona cuando el documento tiene varias figuras del mismo tipo.** Los cuatro pronósticos
por familia son el mismo gráfico con otro modelo, y lo mismo las cuatro curvas de aprendizaje: dos
imágenes casi iguales se confunden, y el documento acabaría con el pronóstico del XGBoost bajo la
leyenda de la LSTM. Dos cosas lo evitan: exigir que cada imagen se use **una sola vez**, y leer en
la leyenda el **nombre del modelo**, que ata la figura al cuaderno de esa familia.

Aun así quedan unas pocas que hay que decidir mirándolas. Son estas, y aquí queda dicho:

| Figura | Sale de | Por qué no se resuelve sola |
|---|---|---|
| 18 | `3 c81 #0` | la matriz de correlación con dendrograma cambió mucho al subir la letra |
| 19 | `3 c73 #0` | se parece a los demás gráficos de barras horizontales del cuaderno 3 |
| 21 | `4 c34 #0` | se parece a las tres figuras de importancia del cuaderno 4 |
| 25 | `5 c41 #1` | la trayectoria desde un origen se parece a los demás gráficos de línea del cuaderno 5 |
| 27 | `6 c35 #0` | el cuaderno 6 dibuja el pronóstico **dos veces**: `c24` a trece pulgadas, para mirarlo en pantalla, y `c35` al ancho de la página. Las dos imágenes son casi iguales y el parecido elegía la primera, cuya letra, metida en la columna, sale a la mitad |
| 28 | `6 c33 #2` | curva de aprendizaje: la misma forma en los cuatro cuadernos de modelo |
| 30 | `7 c21 #1` | igual |
| 32 | `8 c22 #1` | igual |
| 34 | `9 c23 #1` | igual |
| 35 | `9 c30 #0` | error por horizonte de las cinco familias, junto a la versión de cada una |
| 36 | `9 c24 #1` | el cuaderno 9 dibuja dos vistas del error a lo largo del periodo y solo una va al documento |

La clave es `<cuaderno> c<celda> #<salida>`: el número del cuaderno, el índice de la celda dentro
del `.ipynb` y cuál de sus imágenes. **Cambian si se insertan o se borran celdas**, de modo que
si una sustitución sale rara hay que volver a mirarla y corregir la fila.

## La escala no se ve en la imagen

Una figura dibujada a trece pulgadas y metida en siete sale con la letra a la mitad, y **nada en
el documento lo delata**: la imagen tiene los píxeles que debe y se inserta al ancho correcto. Lo
único que lo dice es el `figsize` de la celda que la dibujó. Los cuadernos tienen celdas de las
dos clases —las de exploración, anchas, y las que van al documento, a `ANCHO_PAGINA`—, y alguna
figura está dibujada **dos veces**, una de cada manera. Por eso el instalador comprueba, antes de
llevar una imagen al documento, que su celda la dibuja al ancho de la página, y se detiene si no.

Los **cinco diagramas conceptuales** —las figuras 1, 2, 3, 20 y 24— no salen de ningún cuaderno;
su correspondencia está en `figuras_diagrama/README.md`.

## Cómo volver a decidirlo si hace falta

`contactos.py`, en la carpeta de trabajo, escribe por cada figura dudosa una tira con la imagen
que el documento tiene ahora y sus candidatas más próximas, numeradas y con su distancia. Se mira
la tira y se corrige la fila de esta tabla. No hay forma de hacerlo bien sin mirar: la distancia
entre miniaturas ordena mal justamente en estos casos, y en más de una la candidata correcta era
la segunda o la tercera.
