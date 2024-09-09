En esta sección encontrarás una descripción del contenido de los archivos generados en los procesos *get_recortes.py* y *get_predicciones.py*.

## problemas_imagenes.xlsx
Este archivo se genera al terminar el proceso de *get_recortes.py*, el cual nos indica las preguntas que tuvieron un problema en la obtención del recorte. Esta tabla se compone de 3 columnas, las cuales se describen a continuación:


| Columna| Descripción|
|---|---|
| Pregunta  | Ruta de la imagen que no pudo ser procesada. |
| Error | Detalle del error generado en el procesamiento. Esto puede ser por errores de carga, número incorrecto de subpreguntas, errores al detectar líneas o recuadros, entre otras. |
| Nivel | Nivel del error, el cual puede ser a nivel de pregunta, subpregunta, estudiantes o padres (usado principalmente para debuggear problemas en los códigos del flujo).|


## predicciones_modelo_final.xlsx
Este archivo se genera al terminar el proceso de *get_predicciones.py*. Aquí encontraremos las predicciones del modelo de detección de dobles marcas y los indices de tinta que se identificaron para cada pregunta/subpregunta. Este archivo se compone por 18 columnas, en donde 9 corresponden a la información predicha por el modelo y los indices de tinta, los cuales describiremos a continuación:

|Columna| Descripción|
|---|---|
|pred| Etiqueta de la predicción obtenida, en donde 1 nos indica que la pregunta/subpregunta corresponde a una doble marca y 0 que no es doble marca. |
|proba| *Probabilidad* de la etiqueta. Nos indica que tan seguro está el modelo de su etiqueta predicha, mientras más cercano a 1, más seguro esta el modelo de su decisión. Por otro lado, mientras más cercano a 0.5, menos seguro estará el modelo de la etiqueta indicada.
|indices (*)| Tupla de valores que nos indican la proporción e intensidad promedio de tinta presente en los dos recuadros con mayores valores. |
|indice_tinta_top1| Proporción de tinta promedio del recuadro con mayor valor.|
|indice_tinta_top2| Proporción de tinta promedio del segundo recuadro con mayor valor.|
|indice_intensidad_top1|Intensidad de tinta promedio del recuadro con mayor valor.|
|indice_intensidad_top2 |Intensidad de tinta promedio del segundo recuadro con mayor valor.|
|ratio_tinta| Razón entre la proporción media de tinta presente en los dos recuadros con más tinta (indice_tinta_top1/indice_tinta_top2). Si el valor es 1, nos indica que los dos recuadros evaluados poseen la misma cantidad de tinta rellana en el recuadro.|
|ratio_intensidad| Razón entre la intensidad media de tinta presente en los dos recuadros con mayor tinta (indice_intensidad_top1/indice_intensidad_top2). Si el valor es 1, nos indica que los dos recuadros poseen la misma intensidad de trazos realizados en el recuadro|

Las 9 columnas faltantes nos ayudan a describir la pregunta/subrpregunta analizada (ubicación directorio, serie, rbd, CodigoCurso, entre otras).

#### Indices de tinta (*)

Los indices de tinta se dividen en dos métricas principales; la proporción del recuadro que posee una marca y la intensidad con la que se realizó aquella marca. Veamos los siguientes ejemplos:

| ![tinta1](tinta1.png#center)| 
|:--:| 
| *Figura 1*|

|![tinta2](tinta2.jpg#center)|
|:--:| 
| *Figura 2* |

En la Figura 1, podemos notar que existen marcas realizadas con lápiz mina, en donde tenemos que una de ellas fue parcialmente borrada (3er recuadro). En esta imagen la proporción promedio de marca en los recuadros 2 y 3 fueron de  0.067 y 0.02, es decir, dentro del recuadro blanco, tenemos que el 6.7% y 2.0% posee *tinta* de lápiz. Por otro lado, sabemos que uno posee un *borrón* con goma, por lo que si bien se detectó una marca tinta, estas marcas poseen diferentes intensidades, para ser más exactos, tenemos que la intensidad del trazado es de 37.3% y 16.5% respectivamente. 

Ahora, fijándonos en la Figura 2, tenemos que las marcas fueron realizadas con lápiz pasta y que uno de los recuadros fue casi completamente marcado. En este caso, la proporción media de marca para los recuadros 1 y 4 corresponde al 79.5% y 17.0% y la intensidad del trazado es de 81.2%, 54.7% respectivamente.
