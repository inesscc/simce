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
|indices| Tupla de valores que nos indican el porcentaje e intensidad promedio de tinta presente en los dos recuadros con mayores valores. |
|indice_tinta_top1| Porcentaje de tinta promedio del recuadro con mayor valor.|
|indice_tinta_top2| Porcentaje de tinta promedio del segundo recuadro con mayor valor.|
|indice_intensidad_top1|Intensidad de tinta promedio del recuadro con mayor valor.|
|indice_intensidad_top2 |Intensidad de tinta promedio del segundo recuadro con mayor valor.|
|ratio_tinta| Razón entre el porcentaje medio de tinta presente en los dos recuadros con más tinta (indice_tinta_top1/indice_tinta_top2). Si el valor es 1, nos indica que los dos recuadros evaluados poseen la misma cantidad de tinta rellana en el recuadro.|
|ratio_intensidad| Razón entre la intensidad media de tinta presente en los dos recuadros con mayor tinta (indice_intensidad_top1/indice_intensidad_top2). Si el valor es 1, nos indica que los dos recuadros poseen la misma intensidad de trazos realizados en el recuadro|

Las 9 columnas faltantes nos ayudan a describir la pregunta/subrepgunta analizada (ubicación directorio, serie, rbd, CodigoCurso, entre otras).
