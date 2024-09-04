Acá se explica cómo correr el modelo y las distintas opciones presentes para los scripts que se hacen cargo de esto.

## get_recortes.py

El primer script se encarga del recorte y sistematización de subpreguntas. De acuerdo a lo presentado en el [flujo del proyecto](../#esquema-del-proyecto), esto considera la etapa de generación de insumos, procesamiento de la tabla de dobles marcas y selección y procesamiento de subpreguntas. Durante este proceso se genera la tabla `problemas_imagenes.xlsx`. Respecto a su uso, este es sencillo, los pasos son los siguientes:

1. Ubicarse en la raíz del proyecto. 

2. Correr el siguiente comando en la consola:

    `python get_recortes.py`

    Además, el script cuenta con dos argumentos opcionales. El primero es `--curso`, donde se le indica un `<id_curso>` que corresponde al identificador del curso siendo procesado. Por ejemplo, para cuarto básico, utilizamos:

    `python get_recortes.py --curso 4b`

    Ojo, el curso también puede ser indicado en el [script de configuración](script_configuracion.md), en la variable `CURSO`. Si el curso se indica en ambos archivos, tomará prioridad el que se indica en la línea de comandos. Además, el script de configuración maneja las variables más importantes del modelo, por lo que **siempre es importante revisarlo y actualizarlo antes de correr este script**.

    En segunda instancia, el código cuenta con el comando opcional `-v` o `--verbose`, que si se agrega, hará que se imprima más información en la consola respecto al procesamiento de las imágenes:

    `python get_recortes.py -v`

    o equivalentemente:

    `python get_recortes.py --verbose`

    Ojo, el código puede tomar varias horas en correr. Se imprimirá un mensaje en la consola cuando termine el procesamiento, indicando esto mismo.

## get_predicciones.py

El primer script se encarga de la predicción y cálculo de indicadores de tinta, como puede verse en el [flujo del proyecto](../#esquema-del-proyecto). Durante este proceso se genera la tabla `predicciones_modelo_final.xlsx`. Respecto a su uso, este es sencillo, los pasos son los siguientes:

1. Ubicarse en la raíz del proyecto. 

2. Correr el siguiente comando en la consola:

    `python get_predicciones.py --config <ruta_config_file>`

    En este caso, `<ruta_config_file>` es la ruta al archivo de configuración del modelo. Por ejemplo:

    `python get_predicciones.py --config config/config_pred.json`

    En la sección de [explicación del script configuración](script_configuracion.md) se explica en más detalle cómo funciona este script.


