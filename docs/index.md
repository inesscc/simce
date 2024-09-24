# Modelo de detección de dobles marcas SIMCE

La presente documentación se encargará de explicar el código del proyecto SIMCE, así como su organización y cómo correr el modelo. La estructura del documento es la siguiente:

1. [Estructura del proyecto](#estructura-del-proyecto)
2. [Esquema del proyecto](#esquema-del-proyecto)
3. [Tutorial de uso](tutorial.md)
4. [Guía de outputs del proyecto](outputs.md)
5. [Uso del script de configuración](script_configuracion.md)
6. [Manuales de referencia](generar_insumos_img.md)

En el menú de la izquierda es posible navegar por el sitio y los distintos manuales de referencia.

## Estructura del proyecto

A continuación se presenta la estructura del proyecto, con el fin de que sea posible entender qué hacen los distintos archivos y carpetas del proyecto.

    simce/
    │
    ├── get_recortes.py - script principal para obtener recortes de subpreguntas
    ├── get_predicciones.py - script principal para obtener predicciones del modelo.
    ├── modelo_simce.pt - archivo que contiene el modelo que predice dobles marcas
    │
    ├── base/ - clases abstractas base
    │   ├── base_data_loader.py
    │   ├── base_model.py
    │   └── base_trainer.py
    │
    ├── config/ - carpeta de configuraciones del proyecto
    │   ├── config_pred.json - configuración del modelo para predicción
    │   ├── parse_config.py - clase que maneja archivo de configuración y opciones CLI
    │   └── proc_img.py - configuraciones asociadas al recorte de subpreguntas.
    │
    ├── data/ - carpeta con datos del modelo
    │   ├── input_modelamiento - aquí se guarda la tabla con datos a predecir, que recibe el modelo
    │   ├── input_procesado - aquí se guardan las subpreguntas recortadas y la tabla de dobles marcas procesada
    │   ├── input_bruto - aquí se guardan los datos brutos: tabla origen, tabla campos BD e imágenes si no se realiza conexión al disco NAS
    │   └── output - contiene las predicciones y posibles visualizaciones.
    │
    ├── data_loader/ - carga de datos en Torch 
    │   └── data_loaders.py
    │
    ├── dataset/ - clase que permite cargar dataset de imágenes en Torch
    │   └── dataset.py
    │
    ├── docs/ - archivos de documentación del proyecto
    │   ├── index.md
    │   └── ...
    │
    ├── logger/ - módulo para logging del modelo.
    │   ├── visualization.py
    │   ├── logger.py
    │   └── logger_config.json
    │    
    ├── model/ - modelos, pérdidas, y métricas
    │   ├── model.py
    │   ├── metric.py
    │   └── loss.py
    │
    ├── notebooks/ - notebooks con pruebas y análisis exploratorio 
    │
    ├── saved/ - modelos entrenados y sus logs correspondientes
    │   ├── models/ - trained models are saved here
    │   └── log/ - default logdir for tensorboard and logging output
    │
    ├── simce/ - funciones que procesan las distintas etapas del proyecto
    │   ├── errors.py - manejo de errores
    │   ├── generar_insumos_img.py - genera los insumos que posteriormente son utilizados en el recorte
    │   ├── indicadores_tinta.py - cálculo de indicadores de tinta
    │   ├── modelamiento.py - funciones utilizadas durante el modelamiento.
    │   ├── paralelizacion.py - funciones del recorte de subpreguntas que se ejecutan de forma paralelizada
    │   ├── predicciones.py - funciones para la predicción de datos nuevos
    │   ├── preparar_modelamiento.py - funciones que preparan datos para el modelamiento  
    │   ├── proc_imgs.py - funciones asociadas al procesamiento de imágenes previo al recorte
    │   └── proc_tabla_99.py - generación de tabla de dobles marcas
    │
    ├── site/ - archivos asociados a sitio web de documentación
    │    ├── assets/
    │    └── ...
    │
    └── trainer/ - clase trainer, permite entrenar un modelo
        └── trainer.py


## Esquema del proyecto

Aquí podemos ver las distintas etapas que se ejecutan a lo largo de los códigos, junto con los respectivos outputs:la tabla de errores en el procesamiento de imágenes (`problemas_imagenes.xlsx`) y la tabla de predicciones e indicadores (`predicciones_modelo_final.xlsx`)

<br/><br/>

![](flujo_proyecto.png#only-light)
![](flujo_proyecto_dark.png#only-dark)
