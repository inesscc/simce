En este manual revisaremos qué son y qué información contienen los archivos de configuración. Cada script de procesamiento tiene asociado un archivo, ambos en la carpeta `config`:

- `proc_img.py` para el script de recorte de imágenes 
- Un archivo `.json` que actualmente llamamos `config_pred.json` para el script de predicción de dobles marcas. A diferencia del primero, este puede cambiar de nombre sin problema, pues debe ingresarse su ruta al llamar el script de predicción de dobles marcas, como se vio en [su sección de la documentación](../tutorial#get_predicciones.py).

## proc_img.py

Este script centraliza los valores que podrían variar entre prueba y prueba a lo largo de los scripts que llevan al recorte de subpreguntas. De esta forma, al cambiar valores acá, se cambiarán automáticamente todas las menciones a estos valores en los otros script. La idea, entonces, es que, en caso de necesitar hacer cambios al script entregado, lo más probable es que tengan que ser hechos acá. A continuación explicamos en detalle algunas de las líneas.

### Es bastante probable que estos valores deban ser modificados:

```py linenums="1"
CURSO = Path('4b')
```

Acá se define qué curso está siendo procesado. En este caso, estamos procesando 4° básico. Esto afecta principalmente nombres de archivos y carpetas que dan orden al proyecto, en la medida que se van generando. Por ejemplo, los datos de subpreguntas se guardan en <pre>data/input_proc/<b>4b</b>/subpreg_recortadas</pre>.

```py linenums="2"
carpeta_estudiantes = 'CE'
carpeta_padres = 'CP'
```

Es el nombre de las carpetas de padres y estudiantes. Se utilizan para buscar en el directorio de archivos estas carpetas, además de que se crearán carpetas con estos nombres en los outputs. Si se dejara de usar esta nomenclatura en el sistema de archivos, deberían cambiarse estos valores a algo que corresponda a la nueva nomenclatura.

``` py linenums="4"
nombres_tablas_origen = {'padres': f'{carpeta_padres}_Origen_DobleMarca.csv',
                 'estudiantes': f'{carpeta_estudiantes}_Origen_DobleMarca.csv'}
```
Contiene los nombres de las tablas de Origen. Si los nombres cambian, deben ser cambiados aquí también. Notar que se hace referencia a las variables `carpeta_padres` y `carpeta_estudiantes`, que se mencionaron anteriormente. Por lo tanto, por ejemplo, para padres, el nombre original del archivo era `CP_Origen_DobleMarca.csv`.


``` py linenums="6"
nombre_tabla_para_insumos = 'DD 4° BÁSICO 2023_CE_CP.xlsx'
```
Tabla que se utiliza para parte de la generación de insumos. De esta se pueden extraer número de preguntas, de subpreguntas, de opciones por subpregunta. Deberá ser actualizada al nombre que corresponda en su siguiente versión.

``` py linenums="7"
n_filas_ignorar_tabla_insumos = 4
```
En la versión que recibimos de la tabla, era necesario saltarse las primeras 4 filas para acceder a los datos que nos interesan. Lo dejamos parametrizado, en caso de que esto cambie.

``` py linenums="8"
nombre_col_campo_bd = 'Nombre Campo BD'
nombre_col_val_permitidos = 'Rango de valores Permitidos'
```

Representa el nombre de la columna que contiene los nombres de los campos de la base de datos y los valores permitidos para cada subpregunta. Actualizar si el nombre de alguna de estas columnas cambia.

``` py linenums="10"
IP_NAS = '10.10.100.28'
FOLDER_DATOS = '4b_2023' # OJO, actualizar
```
Valores utilizados para la conexión al disco NAS donde están los datos de imágenes. Si la IP o la carpeta de datos cambiara, debe ser ingresado aquí. 

### Es relativamente probable que estos valores deban ser modificados:

``` py linenums="12"
id_estudiante = 'serie'
variables_identificadoras = ['rbd', 'dvRbd', 'codigoCurso', id_estudiante, 'rutaImagen1']
```

Representan variables que utilizaremos de la tabla origen. Si cualquiera de estas variables cambia de nombre en la tabla origen, deberán ser actualizadas acá.


``` py linenums="14"
regex_extraer_rbd_de_ruta = r'\\(\d+)\\'
```

Expresión regular que el RBD de la ruta que se encuentra en la variable rutaImagen1 (busca uno o más dígitos consecutivos rodeados de paréntesis).

``` py linenums="15"
dic_ignorar_p1 = {'estudiantes': True, 'padres': False}
```
Determina si la pregunta 1 debe ser ignorada o no en cada cuadernillo. En el caso que estudiamos nosotros, el cuadernillo de estudiantes preguntaba la edad en la pregunta 1, por lo que debía ser ignorada. En el caso de los padres, era una pregunta común y corriente, por lo que no se ignora.


```py linenums="16"
regex_estudiante = r'\d{7,}'
```

Acá se define cómo extraeremos los identificadores de los estudiantes de rutas de archivos, principalmente. Lo que busca este código son 7 números seguidos. Por lo tanto, si el identificador de estudiante cambiara a más o menos números o incorporara letras, habría que modificar esto para que funcione de manera acorde.

```py linenums="17"
ENCODING = 'utf-8'
```

Define cuál es el encoding utilizado a la hora de leer la tabla de origen. Nos pasó en una ocasión que se usaba encoding "Latin-1" para una tabla de octavo básico, por lo que decidimos parametrizar esto.

```py linenums="18"
LIMPIAR_RUTA = False
```

En algunos casos las rutas a los archivos de los cuadernillos en la tabla de origen, incluían el identificador del alumno como una carpeta. En general, nosotros trabajamos de forma que dentro de las carpetas de cada rbd se encontraban todas las imágenes, sin una carpeta intermedia por cada alumno del rbd. Así, cuando pasa esto, este parámetro elimina de la ruta el identificador del alumno.

``` py linenums="19"
IGNORAR_PRIMERA_PAGINA = True
```

Este parámetro define si se debe ignorar la primera página en la recolección de insumos. Esto, porque usualmente la primera página tiene preguntas de ejemplo, que deben ser ignoradas. Se agrega en caso en que alguna iteración contenga alguna pregunta que no sea de ejemplo en la primera página.

### Es menos probable que estos valores deban ser modificados:

```py linenums="20"
n_pixeles_entre_lineas = 22
```

Especifica cuántos píxeles se espera que haya como mínimo entre dos líneas separadoras de subpreguntas. Debiera ser relevante solo si cambiara el diseño del cuestionario de forma que las subpreguntas queden más (o menos) espaciadas.



```py linenums="21" hl_lines="11 12"
def get_directorios(curso, filtro=None) -> dict:
    '''Acá se indican todos los directorios del proyecto. Luego, la función crear_directorios() toma todos
    los directorios de este diccionario y los crea. La opción filtro permite cargar solo algunos directorios,
    en caso de requerirse. Si el filtro '''
    dd = dict()
    dd['dir_data'] = Path('data/')
    dd['dir_input'] = dd['dir_data'] / 'input_raw' 

    # En producción nos conectamos a disco NAS para acceso a imágenes
    if os.getenv('ENV') == 'production':
        conectar_a_NAS(IP_NAS, FOLDER_DATOS)
        dd['dir_img_bruta'] = Path('P:/')

    else:
        # Solo aplica a desarrollo local:
        dd['dir_img_bruta'] = dd['dir_input']  
    dd['dir_estudiantes'] = dd['dir_input'] / carpeta_estudiantes
    dd['dir_padres'] = dd['dir_input'] / carpeta_padres

    dd['dir_input_proc'] = Path('data/input_proc/')
    dd['dir_subpreg_aux'] = dd['dir_input_proc'] / curso / 'subpreg_recortadas'
    dd['dir_subpreg'] = dd['dir_subpreg_aux'] / 'base'
    dd['dir_subpreg_aug'] = dd['dir_subpreg_aux'] / 'augmented'


    dd['dir_tabla_99'] = dd['dir_input_proc'] / 'output_tabla_99'
    dd['dir_insumos'] = dd['dir_input_proc'] / curso /  'insumos'

    dd['dir_train_test'] = dd['dir_data'] / 'input_modelamiento'

    dd['dir_output'] = Path('data/output')
    dd['dir_modelos'] = dd['dir_output'] / 'modelos' 
    dd['dir_predicciones'] = dd['dir_output'] / 'predicciones'

    
    
    if filtro:
        if isinstance(filtro, str):
            filtro = [filtro]
        
        dd = { k:v for k,v in dd.items() if k in filtro }

        if len(filtro) == 1:

            dd = list(dd.values())[0]

    return dd
```

Esta función permite obtener todos los directorios del proyecto. Además tiene la opción de filtrar, de forma de obtener algún directorio específico que a uno pudiera interesarle. Si uno quisiera cambiar el nombre de algún directorio, en este lugar debe hacerse. Destacamos en el código dónde se genera la conexión a la máquina NAS que contiene las imágenes de los cuadernillos.

``` py linenums="68"
regex_hoja_cuadernillo = r'_(\d+)'
```

Esta expresión regular se utiliza para extraer el número de la imagen asociada a un cuadernillo. Estas en general tienen un formato `{identificador_estudiante}_1.jpg`, `{identificador_estudiante}_2.jpg`, etc. Entonces, esta expresión regular busca encontrar un guión bajo(_) y uno o más números y luego extrae ese número o números. Se  utiliza para determinar qué preguntas aparecen en qué archivos de imágenes en la [recolección de insumos, en la función que puebla el diccionario de preguntas](../generar_insumos_img#simce.generar_insumos_img.poblar_diccionario_preguntas).


``` py linenums="69"
regex_p1 = r'p1(_\d+)?$'
```

Expresión regular que identifica la pregunta 1, busca que el string termine con "p1" y opcionalmente que puedan existir subpreguntas con un string tipo "p1_1". Se utiliza para quitar la pregunta 1 del análisis en los cuadernillos en que se indique que así debe ser, según el diccionario de la sección anterior.

Estos últimos valores pueden ser ignorados sin problema, están asociados al entrenamiento:

``` py linenums="70"
SEED = 2024
```

Esta es una semilla que permite hacer replicables los resultados. Si se cambia, cada etapa del procesamiento que contenga algo de aleatoriedad, tendrá resultados distintos. Por lo tanto, **es mejor no tocarla**.

``` py linenums="71"
FRAC_SAMPLE = .05
# n° de rondas de aumentado de datos (máximo 5):
N_AUGMENT_ROUNDS = 5
```

Determina qué porcentaje de los datos de estudiantes con doble marca serán  y cuántas rondas de aumentado de datos serán realizadas para el entrenamiento. No son relevantes para esta etapa del proyecto.

``` py linenums="72"
nombre_tabla_predicciones = 'data_pred.csv'
```

Es el nombre de la tabla que contiene las predicciones. Quedó parametrizada para poder hacer referencia a ella en la siguiente sección.

## config_pred.json

Este archivo de configuración, en cambio, es un archivo `.json`, que tiene una estructura anidada. La gran mayoría de sus valores está avocada al entrenamiento, así que no los detallaremos aquí.

```json hl_lines="4"
    "data_loader": {
        "type": "TrainTestDataLoader",
        "args": {
            "data_file": "data_pred.csv",
            "batch_size": 64,
            "shuffle": false,
            "cortar_bordes": false,
            "num_workers": 4,

            "validation_split": 0.0
        }
    },

```

Esta sección hace referencia al data_loader, que es el objeto que se encarga de cargar las imágenes y pasárselas al modelo. Lo importante es que el nombre del `data_file` sea el mismo que [nombre_tabla_predicciones](#__codelineno-20-72) en la sección anterior.

**Importante:** no tocar los otros valores, pues podrían llevar a problemas al predecir.

El resto del archivo de configuración puede ignorarse, lo que no implica que se le pueden borrar los otros elementos, pues muchos de ellos tienen que ver con la configuración del modelo (que es fija, a menos que re-entrenáramos un modelo). 