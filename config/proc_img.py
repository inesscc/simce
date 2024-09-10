# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:50:14 2024

@author: jeconchao
"""

from pathlib import Path
from dotenv import load_dotenv
import os
load_dotenv()

# 1. VARIABLES QUE ES MUY PROBABLE QUE DEBAN SER ACTUALIZADAS --------
CURSO = Path('4b')

## Tablas Origen -----
carpeta_estudiantes = 'CE'
carpeta_padres = 'CP'
nombres_tablas_origen = {'padres': f'{carpeta_padres}_Origen_DobleMarca.csv',
                 'estudiantes': f'{carpeta_estudiantes}_Origen_DobleMarca.csv'}


## Tabla campos BD -----
# Nombre de tabla que contiene n° de subpreguntas, n° de recuadros por subpregunta:
nombre_tabla_para_insumos = 'DD 4° BÁSICO 2023_CE_CP.xlsx'
# N° de filas que hay que saltarse al cargar la tabla (en qué fila se encuentran nombres de columnas)
n_filas_ignorar_tabla_insumos = 4
# Nombre columna con nombres de campos de la Base de datos:
nombre_col_campo_bd = 'Nombre Campo BD'
# Nombre columna con valores permitidos por subpregunta
nombre_col_val_permitidos = 'Rango de valores Permitidos'

## Conexión a NAS -----
IP_NAS = '10.10.100.28'
FOLDER_DATOS = '4b_2023' # OJO, actualizar

# 2. VARIABLES QUE ES PROBABLE QUE DEBAN SER ACTUALIZADAS:

# Variables identificadoras tabla origen
id_estudiante = 'serie'
variables_identificadoras = ['rbd', 'dvRbd', 'codigoCurso', id_estudiante, 'rutaImagen1']
# Expresión regular para extraer rl rbd de la ruta en variable RutaImagen
regex_extraer_rbd_de_ruta = r'\\(\d+)\\'

# Diccionario que indica si la pregunta 1 debe ser ignorada al procesar datos
dic_ignorar_p1 = {'estudiantes': True, 'padres': False}


# Expresión regular para capturar el identificador del estudiante en nombre de archivos
regex_estudiante = r'\d{7,}'

# Encoding para lectura de tablas
ENCODING = 'utf-8'

# Indica que si hay que quitar el identificador de alumno de la ruta en tabla origen
LIMPIAR_RUTA = False

# Determina si se ignora la primera página del cuadernillo
IGNORAR_PRIMERA_PAGINA = True



# 3. VARIABLES QUE ES POCO PROBABLE QUE HAYA QUE ACTUALIZAR:

# IMPORTANTE: parámetro que define cuántos píxeles espera como
# mínimo entre línea y línea de cada subpregunta
n_pixeles_entre_lineas = 22

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

def conectar_a_NAS(IP_NAS, FOLDER_DATOS):
    # Nos conectamos a disco NAS:
    if not Path('P:/').is_dir():
        os.system(rf"NET USE P: \\{IP_NAS}\{FOLDER_DATOS}")



# OBTENCIÓN DE INSUMOS ----------------------------------------
regex_hoja_cuadernillo = r'_(\d+)'


# Expresión regular que permite identificar variables asociadas a la pregunta 1
# Utilizado para obviarla de la selección de preguntas
regex_p1 = r'p1(_\d+)?$'


## 4. VARIABLES QUE PUEDEN SER IGNORADAS (vienen del entrenamiento):

# Semilla para componentes aleatorias del código:
SEED = 2024
# Porcentaje de casos de doble marca que extraemos de estudiantes
FRAC_SAMPLE = .05
# n° de rondas de aumentado de datos (máximo 5):
N_AUGMENT_ROUNDS = 5

# Nombre de tabla que contiene predicciones:
nombre_tabla_predicciones = 'data_pred.csv'