# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:50:14 2024

@author: jeconchao
"""

from pathlib import Path


# Expresión regular para capturar el identificador del estudiante en nombre de archivos
regex_estudiante = r'(\d{7,})_.*jpg'

# IMPORTANTE: parámetro que define cuántos píxeles espera como
# mínimo entre línea y línea de cada subpregunta
n_pixeles_entre_lineas = 22

carpeta_estudiantes = 'CE'
carpeta_padres = 'CP'

dir_data = Path('data/')
dir_input = dir_data / 'input_raw'
dir_estudiantes = dir_input / carpeta_estudiantes
dir_padres = dir_input / carpeta_padres

dir_input_proc = Path('data/input_proc/')
dir_subpreg_aux = dir_input_proc / 'subpreg_recortadas'
dir_subpreg = dir_subpreg_aux / 'base'
dir_subpreg_aug = dir_subpreg_aux / 'augmented'


dir_tabla_99 = dir_input_proc / 'output_tabla_99'
dir_insumos = dir_input_proc / 'insumos'

dir_train_test = dir_data / 'input_modelamiento'

dir_output = Path('data/output')
dir_modelos = dir_output / 'modelos' 



# TRANSVERSALES-----------------------------------------------

# Determina si se ignora la primera página del cuadernillo
IGNORAR_PRIMERA_PAGINA = True

# Semilla para componentes aleatorias del código:
SEED = 2024

# OBTENCIÓN DE INSUMOS ----------------------------------------

regex_hoja_cuadernillo = r'_(\d+)'


# PROCESAMIENTO POSIBLES DOBLES MARCAS -----
nombre_tabla_estud_origen = f'{carpeta_estudiantes}_Origen_DobleMarca.csv'
nombre_tabla_estud_final = f'{carpeta_estudiantes}_Final_DobleMarca.csv'
nombre_tabla_padres_origen = f'{carpeta_padres}_Origen_DobleMarca.csv'
nombre_tabla_padres_final = f'{carpeta_padres}_Final_DobleMarca.csv'

# Variables identificadoras
id_estudiante = 'serie'
variables_identificadoras = ['rbd', 'dvRbd', 'codigoCurso', id_estudiante, 'rutaImagen1']
# Expresión regular para extraer rl rbd de la ruta en variable RutaImagen
regex_extraer_rbd_de_ruta = r'\\(\d+)\\'
# Expresión regular que permite identificar variables asociadas a la pregunta 1
# Utilizado para obviarla de la selección de preguntas
regex_p1 = r'p1(_\d+)?$'
# Diccionario que indica si la pregunta 1 debe ser ignorada al procesar datos
dic_ignorar_p1 = {'estudiantes': True, 'padres': False}
