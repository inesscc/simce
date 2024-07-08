# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:50:14 2024

@author: jeconchao
"""

from pathlib import Path

<<<<<<< HEAD
curso = Path('4b')



=======
CURSO = Path('4b')
>>>>>>> pruebas_8b
# Expresión regular para capturar el identificador del estudiante en nombre de archivos
regex_estudiante = r'(\d{7,})_.*jpg'

# IMPORTANTE: parámetro que define cuántos píxeles espera como
# mínimo entre línea y línea de cada subpregunta
n_pixeles_entre_lineas = 22

carpeta_estudiantes = 'CE'
carpeta_padres = 'CP'



def get_directorios(curso, filtro=None):
    dd = dict()
    dd['dir_data'] = Path('data/')
    dd['dir_input'] = dd['dir_data'] / 'input_raw' / curso
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

    
    
    if filtro:
        if isinstance(filtro, str):
            filtro = [filtro]
        
        dd = { k:v for k,v in dd.items() if k in filtro }

        if len(filtro) == 1:

            dd = list(dd.values())[0]

    return dd



# TRANSVERSALES-----------------------------------------------

# Determina si se ignora la primera página del cuadernillo
IGNORAR_PRIMERA_PAGINA = True

# Semilla para componentes aleatorias del código:
SEED = 2024

# Porcentaje de casos de doble marca que extraemos de estudiantes
FRAC_SAMPLE = .05

# n° de rondas de aumentado de datos (máximo 5):
N_AUGMENT_ROUNDS = 5
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
