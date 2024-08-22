# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:20:37 2024

@author: jeconchao
"""
import pandas as pd
from config.proc_img import  variables_identificadoras, SEED, CURSO, ENCODING, LIMPIAR_RUTA, \
regex_extraer_rbd_de_ruta, dic_ignorar_p1, regex_p1, \
nombre_tabla_estud_origen, nombre_tabla_padres_origen, regex_estudiante 
from simce.utils import timing
import re
import json
import random
import numpy as np
from pathlib import Path
import os
random.seed(SEED)
np.random.seed(SEED)

@timing
def get_tablas_99_total(directorios:list[os.PathLike]):
    '''Obtiene y exporta tabla de dobles marcas para estudiantes y padres.
        **No retorna nada**
        
    Args:
        directorios: lista de directorios del proyecto.'''
    
    print('Generando tabla estudiantes...')

    get_tablas_99(tipo_cuadernillo='estudiantes', 
                  directorios=directorios)
    print('Generando tabla padres...')

    get_tablas_99(tipo_cuadernillo='padres', 
                  directorios=directorios)

def get_tablas_99(tipo_cuadernillo, directorios):
    '''a'''

    if tipo_cuadernillo == 'estudiantes':


        tabla_origen = nombre_tabla_estud_origen


    elif tipo_cuadernillo == 'padres':


        tabla_origen = nombre_tabla_padres_origen


    with open(directorios['dir_insumos'] / 'insumos.json') as f:
        insumos = json.load(f)



    if not tipo_cuadernillo in insumos.keys():
        print('No hay tabla de padres para este curso')
        return ''
    dic_cuadernillo = insumos[tipo_cuadernillo]['dic_cuadernillo']


    Origen_DobleMarca = pd.read_csv(directorios['dir_input'] / tabla_origen, delimiter=';', encoding=ENCODING)



    nombres_col = [i for i in Origen_DobleMarca.columns.to_list() if re.search(r'p\d+', i)]

    casos_99_origen = procesar_casos_99(Origen_DobleMarca, nombres_col, dic_cuadernillo,
                                        tipo_cuadernillo)
    
    casos_99_origen = casos_99_origen.rename(columns={'respuestas': 'dm_sospecha'})


    casos_99_origen['rbd_ruta'] = casos_99_origen.ruta_imagen.astype('string').str.extract(regex_extraer_rbd_de_ruta)

    df_final = casos_99_origen.reset_index()

    if LIMPIAR_RUTA:
        df_final.ruta_imagen = df_final.ruta_imagen.str.replace(r'\\\d{7}', '', regex=True, n=1)
        
    df_final['ruta_imagen_output'] = (directorios['dir_subpreg'] / 
                                      df_final.ruta_imagen.str.replace('\\', '/').str.replace('^/', '', regex=True)
                                      .apply(lambda x: Path(x).parent) /
                                        (df_final.serie.astype(str) + '_' + df_final.preguntas + '.jpg') )



    df_final.to_csv(
        directorios['dir_tabla_99'] / f'casos_99_compilados_{CURSO}_{tipo_cuadernillo}.csv', index=False)

    print('Tabla compilada generada exitosamente!')


def procesar_casos_99(df_rptas, nombres_col, dic_cuadernillo, tipo_cuadernillo):
    '''
    hola'''

    ignorar_p1 = dic_ignorar_p1[tipo_cuadernillo]

    df_melt = df_rptas.melt(id_vars=variables_identificadoras,
                            value_vars=nombres_col,
                            var_name='preguntas',
                            value_name='respuestas')
    # Si pregunta 1 debe ser ignorada, la sacamos de la base:
    if ignorar_p1:
        df_melt = df_melt[df_melt.preguntas.ne(regex_p1)]

    casos_99 = df_melt[(df_melt['respuestas'] == 99)].copy()

    # Si queremos obtener set de entrenamiento agregamos muestra de respuestas normales:
    # if para_entrenamiento:

    #     df_sample = df_melt[df_melt.respuestas.ne(99)].sample(round(casos_99.shape[0] * .2), random_state=SEED )
    #     casos_99 = pd.concat([casos_99, df_sample])

    # Usamos diccionario cuadernillo para ver a qué imagen está asociada esa pregunta específica:
    casos_99['ruta_imagen'] = (casos_99.rutaImagen1.str.replace(r'(_\d+.*)', '_', regex=True) +
                               casos_99.preguntas.str.extract(r'(p\d+)').squeeze().map(dic_cuadernillo) +
                               '.jpg')

    return casos_99.drop(columns=['rutaImagen1']).set_index(['serie', 'preguntas'])


def gen_tabla_entrenamiento(casos_99, casos_99_origen):
    '''hola'''

    casos_99_origen['dm_final'] = casos_99.respuestas
    casos_99_origen['dm_final'] = casos_99_origen['dm_final'].fillna(0).astype(int)
    casos_99_origen = casos_99_origen.rename(columns={'respuestas': 'dm_sospecha'})
    casos_99_origen.dm_final = (casos_99_origen.dm_final == 99).astype(int)
    casos_99_origen.dm_sospecha = (casos_99_origen.dm_sospecha == 99).astype(int)


    return casos_99_origen


# def procesar_casos_sample(df_rptas_fin, df_rptas_or, nombres_col, dic_cuadernillo, series):
#     """
#     Seleccion aleatoria de preguntas dados las series de los casos 99

#     """
#     # df fin
#     df_melt = df_rptas_fin.melt(id_vars=['rbd', 'dvRbd', 'codigoCurso', 'serie',
#                                          'rutaImagen1'],
#                                 value_vars=nombres_col,
#                                 var_name='preguntas',
#                                 value_name='respuestas')

#     # seleccionando aleatoriamente los cuestionarios segun num serie
#     n = len(series)
#     id_series = [random.randint(0, n) for _ in range(round(n*0.2))]
#     series = series[id_series]

#     df_melt = df_melt[(df_melt['respuestas'] != 99)
#                       & (df_melt['preguntas'].ne('p1'))
#                       & (df_melt['serie'].isin(series))
#                       ]

#     random_indices = np.random.randint(0, df_melt.shape[0], size=len(series))
#     df_melt = df_melt.iloc[random_indices]

#     df_melt['ruta_imagen'] = (df_melt.rutaImagen1.str.replace(r'(_\d+.*)', '_', regex=True) +
#                               df_melt.preguntas.str.extract(r'(p\d+)').squeeze().map(dic_cuadernillo) +
#                               '.jpg')

#     df_melt = df_melt.rename(columns={'respuestas': 'dm_final'})

#     # df origen
#     df_melt_or = df_rptas_or.melt(id_vars=['rbd', 'dvRbd', 'codigoCurso', 'serie',
#                                            'rutaImagen1'],
#                                   value_vars=nombres_col,
#                                   var_name='preguntas',
#                                   value_name='respuestas')[['serie', 'preguntas', 'respuestas']]

#     df_melt = df_melt.merge(df_melt_or, how='left', on=['serie', 'preguntas'])
#     df_melt = df_melt.rename(columns={'respuestas': 'dm_sospecha'})

#     return df_melt.drop(columns=['rutaImagen1']).set_index(['serie', 'preguntas'])
