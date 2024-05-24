# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:20:37 2024

@author: jeconchao
"""
import pandas as pd
from simce.config import dir_tabla_99, dir_input, dir_insumos, dir_tabla_sample
import re
import json
import random
import numpy as np

random.seed(2024)
np.random.seed(2024)


def get_tablas_99():

    with open(dir_insumos / 'insumos.json') as f:
        insumos = json.load(f)
    dic_cuadernillo = insumos['dic_cuadernillo']

    CE_Final_DobleMarca = pd.read_csv(dir_input / 'CE_Final_DobleMarca.csv', delimiter=';')
    CE_Origen_DobleMarca = pd.read_csv(dir_input / 'CE_Origen_DobleMarca.csv', delimiter=';')

    nombres_col = [i for i in CE_Final_DobleMarca.columns.to_list() if re.search(r'p\d', i)]

    casos_99 = procesar_casos_99(CE_Final_DobleMarca, nombres_col, dic_cuadernillo)
    casos_99_origen = procesar_casos_99(CE_Origen_DobleMarca, nombres_col, dic_cuadernillo)

    df_final = gen_tabla_entrenamiento(casos_99, casos_99_origen)

    # Exportando tablas:
    df_final.to_csv(dir_tabla_99 / 'casos_99_compilados.csv')

    print('Tabla compilada generada exitosamente!')


def procesar_casos_99(df_rptas, nombres_col, dic_cuadernillo):
    df_melt = df_rptas.melt(id_vars=['rbd', 'dvRbd', 'codigoCurso', 'serie',
                                     'rutaImagen1'],
                            value_vars=nombres_col,
                            var_name='preguntas',
                            value_name='respuestas')

    casos_99 = df_melt[(df_melt['respuestas'] == 99) & (df_melt.preguntas.ne('p1'))].copy()
    casos_99['ruta_imagen'] = (casos_99.rutaImagen1.str.replace(r'(_\d+.*)', '_', regex=True) +
                               casos_99.preguntas.str.extract(r'(p\d+)').squeeze().map(dic_cuadernillo) +
                               '.jpg')

    return casos_99.drop(columns=['rutaImagen1']).set_index(['serie', 'preguntas'])


def gen_tabla_entrenamiento(casos_99, casos_99_origen):

    casos_99_origen['dm_final'] = casos_99.respuestas
    casos_99_origen['dm_final'] = casos_99_origen['dm_final'].fillna(0).astype(int)
    casos_99_origen = casos_99_origen.rename(columns={'respuestas': 'dm_sospecha'})

    return casos_99_origen


def procesar_casos_sample(df_rptas_fin, df_rptas_or, nombres_col, dic_cuadernillo, series):
    """
    Seleccion aleatoria de preguntas dados las series de los casos 99
    
    """
    ## df fin
    df_melt = df_rptas_fin.melt(id_vars=['rbd', 'dvRbd', 'codigoCurso', 'serie',
                                     'rutaImagen1'],
                                value_vars=nombres_col,
                                var_name='preguntas',
                                value_name='respuestas')


    ## seleccionando aleatoriamente los cuestionarios segun num serie
    n = len(series)
    id_series = [random.randint(0,n) for _ in range(round(n*0.2))]
    series = series[id_series]

    df_melt = df_melt[(df_melt['respuestas'] != 99)
                    & (df_melt['preguntas'].ne('p1')) 
                    & (df_melt['serie'].isin(series))
                    ]
    
    random_indices = np.random.randint(0, df_melt.shape[0], size=len(series))
    df_melt = df_melt.iloc[random_indices]

    df_melt['ruta_imagen'] = (df_melt.rutaImagen1.str.replace(r'(_\d+.*)', '_', regex=True) +
                               df_melt.preguntas.str.extract(r'(p\d+)').squeeze().map(dic_cuadernillo) +
                               '.jpg')
    
    df_melt = df_melt.rename(columns={'respuestas': 'dm_final'})
    
    ### df origen
    df_melt_or = df_rptas_or.melt(id_vars=['rbd', 'dvRbd', 'codigoCurso', 'serie',
                                    'rutaImagen1'],
                               value_vars=nombres_col,
                               var_name='preguntas',
                               value_name='respuestas')[['serie', 'preguntas', 'respuestas']]
    
    df_melt = df_melt.merge(df_melt_or, how = 'left', on = ['serie', 'preguntas'])
    df_melt = df_melt.rename(columns={'respuestas': 'dm_sospecha'})

    return df_melt.drop(columns=['rutaImagen1']).set_index(['serie', 'preguntas'])


def get_tablas_out(sample_no99 = False):
    """Generador de tabla csv para extraer las preguntas de interes.

    Args:
        sample_no99 (bool): indicar si se desea una muestra de preguntas sin doble marca. Defaults to False.
    """

    with open(dir_insumos / 'insumos.json') as f:
        insumos = json.load(f)
    dic_cuadernillo = insumos['dic_cuadernillo']

    CE_Final_DobleMarca = pd.read_csv(dir_input / 'CE_Final_DobleMarca.csv', delimiter=';')
    CE_Origen_DobleMarca = pd.read_csv(dir_input / 'CE_Origen_DobleMarca.csv', delimiter=';')

    nombres_col = [i for i in CE_Final_DobleMarca.columns.to_list() if re.search(r'p\d', i)]

    casos_99 = procesar_casos_99(CE_Final_DobleMarca, nombres_col, dic_cuadernillo)
    casos_99_origen = procesar_casos_99(CE_Origen_DobleMarca, nombres_col, dic_cuadernillo)
    
    df_final = gen_tabla_entrenamiento(casos_99, casos_99_origen)
    
    ## sample no 99
    if sample_no99:
        series = casos_99.index.get_level_values('serie')  #  series de los casos 99 para extraer una muestra
        casos_sample = procesar_casos_sample(CE_Final_DobleMarca, CE_Origen_DobleMarca, nombres_col, dic_cuadernillo, series)        
        df_final = df_final._append(casos_sample)
        
        # Exportando tablas:
        df_final.to_csv(dir_tabla_sample / 'casos_99_sample_compilados.csv')
    else:
        # Exportando tablas:
        df_final.to_csv(dir_tabla_99 / 'casos_99_compilados.csv')
    
    print('Tabla compilada generada exitosamente!')
    #return df_final

#get_tablas_out(sample_no99=True)
