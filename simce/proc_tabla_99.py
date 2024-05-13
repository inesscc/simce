# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:20:37 2024

@author: jeconchao
"""
import pandas as pd
from simce.config import dir_tabla_99, dir_input
import re
from simce.proc_imgs import dic_cuadernillo


def get_tablas_99():
    CE_Final_DobleMarca = pd.read_csv(dir_input / 'CE_Final_DobleMarca.csv', delimiter=';')
    CE_Origen_DobleMarca = pd.read_csv(dir_input / 'CE_Origen_DobleMarca.csv', delimiter=';')

    nombres_col = [i for i in CE_Final_DobleMarca.columns.to_list() if re.search(r'p\d', i)]

    casos_99 = procesar_casos_99(CE_Final_DobleMarca, nombres_col, dic_cuadernillo)
    casos_99_origen = procesar_casos_99(CE_Origen_DobleMarca, nombres_col, dic_cuadernillo)

    df_final = gen_tabla_entrenamiento(casos_99, casos_99_origen)

    # Exportando tablas:
    df_final.to_csv(dir_tabla_99 / 'casos_99_compilados.csv')


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
