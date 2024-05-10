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

    # Exportando tablas:
    casos_99.to_csv(dir_tabla_99 / 'CASOS99_FINAL.csv')
    casos_99_origen.to_csv(dir_tabla_99 / 'CASOS99_ORIGEN.csv')


# %%

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

    return casos_99
