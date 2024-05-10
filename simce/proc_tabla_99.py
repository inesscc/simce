# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:20:37 2024

@author: jeconchao
"""
import pandas as pd

CE_Final_DobleMarca = pd.read_csv('data/input/CE_Final_DobleMarca.csv', delimiter=';')
CE_Origen_DobleMarca = pd.read_csv('data/input/CE_Origen_DobleMarca.csv', delimiter=';')
# %%

nombres_col = CE_Final_DobleMarca.columns.to_list()

df_nombres_col = pd.DataFrame({'nombres_col': nombres_col,
                               'id': [x.find('p') for x in nombres_col]})

df_nombres_col = df_nombres_col[df_nombres_col['id'] == 0][df_nombres_col['nombres_col'] != 'prueba']


# %%
