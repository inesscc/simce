# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:45:34 2024

@author: jeconchao
"""

import pandas as pd
df = pd.read_excel('problemas_datos (1).xlsx')
dic_cuadernillo = {'p29': '1', 'p28': '1', 'p27': '1', 'p2': '2', 'p3': '2', 'p26': '2', 'p25': '2',
                   'p24': '3', 'p23': '3', 'p22': '3', 'p21': '3', 'p4': '3', 'p5': '3', 'p6': '4',
                   'p7': '4', 'p20': '4', 'p19': '4', 'p18': '4', 'p17': '5', 'p16': '5', 'p15': '5',
                   'p8': '5', 'p9': '5', 'p10': '6', 'p11': '6', 'p14': '6', 'p13': '6', 'p12': '6'}
len(df.Pregunta.str.extract('(\d{7})').squeeze().unique())
df['rbd'] = df.Pregunta.str.extract('(\d{7})')
df['preg'] = df.Pregunta.str.extract('(p\d+)')
df['pag'] = df.preg.map(dic_cuadernillo)
df['rbd_pag'] = df.rbd + '_' + df.pag
df[['rbd_pag', 'rbd']].drop_duplicates().rbd.nunique()
df[df.rbd_pag.notnull()].rbd_pag.nunique()
otros_est = set(df[(df.rbd_pag.isnull())].rbd).difference(set(df[(df.rbd_pag.notnull())].rbd))
