# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:54:35 2024

@author: jeconchao
"""


from pathlib import Path
from simce.config import dir_data


def crear_directorios():

        dir_data.mkdir(exist_ok=True)
        
        (dir_data / 'input').mkdir(exist_ok=True)
        (dir_data / 'output').mkdir(exist_ok=True)
        
        (dir_data / 'input/cuestionario_estudiantes').mkdir(exist_ok=True)
        (dir_data / 'input/cuestionario_padres').mkdir(exist_ok=True)
        
    
    
dic_img_preg = {'1_0': 'p29',
                '1_1': 'p28',
                '1_2': 'p27',
                '1_3': '_',
                '1_4': '_',
                '2_0': 'p01',
                '2_1': 'p03',
                '2_2': 'p02',
                '2_3': 'p26',
                '2_4': 'p25',
                '3_0': 'p21',
                '3_1': 'p22',
                '3_2': 'p23',
                '3_3': 'p24',
                '3_4': 'p05',
                '3_5': 'p04',
                '4_0': 'p07',
                '4_1': 'p06',
                '4_2': 'p19',
                '4_3': 'p20',
                '4_4': 'p18',
                '5_0': 'p17',
                '5_1': 'p16',
                '5_2': 'p15',
                '5_3': 'p09',
                '5_4': 'p08',
                '6_0': 'p11',
                '6_1': 'p10',
                '6_2': 'p12',
                '6_3': 'p13',
                '6_4': 'p14',

                }