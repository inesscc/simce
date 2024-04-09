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
        
    
    
