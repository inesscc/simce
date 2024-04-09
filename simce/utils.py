# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:54:35 2024

@author: jeconchao
"""


from pathlib import Path
from simce.config import data_dir


def crear_directorios():

        data_dir.mkdir(exist_ok=True)
        
        (data_dir / 'input').mkdir(exist_ok=True)
        (data_dir / 'output').mkdir(exist_ok=True)
        
        (data_dir / 'input/cuestionario_estudiantes').mkdir(exist_ok=True)
        (data_dir / 'input/cuestionario_padres').mkdir(exist_ok=True)
        
    
    
