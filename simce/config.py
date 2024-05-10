# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:50:14 2024

@author: jeconchao
"""

from pathlib import Path

dir_data = Path('data/')

dir_estudiantes = Path('data/input/cuestionario_estudiantes')
dir_padres = Path('data/input/cuestionario_padres')

dir_output = Path('data/output')

regex_estudiante = r'(\d{7,})_.*jpg'
