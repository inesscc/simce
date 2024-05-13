# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:50:14 2024

@author: jeconchao
"""

from pathlib import Path

dir_data = Path('data/')
dir_input = dir_data / 'input'
dir_estudiantes = dir_input / 'CE'
dir_padres = dir_input / 'CP'

dir_output = Path('data/output/output_subpreg')
dir_tabla_99 = Path('data/output/output_tabla_99/')

regex_estudiante = r'(\d{7,})_.*jpg'
