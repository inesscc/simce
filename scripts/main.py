# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:34:05 2024

@author: jeconchao
"""

from simce.utils import crear_directorios
from simce.generar_insumos_img import generar_insumos_total
from simce.proc_imgs import get_subpreguntas
from simce.proc_tabla_99 import get_tablas_99_total
import cv2
from pathlib import Path
import re
import simce.proc_imgs as proc
import pandas as pd


# %% Subpreguntas

# %%


if __name__ == '__main__':
    # Define si estamos obteniendo datos para entrenamiento o predicción
    IS_TRAINING = True
    # 0. Creamos directorios
  #  crear_directorios()
    # 1.  Generar insumos para procesamiento
    generar_insumos_total()
    # 2. Generar tablas con dobles marcas
   # get_tablas_99_total(para_entrenamiento=IS_TRAINING)

    # 3. Recortar subpreguntas
    # get_subpreguntas(tipo_cuadernillo='estudiantes', para_entrenamiento=IS_TRAINING)
    get_subpreguntas(tipo_cuadernillo='padres', para_entrenamiento=IS_TRAINING, filter_estudiante=4008988)
