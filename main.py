# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:34:05 2024

@author: jeconchao
"""

from simce.trabajar_rutas import crear_directorios
from simce.generar_insumos_img import generar_insumos_total
from simce.proc_imgs import get_subpreguntas
from simce.proc_tabla_99 import get_tablas_99_total
from simce.preparar_modelamiento import gen_train_test
from config.proc_img import N_AUGMENT_ROUNDS, FRAC_SAMPLE, CURSO
import config.proc_img as module_config
from config.parse_config import ConfigParser
from simce.utils import read_json

# import pandas as pd

# %% Subpreguntas

# %%


if __name__ == '__main__':
    # Define si estamos obteniendo datos para entrenamiento o predicción
    IS_TRAINING = True
    # 0. Creamos directorios
    config_dict = read_json('config/model.json')
    config = ConfigParser(config_dict)
    directorios = config.init_obj('directorios', module_config, curso=str(CURSO) )
    crear_directorios(directorios)
    # 1.  Generar insumos para procesamiento
    generar_insumos_total(directorios, CURSO) 
    # 2. Generar tablas con dobles marcas
    get_tablas_99_total(para_entrenamiento=IS_TRAINING, directorios=directorios)

    # 3. Recortar subpreguntas
    get_subpreguntas(tipo_cuadernillo='estudiantes', directorios=directorios, para_entrenamiento=IS_TRAINING)
    get_subpreguntas(tipo_cuadernillo='padres', directorios=directorios, para_entrenamiento=IS_TRAINING)

    if IS_TRAINING:
        #4. Obtener set de entrenamiento y test y aumentamos train
        gen_train_test(n_augment_rounds=N_AUGMENT_ROUNDS, fraccion_sample=FRAC_SAMPLE, directorios=directorios)

# %%
