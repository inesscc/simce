# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:34:05 2024

@author: jeconchao
"""

from simce.generar_insumos_img import generar_insumos_total
from simce.proc_imgs import get_subpreguntas
from simce.proc_tabla_99 import get_tablas_99_total
from simce.preparar_modelamiento import gen_pred_set
from config.proc_img import  get_directorios, CURSO
from simce.utils import crear_directorios

# import pandas as pd

# %% Subpreguntas

# %%


if __name__ == '__main__':

    # 0. Creamos directorios

    
    directorios = get_directorios()
    crear_directorios(directorios)
    # 1.  Generar insumos para procesamiento
    generar_insumos_total(directorios) 
    # 2. Generar tablas con dobles marcas
    get_tablas_99_total(directorios=directorios)

    # 3. Recortar subpreguntas
    get_subpreguntas(tipo_cuadernillo='estudiantes', directorios=directorios, curso=str(CURSO))
    #get_subpreguntas(tipo_cuadernillo='padres', directorios=directorios, curso=str(module_config.CURSO), para_entrenamiento=IS_TRAINING)


    #4. Obtener set de entrenamiento y test y aumentamos train
    #gen_train_test(n_augment_rounds=N_AUGMENT_ROUNDS, fraccion_sample=FRAC_SAMPLE, config=config)
    gen_pred_set(directorios, curso=CURSO)

# %%
