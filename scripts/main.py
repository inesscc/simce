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
    IS_TRAINING = True
    # 0. Creamos directorios
    crear_directorios()
    # 1.  Generar insumos para procesamiento
    generar_insumos_total()
    # 2. Generar tablas con dobles marcas
    get_tablas_99_total(para_entrenamiento=IS_TRAINING)

    # 3. Recortar subpreguntas
    get_subpreguntas(tipo_cuadernillo='estudiantes', para_entrenamiento=IS_TRAINING)
    get_subpreguntas(tipo_cuadernillo='padres', para_entrenamiento=IS_TRAINING)

    # a = get_subpreguntas(filter_estudiante='4279607')

    # %%
    folder = '09954'

    for folder in Path('data/output/').iterdir():

        s = pd.Series([re.match(r'\d+', i.name).group(0) for i in folder.iterdir()])
        s2 = pd.Series([re.search(r'p\d{1,2}', i.name).group(0)
                       for i in folder.iterdir()])
        s3 = pd.Series(
            [re.search(r'p\d{1,2}_\d{1,2}', i.name).group(0) for i in folder.iterdir()])
        df_check = pd.concat([s.rename('id_est'), s2.rename('preg'),
                              s3.rename('subpreg')], axis=1)

        n_est = df_check.id_est.nunique()
        subpregs = df_check.groupby('subpreg').id_est.count()

        df_check.groupby('id_est').preg.value_counts()

        nsubpreg_x_alumno = s.value_counts()

        if not nsubpreg_x_alumno[nsubpreg_x_alumno.ne(165)].empty:
            print(f'RBD {folder.name}:\n')
            print(nsubpreg_x_alumno[nsubpreg_x_alumno.ne(165)])
            print(subpregs[subpregs.ne(n_est)])
            print('\n')

    # %%

    e3 = Path('data/output')

    for n, i in enumerate(e3.rglob('*')):
        pass

    # %%

    cv2.imshow("Detected Lines", img_pregunta)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # %%

    hsv_img = cv2.cvtColor(mask_naranjo,  cv2.COLOR_GRAY2BGR)
    hsv_img = cv2.drawContours(mask_naranjo, big_contours, -1, (60, 200, 200), 3)
    cv2.imshow("Detected Lines", cv2.resize(mask_naranjo, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%

    big_contours
    img_pregunta = bound_and_crop(media_img, c)

# %%

    img_crop = proc.recorte_imagen(cropped_img)
    img_crop_col = proc.procesamiento_color(img_crop)

    puntoy = proc.obtener_puntos(img_crop_col)

    for i in range(len(puntoy)-1):
        print(i)
        cropped_img_sub = img_crop[puntoy[i]:puntoy[i+1],]

        cv2.imshow("Detected Lines", cropped_img_sub)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def apply_approx(cnt):

    #     epsilon = 0.45*cv2.arcLength(cnt,True)
    #     approx = cv2.approxPolyDP(cnt,epsilon,True)
    #     return approx

    # %%

    cv2.imshow("Detected Lines", cv2.resize(cropped_img, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
