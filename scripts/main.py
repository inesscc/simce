# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:34:05 2024

@author: jeconchao
"""
from os import environ
from simce.proc_imgs import get_mask_naranjo, recorte_imagen

from simce.config import dir_estudiantes
from simce.utils import crear_directorios
from simce.trabajar_rutas import get_n_paginas, get_n_preguntas
from simce.errors import anotar_error
import cv2
from pathlib import Path
import re
import numpy as np
import simce.proc_imgs as proc
import pandas as pd
from dotenv import load_dotenv
from itertools import islice


load_dotenv()

# Creamos directorios
crear_directorios()

# %% Subpreguntas


def get_baseline():
    rbds = set()
    paths = []

    for rbd in (islice(dir_estudiantes.iterdir(), 3)):
        print(rbd)
        paths.extend(list(rbd.iterdir()))
        rbds.update([rbd.name])

    get_subpreguntas(filter_rbd=rbds)

    rutas_output = [i for i in islice(Path('data/output').iterdir(), 3)]

    rutas_output_total = []

    for ruta in rutas_output:
        rutas_output_total.extend(list(ruta.iterdir()))

    df = pd.DataFrame([str(i) for i in rutas_output_total], columns=['ruta'])

    df['est'] = df.ruta.str.extract(r'(\d{7})')
    df['preg'] = df.ruta.str.extract(r'p(\d{1,2})').astype(int)
    df['subpreg'] = df.ruta.str.extract(r'p(\d{1,2}_\d{1,2})')
    # n° mediano de subpreguntas por pregunta, de acuerdo a datos obtenidos de
    # alumnos en primeros 3 colegios
    df_resumen = (df.groupby(['est']).preg.value_counts()
                  .groupby('preg').median().sort_values()
                  .sort_index().astype(int))

    df_resumen.index = 'p'+df_resumen.index.astype('string')

    return df_resumen


def get_subpreguntas(filter_rbd=None, filter_estudiante=None,
                     filter_rbd_int=False):

    # Si queremos correr función para rbd específico
    if filter_rbd:
        # Si queremos correr función desde un rbd en adelante
        if filter_rbd_int:
            directorios = [i for i in dir_estudiantes.iterdir()
                           if int(i.name) >= filter_rbd]
        # Si queremos correr función solo para el rbd ingresado
        else:
            if isinstance(filter_rbd, str):
                filter_rbd = [filter_rbd]
            directorios = [i for i in dir_estudiantes.iterdir() if i.name in filter_rbd]
    else:
        directorios = dir_estudiantes.iterdir()

    for num, rbd in enumerate(directorios):
        if not filter_estudiante:
            print('############################')
            print(rbd)
            print(num)
            print('############################')

        estudiantes_rbd = {re.search(r'\d{7}', str(i)).group(0)
                           for i in rbd.iterdir()}
        # Si queremos correr función para un estudiante específico:
        if filter_estudiante:
            if isinstance(filter_estudiante, str):
                filter_estudiante = [filter_estudiante]
            estudiantes_rbd = {
                i for i in estudiantes_rbd if i in filter_estudiante}

        for estudiante in estudiantes_rbd:

            n_subpreg = 0

            # páginas del cuardenillo
            pages = (n_pages, 1)
            # pregunta inicial páginas bajas
            q1 = 0
            # pregunta inicial páginas altas
            q2 = n_preguntas + 1

            # Para cada imagen del cuadernillo de un estudiante (2 pág x img):
            for num_pag, pag in enumerate(rbd.glob(f'{estudiante}*')):

                # Obtengo carpeta del rbd y archivo del estudiante a
                # partir del path:
                folder, file = (pag.parts[-2], pag.parts[-1])

                print('file:', file)
                print(f'num_pag: {num_pag}')
                # print(pages)
                # Quitamos extensión al archivo
                # file_no_ext = Path(file).with_suffix('')
                # Creamos directorio si no existe
                Path(f'data/output/{folder}').mkdir(exist_ok=True)

                # Obtenemos página del archivo
                # page = re.search('\d+$',str(file_no_ext)).group(0)

                # Leemos imagen
                img_preg = cv2.imread(str(pag), 1)

                # Recortamos info innecesaria de imagen
                img_crop = recorte_imagen(img_preg, 0, 200, 50, 160)

                # Buscamos punto medio de imagen para dividirla en las dos
                # páginas del cuadernillo
                punto_medio = int(np.round(img_crop.shape[1] / 2, 1))

                img_p1 = img_crop[:, :punto_medio]  # página izquierda
                img_p2 = img_crop[:, punto_medio:]  # página derecha

                # Obtenemos páginas del cuadernillo actual:
                # si num_pag es par y no es la primera página
                if (num_pag % 2 == 0) & (num_pag != 0):
                    pages = (pages[1]-1, pages[0] + 1)
                elif num_pag % 2 == 1:
                    pages = (pages[1]+1, pages[0] - 1)
            #      print(pages)

                # Para cada una de las dos imágenes del cuadernillo
                for p, media_img in enumerate([img_p1, img_p2]):

                    # Detecto recuadros naranjos
                    m = get_mask_naranjo(media_img)

                    # Obtengo contornos
                    contours = cv2.findContours(
                        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
                    # Me quedo contornos grandes
                    big_contours = [
                        i for i in contours if cv2.contourArea(i) > 30000]
                    #  print([i[0][0][1] for i in big_contours] )

                    #  print(f'página actual: {pages[p]}')

                    if pages[p] < pages[1-p]:
                        # revertimos orden de contornos cuando es la página baja
                        # del cuadernillo
                        big_contours = big_contours[::-1]

                    # Agregamos conteo número de preguntas

                    for c in (big_contours):

                        # Obtengo coordenadas de contornos
                        x, y, w, h = cv2.boundingRect(c)
                        img_pregunta = media_img[y:y+h, x:x+w]

                        # Obtengo n° de pregunta en base a lógica de cuadernillo:
                        # si es la pág + alta del cuadernillo:
                        if pages[p] > pages[1-p]:
                            q2 -= 1
                            q = q2
                        # si es la pág más baja del cuardenillo
                        elif (pages[p] < pages[1-p]) & (pages[p] != 1):
                            q1 += 1
                            q = q1
                        else:  # Para la portada
                            q = '_'

                        # exportamos preguntas válidas:
                        if q not in ['_', 1]:

                            try:
                                # Obtenemos subpreguntas:
                                img_pregunta_crop = proc.recorte_imagen(
                                    img_pregunta)
                                #  print(q)
                                img_crop_col = get_mask_naranjo(img_pregunta_crop,
                                                                lower_color=np.array(
                                                                    [0, 114, 139]),
                                                                upper_color=np.array([30, 255, 255]))
                                # img_crop_col = proc.procesamiento_color(img_pregunta_crop)

                                puntoy = proc.obtener_puntos(
                                    img_crop_col, minLineLength=250)

                                n_subpreg += len(puntoy) - 1

                                for i in range(len(puntoy)-1):
                                    try:
                                        #  print(i)
                                        cropped_img_sub = img_pregunta_crop[puntoy[i]:
                                                                            puntoy[i+1],]

                                        # id_img = f'{page}_{n}'
                                        file_out = f'data/output/{folder}/{estudiante}_p{q}_{i+1}.jpg'
                                        # print(file_out)
                                        cv2.imwrite(file_out, cropped_img_sub)

                                    except Exception as e:
                                        print(
                                            f'Ups, ocurrió un error al recortar la imagen \
                                            con subpregunta {i+1}')
                                        print(e)
                                        preg_error = f'data/output/{folder}/{estudiante}_p{q}_{i+1}'
                                        anotar_error(
                                            pregunta=preg_error,
                                            error='Subregunta no pudo ser procesada')

                                        continue

                            except Exception as e:

                                preg_error = f'data/output/{folder}/{estudiante}_p{q}'
                                anotar_error(
                                    pregunta=preg_error, error='Pregunta no pudo ser procesada')
                                print(
                                    f'Ups, ocurrió un error con la pregunta {preg_error}')
                                print(e)

                                continue

            if n_subpreg != n_subpreg_tot:
                error = f'N° de subpreguntas incorrecto para estudiante {estudiante},\
                    se encontraron {n_subpreg} subpreguntas'
                print(error)
                preg_error = f'data/output/{folder}/{estudiante}'

                anotar_error(
                    pregunta=preg_error, error=error)

    return revisar_pregunta


if environ.get('ENVIRONMENT') == 'dev':
    n_pages = 12
    n_preguntas = 29
    n_subpreg_tot = 165
    subpreg_x_preg = {'p2': 12,
                      'p3': 6,
                      'p4': 10,
                      'p5': 6,
                      'p6': 7,
                      'p7': 6,
                      'p8': 8,
                      'p9': 5,
                      'p10': 8,
                      'p11': 9,
                      'p12': 4,
                      'p13': 4,
                      'p14': 7,
                      'p15': 5,
                      'p16': 4,
                      'p17': 4,
                      'p18': 6,
                      'p19': 6,
                      'p20': 4,
                      'p21': 4,
                      'p22': 4,
                      'p23': 4,
                      'p24': 6,
                      'p25': 11,
                      'p26': 6,
                      'p27': 4,
                      'p28': 2,
                      'p29': 3}
else:
    n_pages = get_n_paginas()
    n_preguntas = get_n_preguntas()
    subpreg_x_preg = get_baseline()

revisar_pregunta = []


# %%


if __name__ == '__main__':
    # get_subpreguntas(filter_rbd='10121', filter_rbd_int=False)

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
    cv2.imshow("Detected Lines", cv2.resize(m, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
