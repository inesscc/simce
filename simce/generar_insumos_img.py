# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:46:31 2024

@author: jeconchao
"""

import numpy as np
import cv2
from simce.config import dir_estudiantes, dir_output, regex_estudiante, dir_insumos
from simce.errors import anotar_error
# from simce.apoyo_proc_imgs import get_subpreguntas_completo
from itertools import islice
import pandas as pd
import re

from simce.trabajar_rutas import get_n_paginas, get_n_preguntas
from simce.utils import get_mask_naranjo
import simce.proc_imgs as proc
import json


def calcular_pregunta_actual(pages, p, dic_q):

    # si es la pág + alta del cuadernillo:
    if pages[p] > pages[1-p]:
        dic_q['q2'] -= 1
        q = dic_q['q2']
    # si es la pág más baja del cuardenillo
    elif (pages[p] < pages[1-p]) & (pages[p] != 1):
        dic_q['q1'] += 1
        q = dic_q['q1']
    else:  # Para la portada
        q = '_'

    return q, dic_q


def get_current_pages_cuadernillo(num_pag, pages):

    # si num_pag es par y no es la primera página
    if (num_pag % 2 == 0) & (num_pag != 0):
        pages = (pages[1]-1, pages[0] + 1)
    elif num_pag % 2 == 1:
        pages = (pages[1]+1, pages[0] - 1)

    return pages


def get_subpreguntas_completo(n_pages, n_preguntas, filter_rbd=None, filter_estudiante=None,
                              filter_rbd_int=False, nivel=None):

    #  df99 = pd.read_csv(dir_tabla_99 / 'casos_99_compilados.csv')

    #   dir_preg99 = [dir_input / Path(i) for i in df99.ruta_imagen]

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

    # Permite armar diccionario con mapeo pregunta -> página cuadernillo (archivo input)
    if nivel:
        diccionario_nivel = dict()

    for num, rbd in enumerate(directorios):
        if not filter_estudiante:
            print('############################')
            print(rbd)
            print(num)
            print('############################')

        estudiantes_rbd = {re.search(regex_estudiante, str(i)).group(1)
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

            dic_q = {
                # pregunta inicial páginas bajas
                'q1': 0,
                # pregunta inicial páginas altas
                'q2': n_preguntas + 1}

            # Para cada imagen del cuadernillo de un estudiante (2 pág x img):
            for num_pag, pag in enumerate(rbd.glob(f'{estudiante}*')):

                # Obtengo carpeta del rbd y archivo del estudiante a
                # partir del path:
                folder, file = (pag.parts[-2], pag.parts[-1])

                print(f'{file=}')
                print(f'{num_pag=}')

                # Creamos directorio si no existe
                (dir_output / f'{folder}').mkdir(exist_ok=True)

                # Leemos imagen y eliminamos franjas negras en caso de existir
                img_preg2 = proc.eliminar_franjas_negras(cv2.imread(str(pag), 1))

                # Recortamos info innecesaria de imagen
                img_crop = proc.recorte_imagen(img_preg2, 0, 200, 50, 160)

                # Divimos imagen en dos páginas del cuadernillo
                img_p1, img_p2 = proc.partir_imagen_por_mitad(img_crop)

                # Obtenemos páginas del cuadernillo actual:
                pages = get_current_pages_cuadernillo(num_pag, pages)

                # Para cada una de las dos imágenes del cuadernillo
                for p, media_img in enumerate([img_p1, img_p2]):

                    # Detecto recuadros naranjos
                    mask_naranjo = get_mask_naranjo(media_img)

                    # Obtengo contornos
                    big_contours = proc.get_contornos_grandes(mask_naranjo, pages, p)

                    # Para cada contorno de pregunta:
                    for c in (big_contours):

                        # Obtengo coordenadas de contornos y corto imagen
                        img_pregunta = proc.bound_and_crop(media_img, c)

                        # Obtengo n° de pregunta en base a lógica de cuadernillo:
                        q, dic_q = calcular_pregunta_actual(pages, p, dic_q)

                        if nivel:
                            diccionario_nivel = proc.poblar_diccionario_preguntas(q, diccionario_nivel,
                                                                                  nivel=nivel,
                                                                                  pag=pag, page=pages[p])

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
                                                                upper_color=np.array([23, 255, 255]))
                                # img_crop_col = proc.procesamiento_color(img_pregunta_crop)

                                puntoy = proc.obtener_puntos(
                                    img_crop_col, minLineLength=250)

                                n_subpreg += len(puntoy) - 1

                                for i in range(len(puntoy)-1):
                                    try:

                                        proc.crop_and_save_subpreg(img_pregunta_crop,
                                                                   puntoy, i, dir_output,
                                                                   folder, estudiante, q)

                                    # Si hay error en procesamiento subpregunta
                                    except Exception as e:

                                        preg_error = str(
                                            dir_output / f'{folder}/{estudiante}_p{q}_{i+1}')
                                        anotar_error(
                                            pregunta=preg_error,
                                            error='Subregunta no pudo ser procesada',
                                            e=e, i=i)

                                        continue
                            # Si hay error en procesamiento pregunta
                            except Exception as e:

                                preg_error = str(dir_output / f'{folder}/{estudiante}_p{q}')
                                anotar_error(
                                    pregunta=preg_error, error='Pregunta no pudo ser procesada', e=e)

                                continue

    if nivel:
        return diccionario_nivel

    else:
        return 'Éxito!'


def get_preg_por_hoja(n_pages, n_preguntas, nivel='cuadernillo'):

    if nivel not in proc.VALID_INPUT:
        raise ValueError(f"nivel debe ser uno de los siguientes valores: {proc.VALID_INPUT}")

    primer_est = re.search(
        regex_estudiante,
        # primer estudiante del primer rbd:
        str(next(next(dir_estudiantes.iterdir()).iterdir()))).group(1)
    if nivel == 'cuadernillo':
        dic = get_subpreguntas_completo(n_pages, n_preguntas, filter_estudiante=primer_est,
                                        nivel=nivel)
    elif nivel == 'pagina':
        dic = get_subpreguntas_completo(n_pages, n_preguntas, filter_estudiante=primer_est, nivel=nivel)
    return dic


def get_baseline(n_pages, n_preguntas):
    rbds = set()
    paths = []

    for rbd in (islice(dir_estudiantes.iterdir(), 2)):
        print(rbd)
        paths.extend(list(rbd.iterdir()))
        rbds.update([rbd.name])

    get_subpreguntas_completo(n_pages, n_preguntas, filter_rbd=rbds)

    rutas_output = [i for i in islice(dir_output.iterdir(), 2)]

    rutas_output_total = []

    for ruta in rutas_output:
        rutas_output_total.extend(list(ruta.iterdir()))

    df = pd.DataFrame([str(i) for i in rutas_output_total], columns=['ruta'])

    df['est'] = df.ruta.str.extract(regex_estudiante)
    df['preg'] = df.ruta.str.extract(r'p(\d{1,2})').astype(int)
    df['subpreg'] = df.ruta.str.extract(r'p(\d{1,2}_\d{1,2})')
    # n° mediano de subpreguntas por pregunta, de acuerdo a datos obtenidos de
    # alumnos en primeros 3 colegios
    df_resumen = (df.groupby(['est']).preg.value_counts()
                  .groupby('preg').median().sort_values()
                  .sort_index().astype(int))

    df_resumen.index = 'p'+df_resumen.index.astype('string')

    return df_resumen


def generar_insumos():
    n_pages = get_n_paginas()
    n_preguntas = get_n_preguntas()
    subpreg_x_preg = get_baseline(n_pages, n_preguntas)
    n_subpreg_tot = str(subpreg_x_preg.sum())
    dic_cuadernillo = get_preg_por_hoja(n_pages, n_preguntas, nivel='cuadernillo')
    dic_pagina = get_preg_por_hoja(n_pages, n_preguntas, nivel='pagina')

    insumos = {'n_pages': n_pages,
               'n_preguntas': n_preguntas,
               'n_subpreg_tot': n_subpreg_tot,
               'dic_cuadernillo': dic_cuadernillo,
               'dic_pagina': dic_pagina,
               'subpreg_x_preg': subpreg_x_preg.to_dict()}

    dir_insumos.mkdir(exist_ok=True)
    with open(dir_insumos / 'insumos.json', 'w') as fp:
        json.dump(insumos, fp)

    print('Insumos generados exitosamente!')


if __name__ == '__main__':
    generar_insumos()
