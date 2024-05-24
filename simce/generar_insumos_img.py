# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:46:31 2024

@author: jeconchao
"""

import numpy as np
import cv2
from simce.config import dir_output, regex_estudiante, dir_insumos
from simce.errors import anotar_error
from simce.utils import timing
# from simce.apoyo_proc_imgs import get_subpreguntas_completo
from itertools import islice
import pandas as pd
import re

from simce.trabajar_rutas import get_n_paginas, get_n_preguntas
from simce.utils import get_mask_naranjo
import simce.proc_imgs as proc
import json


def calcular_pregunta_actual(pages, p, dic_q):
    '''Método programático para obtener pregunta del cuadernillo que se está
    procesando. Dado que siempre una página tiene preguntas que vienen en orden
    ascendente y la otra en orden descendente (por la lógica de cuadernillo), hubo
    que incorporar esto en el algoritmo

    Args:
        pages (tuple): tupla que contiene la página izquierda y la página derecha de la página del
        cuadernillo que se está procesando. Ejemplo: (10,3) para la página 2 del cuadernillo
        estudiantes 2023

        p (int): integer que toma valor 0 ó 1. Si es 0 es la primera página del cuadernillo, si es  1, es
        la segunda.

        dic_q (dict): diccionario que contiene dos llaves, q1 y q2. q1 es la pregunta actual desde el lado
        bajo y q2 es la pregunta actual desde el lado alto del cuadernillo.

    Returns:
        q: pregunta actual siendo procesada
        dic_q: diccionario actualizado con pregunta alta y pregunta baja

    '''

    # si es la pág + alta del cuadernillo:
    if pages[p] > pages[1-p]:
        dic_q['q2'] -= 1
        q = dic_q['q2']
    # si es la pág más baja del cuardenillo
    elif (pages[p] < pages[1-p]) & (pages[p] != 1):
        dic_q['q1'] += 1
        q = dic_q['q1']
    else:  # Para la portada
        q = 0

    return q, dic_q


def get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes, dic_pagina=None,
                              filter_rbd=None, filter_estudiante=None,
                              filter_rbd_int=False, nivel=None):
    '''
    Versión de función get_subpreguntas() diseñada para obtener todas las subpreguntas de cada alumno/a
    que recibe. Se utiliza principalmente para insumar los diccionarios automáticos, en particular, número
    de subpreguntas por pregunta, preguntas por página del cuadernillo y preguntas por imagen del
    cuadernillo. Función exporta imágenes para cada subpregunta de cada pregunta que procesa.

    Args:
        n_pages (int): n° de páginas que tiene el cuestionario

        n_pages (int): n° de preguntas que tiene el cuestionario

        directorio_imagenes (pathlib.Path): directorio desde el que se recogen imágenes a procesar


    Returns:
        None

    '''

    # TODO: eliminar parámetros de filtro de función
    # Si queremos correr función para rbd específico
    if filter_rbd:
        # Si queremos correr función desde un rbd en adelante
        if filter_rbd_int:
            directorios = [i for i in directorio_imagenes.iterdir()
                           if int(i.name) >= filter_rbd]
        # Si queremos correr función solo para el rbd ingresado
        else:
            if isinstance(filter_rbd, str):
                filter_rbd = [filter_rbd]
            directorios = [i for i in directorio_imagenes.iterdir() if i.name in filter_rbd]
    else:
        directorios = directorio_imagenes.iterdir()

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

            dir_output_rbd = (dir_output / f'{directorio_imagenes.name}/{rbd.name}/')
            dir_output_rbd.mkdir(parents=True, exist_ok=True)
            # Para cada imagen del cuadernillo de un estudiante (2 pág x img):
            for num_pag, pag in enumerate(rbd.glob(f'{estudiante}*')):
                # Creamos directorio para guardar imágenes

                # Obtenemos páginas del cuadernillo actual:
                pages = proc.get_current_pages_cuadernillo(num_pag, pages)

                # Obtengo carpeta del rbd y archivo del estudiante a
                # partir del path:
                file = pag.parts[-1]

                print(f'{file=}')
                print(f'{num_pag=}')

                # Creamos directorio si no existe

                # Leemos imagen
                img_preg = cv2.imread(str(pag), 1)
                img_crop = proc.recorte_imagen(img_preg, 0, 100, 50, 160)
                # Eliminamos franjas negras en caso de existir
                img_sin_franja = proc.eliminar_franjas_negras(img_crop)

                # Divimos imagen en dos páginas del cuadernillo
                img_p1, img_p2 = proc.partir_imagen_por_mitad(img_sin_franja)

                # Para cada una de las dos imágenes del cuadernillo
                for p, media_img in enumerate([img_p1, img_p2]):

                    # Detecto recuadros naranjos
                    mask_naranjo = get_mask_naranjo(media_img)

                    # Obtengo contornos
                    big_contours = proc.get_contornos_grandes(mask_naranjo)

                    if not nivel:
                        q_base = proc.get_pregunta_inicial_pagina(dic_pagina, pages, p)

                    # Para cada contorno de pregunta:
                    for num_preg, c in enumerate(big_contours):

                        # Obtengo coordenadas de contornos y corto imagen
                        img_pregunta = proc.bound_and_crop(media_img, c)

                        if nivel:
                            # Obtengo n° de pregunta en base a lógica de cuadernillo:
                            q, dic_q = calcular_pregunta_actual(pages, p, dic_q)
                        else:
                            q = q_base + num_preg

                        if nivel:
                            diccionario_nivel = proc.poblar_diccionario_preguntas(q, diccionario_nivel,
                                                                                  nivel=nivel,
                                                                                  pag=pag, page=pages[p])
                            continue

                        # exportamos preguntas válidas:
                        if q not in [0, 1]:

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

                                lineas_horizontales = proc.obtener_puntos(
                                    img_crop_col, minLineLength=250)

                                if lineas_horizontales is not None:

                                    n_subpreg += len(lineas_horizontales) - 1

                                    for i in range(len(lineas_horizontales)-1):
                                        try:

                                            file_out = str(
                                                dir_output_rbd / f'{estudiante}_p{q}_{i+1}.jpg')
                                            proc.crop_and_save_subpreg(img_pregunta_crop,
                                                                       lineas_horizontales,
                                                                       i, file_out)

                                        # Si hay error en procesamiento subpregunta
                                        except Exception as e:

                                            preg_error = str(
                                                dir_output_rbd / f'{estudiante}_p{q}_{i+1}')
                                            anotar_error(
                                                pregunta=preg_error,
                                                error='Subregunta no pudo ser procesada',
                                                e=e, i=i)

                                        continue

                                else:
                                    print('Pregunta no cuenta con subpreguntas, se guardará imagen')
                                    file_out = str(
                                        dir_output_rbd / f'{estudiante}_p{q}.jpg')
                                    cv2.imwrite(file_out, img_pregunta)

                                    # Si hay error en procesamiento pregunta
                            except Exception as e:

                                preg_error = str(
                                    dir_output_rbd / f'{estudiante}_p{q}')
                                anotar_error(
                                    pregunta=preg_error, error='Pregunta no pudo ser procesada', e=e)

                                continue

    if nivel:
        return diccionario_nivel

    else:
        return 'Éxito!'


def get_preg_por_hoja(n_pages, n_preguntas, directorio_imagenes, nivel='cuadernillo'):

    if nivel not in proc.VALID_INPUT:
        raise ValueError(f"nivel debe ser uno de los siguientes valores: {proc.VALID_INPUT}")

    primer_est = re.search(
        regex_estudiante,
        # primer estudiante del primer rbd:
        str(next(next(directorio_imagenes.iterdir()).iterdir()))).group(1)
    if nivel == 'cuadernillo':
        dic = get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
                                        filter_estudiante=primer_est,
                                        nivel=nivel)
    elif nivel == 'pagina':
        dic = get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
                                        filter_estudiante=primer_est, nivel=nivel)
    return dic


def get_baseline(n_pages, n_preguntas, directorio_imagenes, dic_pagina):
    rbds = set()
    paths = []

    for rbd in (islice(directorio_imagenes.iterdir(), 2)):
        print(rbd)
        paths.extend(list(rbd.iterdir()))
        rbds.update([rbd.name])

    get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
                              dic_pagina=dic_pagina, filter_rbd=rbds)

    dir_output_rbd = (dir_output / f'{directorio_imagenes.name}')
    rutas_output = [i for i in islice(dir_output_rbd.iterdir(), 2)]

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


def generar_insumos(tipo_cuadernillo):

    directorio_imagenes = proc.select_directorio(tipo_cuadernillo)

    n_pages = get_n_paginas(directorio_imagenes)
    n_preguntas = get_n_preguntas(directorio_imagenes, tipo_cuadernillo=tipo_cuadernillo)
    dic_cuadernillo = get_preg_por_hoja(n_pages, n_preguntas, directorio_imagenes, nivel='cuadernillo')
    dic_pagina = get_preg_por_hoja(n_pages, n_preguntas, directorio_imagenes, nivel='pagina')
    subpreg_x_preg = get_baseline(n_pages, n_preguntas, directorio_imagenes, dic_pagina)
    n_subpreg_tot = str(subpreg_x_preg.sum())

    insumos_tipo_cuadernillo = {'n_pages': n_pages,
                                'n_preguntas': n_preguntas,
                                'n_subpreg_tot': n_subpreg_tot,
                                'dic_cuadernillo': dic_cuadernillo,
                                'dic_pagina': dic_pagina,
                                'subpreg_x_preg': subpreg_x_preg.to_dict()}

    return insumos_tipo_cuadernillo


@timing
def generar_insumos_total():
    print('Generando insumos estudiantes...')

    insumos_est = generar_insumos(tipo_cuadernillo='estudiantes')
    print('Generando insumos padres...')

    insumos_padres = generar_insumos(tipo_cuadernillo='padres')

    insumos = {'estudiantes': insumos_est,
               'padres': insumos_padres}

    dir_insumos.mkdir(exist_ok=True)
    with open(dir_insumos / 'insumos.json', 'w') as fp:
        json.dump(insumos, fp)

    print('Insumos generados exitosamente!')


if __name__ == '__main__':
    generar_insumos_total()
