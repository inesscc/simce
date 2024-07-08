# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:46:31 2024

@author: jeconchao
"""

import numpy as np
import cv2
from config.proc_img import regex_estudiante, regex_hoja_cuadernillo, IGNORAR_PRIMERA_PAGINA
from simce.errors import anotar_error
from simce.utils import timing
from itertools import islice
import pandas as pd
import re

from simce.trabajar_rutas import get_n_paginas, get_n_preguntas
from simce.utils import get_mask_imagen
import simce.proc_imgs as proc
import json
from shutil import rmtree
import config.proc_img as module_config



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


def get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes, dir_subpreg, dic_pagina=None,
                              filter_rbd=None, filter_estudiante=None,
                              filter_rbd_int=False, nivel=None,
                              ignorar_primera_pagina=True):
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

            dir_subpreg_rbd = (dir_subpreg / f'{directorio_imagenes.name}/{rbd.name}/')
            dir_subpreg_rbd.mkdir(parents=True, exist_ok=True)
            # Para cada imagen del cuadernillo de un estudiante (2 pág x img):
            for num_pag, dir_pag in enumerate(sorted(list(rbd.glob(f'{estudiante}*')))):
                # Creamos directorio para guardar imágenes

                # Obtenemos páginas del cuadernillo actual:
                pages = get_current_pages_cuadernillo(num_pag, pages)

                # Obtengo carpeta del rbd y archivo del estudiante a
                # partir del path:
                file = dir_pag.parts[-1]

                print(f'{file=}')
                print(f'{num_pag=}')

                # Creamos directorio si no existe

                # Leemos imagen
                img_preg = cv2.imread(str(dir_pag), 1)
                img_crop = proc.recorte_imagen(img_preg, 0, 150, 50, 160)
                # Eliminamos franjas negras en caso de existir
                img_sin_franja = proc.eliminar_franjas_negras(img_crop)

                # Divimos imagen en dos páginas del cuadernillo
                img_p1, img_p2 = proc.partir_imagen_por_mitad(img_sin_franja)

                # Para cada una de las dos imágenes del cuadernillo
                for p, media_img in enumerate([img_p1, img_p2]):
                    if p == 1 and num_pag == 0 and ignorar_primera_pagina:
                        continue

                    # Detecto recuadros naranjos
                    mask_naranjo = get_mask_imagen(media_img)

                    # Obtengo contornos
                    big_contours = proc.get_contornos_grandes(mask_naranjo)

                    if not nivel:
                        q_base = proc.get_pregunta_inicial_pagina(dic_pagina, pages[p])

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
                            diccionario_nivel = poblar_diccionario_preguntas(q, diccionario_nivel,
                                                                             nivel=nivel,
                                                                             dir_pag=dir_pag,
                                                                             page=pages[p])
                            continue

                        try:
                            # Obtenemos subpreguntas:
                            img_pregunta_crop = proc.recorte_imagen(
                                img_pregunta)
                            #  print(q)

                            img_crop_col = proc.get_mascara_lineas_horizontales(img_pregunta_crop)
            
                            lineas_horizontales = proc.obtener_lineas_horizontales(
                                img_crop_col[:-10, :-10],
                                  minLineLength=np.round(img_crop_col.shape[1] * .6))

                            if lineas_horizontales is not None:

                                n_subpreg += len(lineas_horizontales) - 1

                                for i in range(len(lineas_horizontales)-1):
                                    try:

                                        file_out = str(
                                            dir_subpreg_rbd / f'{estudiante}_p{q}_{i+1}.jpg')
                                        proc.crop_and_save_subpreg(img_pregunta_crop[:-10, :-10],
                                                                   lineas_horizontales,
                                                                   i, file_out)

                                    # Si hay error en procesamiento subpregunta
                                    except Exception as e:

                                        preg_error = str(
                                            dir_subpreg_rbd / f'{estudiante}_p{q}_{i+1}')
                                        anotar_error(
                                            pregunta=preg_error,
                                            error='Subregunta no pudo ser procesada',
                                            e=e, i=i)

                                    continue

                            else:
                                print('Pregunta no cuenta con subpreguntas, se guardará imagen')
                                file_out = str(
                                    dir_subpreg_rbd / f'{estudiante}_p{q}.jpg')
                                cv2.imwrite(file_out, img_pregunta)

                                # Si hay error en procesamiento pregunta
                        except Exception as e:

                            preg_error = str(
                                dir_subpreg_rbd / f'{estudiante}_p{q}')
                            anotar_error(
                                pregunta=preg_error, error='Pregunta no pudo ser procesada', e=e)

                            continue

    if nivel:
        return diccionario_nivel

    else:
        return 'Éxito!'


def get_current_pages_cuadernillo(num_pag, pages):
    '''Método programático para obtener páginas del cuadernillo que se están
    procesando en la imagen actualmente abierta. Dado que siempre una página tiene preguntas
    que vienen en orden ascendente y la otra en orden descendente (por la lógica de cuadernillo), hubo
    que incorporar esto en el algoritmo. Se actualiza en cada iteración del loop

    Args:
        num_pag (int): número de imagen del cuadernillo que se está procesando. Parte en 0.

        pages (tuple): tupla que contiene páginas del cuadernillo en la iteración anterior.
        Ejemplo: (10,3) para la página 2 del cuadernillo estudiantes 2023


    Returns:
        pages (tuple): tupla actualizada con páginas del cuadernillo siendo procesadas actualmente


    '''

    if num_pag == 0:
        pass
    # si num_pag es par y no es la primera página
    elif (num_pag % 2 == 0):
        pages = (pages[1]-1, pages[0] + 1)
    elif num_pag % 2 == 1:
        pages = (pages[1]+1, pages[0] - 1)

    return pages


def poblar_diccionario_preguntas(q, diccionario, nivel='cuadernillo',
                                 dir_pag=None, page=None):
    '''Función va poblando un diccionario que, para cada pregunta del cuestionario, indica a qué página
    del cuadernillo pertenece o a qué imagen pertenece, si el nivel es página o cuadernillo,
    respectivamente.

    Por ejemplo, si usamos el diccionario de estudiantes 2023, buscamos el valor asociado a p21, nos dirá
    que esta se encuentra en la imagen 3 del cuadernillo (nivel cuadernillo) o en la página 10 del
    cuadernillo (nivel página).

    Args:

        q (int): pregunta siendo poblada actualmente en el diccionario. Por ejemplo, 2, 14.

        diccionario (dict): diccionario siendo poblado, puede ser a nivel de cuadernillo (imagen) o de
        página del cuadernillo

        nivel (str): nivel del diccionario que estamos poblando: cuadernillo o página.

        dir_pag (pathlib.Path): directorio de imagen siendo procesada actualmente (solo se usa si es a
                                                                                   nivel cuadernillo)

        page (int): página del cuadernillo siendo procesada actualmente (solo se usa si es a nivel página)

    Returns:
        diccionario (dict): diccionario actualizado con la pregunta q.


    '''

    if nivel == 'cuadernillo':
        print(dir_pag)
        hoja_cuadernillo = re.search(regex_hoja_cuadernillo, dir_pag.name).group(1)
        diccionario[f'p{q}'] = hoja_cuadernillo
    elif nivel == 'pagina':
        diccionario[f'p{q}'] = page

    return diccionario


def get_preg_por_hoja(n_pages, n_preguntas, directorio_imagenes, directorios, nivel='cuadernillo' ):
    '''Función que pobla diccionario completo que mapea preguntas del cuestionario a su hoja o imagen
    correspondiente en el cuadernillo. Utiliza como insumo el número de páginas del cuadernillo y el n°
    de preguntas del cuestionario.

    Args:
        n_pages (int): número de páginas del cuadernillo

        n_preguntas (int): número de preguntas del cuadernillo

        directorio_imagenes (pathlib.Path): directorio donde se encuentran imágenes del tipo de
        cuadernillo que se está procesando (padres o estudiantes).

        nivel (str): indica si se está obteniendo diccionario a nivel cuadernillo o página.


    Returns:
        dic (dict): tupla actualizada con páginas del cuadernillo siendo procesadas actualmente


    '''
    dir_subpreg = directorios['dir_subpreg']
    if nivel not in proc.VALID_INPUT:
        raise ValueError(f"nivel debe ser uno de los siguientes valores: {proc.VALID_INPUT}")

    primer_est = re.search(
        regex_estudiante,
        # primer estudiante del primer rbd:
        str(next(next(directorio_imagenes.iterdir()).iterdir()))).group(1)
    if nivel == 'cuadernillo':
        dic = get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
                                        filter_estudiante=primer_est,
                                        nivel=nivel, ignorar_primera_pagina=IGNORAR_PRIMERA_PAGINA,
                                        dir_subpreg=dir_subpreg)
    elif nivel == 'pagina':
        dic = get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
                                        filter_estudiante=primer_est, nivel=nivel,
                                        ignorar_primera_pagina=IGNORAR_PRIMERA_PAGINA,
                                        dir_subpreg=dir_subpreg)
    return dic


def get_baseline(n_pages, n_preguntas, directorio_imagenes, dic_pagina, directorios):
    rbds = set()
    paths = []
    dir_subpreg = directorios['dir_subpreg']

    for rbd in (islice(directorio_imagenes.iterdir(), 2)):
        print(rbd)
        paths.extend(list(rbd.iterdir()))
        rbds.update([rbd.name])

    get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
                              dic_pagina=dic_pagina, filter_rbd=rbds,
                              ignorar_primera_pagina=IGNORAR_PRIMERA_PAGINA,
                              dir_subpreg=dir_subpreg)

    dir_subpreg_rbd = (dir_subpreg / f'{directorio_imagenes.name}')
    rutas_output = [dir_subpreg_rbd / i for i in rbds]

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

    # Eliminamos archivos creados para generar insumos
    [rmtree(str(i)) for i in rutas_output]

    return df_resumen


def generar_insumos(tipo_cuadernillo, directorios):

    directorio_imagenes = proc.select_directorio(tipo_cuadernillo, directorios)

    n_pages = get_n_paginas(directorio_imagenes)
    n_preguntas = get_n_preguntas(directorio_imagenes, ignorar_primera_pagina=IGNORAR_PRIMERA_PAGINA)
    dic_cuadernillo = get_preg_por_hoja(n_pages, n_preguntas, directorio_imagenes, directorios, nivel='cuadernillo')
    dic_pagina = get_preg_por_hoja(n_pages, n_preguntas, directorio_imagenes, directorios, nivel='pagina')
    subpreg_x_preg = get_baseline(n_pages, n_preguntas, directorio_imagenes, dic_pagina, directorios)
    n_subpreg_tot = str(subpreg_x_preg.sum())

    insumos_tipo_cuadernillo = {'n_pages': n_pages,
                                'n_preguntas': n_preguntas,
                                'n_subpreg_tot': n_subpreg_tot,
                                'dic_cuadernillo': dic_cuadernillo,
                                'dic_pagina': dic_pagina,
                                'subpreg_x_preg': subpreg_x_preg.to_dict()}

    return insumos_tipo_cuadernillo


@timing
def generar_insumos_total(directorios, curso):
    print('Generando insumos estudiantes...')

    insumos_est = generar_insumos(tipo_cuadernillo='estudiantes', directorios=directorios)
    print('Generando insumos padres...')

    insumos_padres = generar_insumos(tipo_cuadernillo='padres', directorios=directorios)

    insumos = {'estudiantes': insumos_est,
               'padres': insumos_padres}

    dir_insumos = directorios['dir_insumos']
    with open(dir_insumos / 'insumos.json', 'w') as fp:
        json.dump(insumos, fp)

    print('Insumos generados exitosamente!')


if __name__ == '__main__':
    generar_insumos_total()
