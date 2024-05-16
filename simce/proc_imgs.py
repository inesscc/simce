# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:22:37 2024

@author: jeconchao
"""
import numpy as np
import cv2
from itertools import chain
from simce.config import dir_estudiantes, dir_output, regex_estudiante, dir_tabla_99, dir_input
from simce.errors import anotar_error
from itertools import islice
import pandas as pd
import re
from dotenv import load_dotenv
from simce.trabajar_rutas import get_n_paginas, get_n_preguntas
from os import environ
from pathlib import Path
load_dotenv()

VALID_INPUT = {'cuadernillo', 'pagina'}


def get_preg_por_hoja(nivel='cuadernillo'):

    if nivel not in VALID_INPUT:
        raise ValueError(f"nivel debe ser uno de los siguientes valores: {VALID_INPUT}")

    primer_est = re.search(
        regex_estudiante,
        # primer estudiante del primer rbd:
        str(next(next(dir_estudiantes.iterdir()).iterdir()))).group(1)
    if nivel == 'cuadernillo':
        dic = get_subpreguntas(filter_estudiante=primer_est,
                               nivel=nivel)
    elif nivel == 'pagina':
        dic = get_subpreguntas(filter_estudiante=primer_est, nivel=nivel)
    return dic


def get_baseline():
    rbds = set()
    paths = []

    for rbd in (islice(dir_estudiantes.iterdir(), 3)):
        print(rbd)
        paths.extend(list(rbd.iterdir()))
        rbds.update([rbd.name])

    get_subpreguntas(filter_rbd=rbds)

    rutas_output = [i for i in islice(dir_output.iterdir(), 3)]

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


if environ.get('ENVIRONMENT') == 'dev':
    n_pages = 12
    n_preguntas = 29
    n_subpreg_tot = 165
    subpreg_x_preg = {'p2': 12, 'p3': 6, 'p4': 10, 'p5': 6,
                      'p6': 7,                      'p7': 6,
                      'p8': 8,                      'p9': 5,
                      'p10': 8,                      'p11': 9,
                      'p12': 4,                      'p13': 4,
                      'p14': 7,                      'p15': 5,
                      'p16': 4,                      'p17': 4,
                      'p18': 6,                      'p19': 6,
                      'p20': 4,                      'p21': 4,
                      'p22': 4,                      'p23': 4,
                      'p24': 6,                      'p25': 11,
                      'p26': 6, 'p27': 4, 'p28': 2, 'p29': 3}

    dic_cuadernillo = {'p29': '1', 'p28': '1', 'p27': '1', 'p_': '1', 'p1': '2', 'p2': '2', 'p3': '2',
                       'p26': '2', 'p25': '2', 'p24': '3', 'p23': '3', 'p22': '3', 'p21': '3', 'p4': '3',
                       'p5': '3', 'p6': '4', 'p7': '4', 'p20': '4', 'p19': '4', 'p18': '4', 'p17': '5',
                       'p16': '5', 'p15': '5', 'p8': '5', 'p9': '5', 'p10': '6', 'p11': '6', 'p14': '6',
                       'p13': '6', 'p12': '6'}.pop('p_')
    dic_pagina = {'p29': 12, 'p28': 12, 'p27': 12, 'p_': 1, 'p1': 2, 'p2': 2, 'p3': 2, 'p26': 11,
                  'p25': 11, 'p24': 10, 'p23': 10, 'p22': 10, 'p21': 10, 'p4': 3, 'p5': 3, 'p6': 4,
                  'p7': 4, 'p20': 9, 'p19': 9, 'p18': 9, 'p17': 8, 'p16': 8, 'p15': 8, 'p8': 5, 'p9': 5,
                  'p10': 6, 'p11': 6, 'p14': 7, 'p13': 7, 'p12': 7}.pop('p_')
else:
    n_pages = get_n_paginas()
    n_preguntas = get_n_preguntas()
    subpreg_x_preg = get_baseline()
    dic_cuadernillo = get_preg_por_hoja(nivel='cuadernillo')
    dic_cuadernillo.pop('p_')
    dic_pagina = get_preg_por_hoja(nivel='pagina')
    dic_pagina.pop('p_')


def get_subpreguntas(filter_rbd=None, filter_estudiante=None,
                     filter_rbd_int=False, nivel=None):

    df99 = pd.read_csv(dir_tabla_99 / 'casos_99_compilados.csv')

    dir_preg99 = [dir_input / i for i in df99.ruta_imagen]

    # Si queremos correr función para rbd específico
    if filter_rbd:
        # Si queremos correr función desde un rbd en adelante
        if filter_rbd_int:
            directorios = [i for i in dir_preg99
                           if int(i.name) >= filter_rbd]
        # Si queremos correr función solo para el rbd ingresado
        else:
            if isinstance(filter_rbd, str):
                filter_rbd = [filter_rbd]
            directorios = [i for i in dir_preg99 if i.name in filter_rbd]
    else:
        directorios = dir_preg99

    # Permite armar diccionario con mapeo pregunta -> página cuadernillo (archivo input)
    if nivel:
        diccionario_nivel = dict()

    for num, rbd in enumerate(directorios):
        if not filter_estudiante:
            print('############################')
            print(rbd)
            print(num)
            print('############################')
        pregunta_selec, subpreg_selec = df99.iloc[num].preguntas.split('_')
        # estudiantes_rbd = {re.search(regex_estudiante, str(i)).group(1)
        #                    for i in rbd.iterdir()}

        # Si queremos correr función para un estudiante específico:
        # if filter_estudiante:
        #     if isinstance(filter_estudiante, str):
        #         filter_estudiante = [filter_estudiante]
        #     estudiantes_rbd = {
        #         i for i in estudiantes_rbd if i in filter_estudiante}

        estudiante = re.search(regex_estudiante, str(rbd)).group(1)

        # for estudiante in estudiantes_rbd:

        n_subpreg = 0

        # páginas del cuardenillo
        pages = (n_pages, 1)

        dic_q = {
            # pregunta inicial páginas bajas - 1
            'q1': 1 - 1,
            # pregunta inicial páginas altas + 1
            'q2': n_preguntas + 1}

        # Para cada imagen del cuadernillo de un estudiante (2 pág x img):
        for num_pag, pag in enumerate(rbd.parent.glob(f'{estudiante}*')):

            # Obtenemos páginas del cuadernillo actual:
            pages = get_current_pages_cuadernillo(num_pag, pages)

            # Si archivo no corresponde a la página del cuadernillo correcta, saltamos loop
            if dic_cuadernillo[pregunta_selec] != str(num_pag + 1):
                continue

            # Obtengo carpeta del rbd y archivo del estudiante a
            # partir del path:
            folder, file = (pag.parts[-2], pag.parts[-1])

            print(f'{file=}')
            print(f'{num_pag=}')

            # Creamos directorio si no existe
            (dir_output / f'{folder}').mkdir(exist_ok=True)

            # Leemos imagen
            img_preg = cv2.imread(str(pag), 1)
            # Eliminamos franjas negras en caso de existir
            img_preg2 = eliminar_franjas_negras(img_preg)

            # Recortamos info innecesaria de imagen
            img_crop = recorte_imagen(img_preg2, 0, 200, 50, 160)

            # Divimos imagen en dos páginas del cuadernillo
            img_p1, img_p2 = partir_imagen_por_mitad(img_crop)

            # Para cada una de las dos imágenes del cuadernillo
            for p, media_img in enumerate([img_p1, img_p2]):

                # Si página actual no corresponde a la página que nos interesa, saltamos loop
                if dic_pagina[pregunta_selec] != pages[p]:
                    continue

                # Detecto recuadros naranjos
                mask_naranjo = get_mask_naranjo(media_img)

                # Obtengo contornos
                big_contours = get_contornos_grandes(mask_naranjo, pages, p)

                # Para cada contorno de pregunta:
                for num_preg, c in enumerate(big_contours):

                    # Obtengo n° de pregunta en base a lógica de cuadernillo:
                    q, dic_q = calcular_pregunta_actual(pages, p, dic_q)

                    # Obtengo coordenadas de contornos y corto imagen
                    img_pregunta = bound_and_crop(media_img, c)

                    if nivel:
                        diccionario_nivel = poblar_diccionario_preguntas(q, diccionario_nivel,
                                                                         nivel=nivel,
                                                                         pag=pag, page=pages[p])

                    # exportamos preguntas válidas:
                    if q not in ['_', 1]:

                        try:
                            # Obtenemos subpreguntas:
                            img_pregunta_crop = recorte_imagen(
                                img_pregunta)
                            #  print(q)
                            img_crop_col = get_mask_naranjo(img_pregunta_crop,
                                                            lower_color=np.array(
                                                                [0, 114, 139]),
                                                            upper_color=np.array([23, 255, 255]))
                            # img_crop_col = proc.procesamiento_color(img_pregunta_crop)

                            puntoy = obtener_puntos(
                                img_crop_col, minLineLength=250)

                            n_subpreg += len(puntoy) - 1

                            for i in range(len(puntoy)-1):
                                try:

                                    crop_and_save_subpreg(img_pregunta_crop,
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

            if n_subpreg != n_subpreg_tot:

                preg_error = str(dir_output / f'{folder}/{estudiante}')

                dic_dif = get_subpregs_distintas(folder, estudiante)

                error = f'N° de subpreguntas incorrecto para estudiante {estudiante},\
    se encontraron {n_subpreg} subpreguntas {dic_dif}'

                anotar_error(
                    pregunta=preg_error, error=error)

    if nivel:
        return diccionario_nivel

    else:
        return 'Éxito!'


def get_subpregs_distintas(folder, estudiante):
    df = pd.DataFrame(
        [str(i) for i in (dir_output / f'{folder}/').iterdir() if estudiante in str(i)], columns=['ruta'])

    df['preg'] = df.ruta.str.extract(r'p(\d{1,2})').astype(int)
    df['subpreg'] = df.ruta.str.extract(r'p(\d{1,2}_\d{1,2})')

    df_resumen = pd.DataFrame(df.preg.value_counts().sort_values()
                              .sort_index().astype(int))

    df_resumen.index = 'p'+df_resumen.index.astype('string')

    df_resumen['baseline'] = subpreg_x_preg
    df_resumen = df_resumen.rename(columns={'count': 'origen'})
    dic_dif = df_resumen[df_resumen['origen'].ne(df_resumen.baseline)].to_dict()
    return dic_dif


def get_mask_naranjo(media_img, lower_color=np.array([13, 52, 0]), upper_color=np.array([29, 255, 255]),
                     iters=4):
    """
    Genera una máscara binaria para una imagen dada, basada en un rango de color en el espacio de color HSV.

    Args:
    media_img (np.ndarray): La imagen de entrada en formato BGR.
    lower_color (np.ndarray, optional): El límite inferior del rango de color en formato HSV. Por defecto es np.array([13, 31, 0]), que corresponde al color naranjo.
    upper_color (np.ndarray, optional): El límite superior del rango de color en formato HSV. Por defecto es np.array([29, 255, 255]), que corresponde al color naranjo.

    Returns:
    mask (numpy.ndarray): Una máscara binaria donde los píxeles de la imagen que están dentro del rango de color especificado son blancos, y todos los demás píxeles son negros.
    """
    # Convierte la imagen de entrada de BGR a HSV
    hsv = cv2.cvtColor(media_img, cv2.COLOR_BGR2HSV)

    # Crea una máscara binaria donde los píxeles de la imagen que están dentro del rango de color
    # especificado son blancos, y todos los demás píxeles son negros.
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=iters)

    # Calculamos la media de cada fila
    mean_row = mask.mean(axis=1)
    # Si la media es menor a 100, reemplazamos con 0 (negro):
    # Esto permite eliminar manchas de color que a veces se dan
    idx_low_rows = np.where(mean_row < 100)[0]
    mask[idx_low_rows, :] = 0

    return mask


def eliminar_franjas_negras(img_preg):
    im2 = get_mask_naranjo(img_preg, lower_color=np.array([0, 0, 241]),
                           upper_color=np.array([179, 255, 255]), iters=2)
    contours = cv2.findContours(
        im2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    area_cont = [cv2.contourArea(i) for i in contours]
    c = contours[area_cont.index(max(area_cont))]

    img_pregunta = bound_and_crop(img_preg, c)
    return img_pregunta


def recorte_imagen(img_preg, x0=110, x1=20, y0=50, y1=50):
    """Funcion para recortar margenes de las imagenes

    Args:
        img_preg (array imagen): _description_
        x0 (int, optional): _description_. Defaults to 130.
        x1 (int, optional): _description_. Defaults to 30.
        y0 (int, optional): _description_. Defaults to 50.
        y1 (int, optional): _description_. Defaults to 50.

    Returns:
        (array imagen): imagen cortada
    """

    x, y = img_preg.shape[:2]
    img_crop = img_preg[x0:x-x1, y0:y-y1]
    return img_crop


def procesamiento_color(img_crop):
    """
    Funcion que procesa el color de la imagen

    Args:
        img_crop (_type_): imagen recortada

    Returns:
        canny image
    """
    # transformando color
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    Canny = cv2.Canny(gray, 50, 150, apertureSize=3)

    return Canny


# Procesamiento sub-pregunta

def obtener_puntos(img_crop_canny, threshold=100, minLineLength=200):
    """
    Funcion que identifica lineas para obtener puntos en el eje "y" para realizar el recorte a subpreguntas

    Args:
        img_crop_canny (_type_): _description_

    Returns:
        lines: _description_
    """
    # obteniendo lineas
    lines = cv2.HoughLinesP(img_crop_canny, 1, np.pi/180,
                            threshold=threshold, minLineLength=minLineLength)

    indices_ordenados = np.argsort(lines[:, :, 1].flatten())
    lines_sorted = lines[indices_ordenados]

    puntoy = list(set(chain.from_iterable(lines_sorted[:, :, 1].tolist())))
    puntoy.append(img_crop_canny.shape[0])
    puntoy = sorted(puntoy)

    # print(puntoy)

    y = []
    for i in range(len(puntoy)-1):
        if puntoy[i+1] - puntoy[i] < 27:
            y.append(i+1)

    # print(puntoy)
    # print(y)

    for index in sorted(y, reverse=True):
        del puntoy[index]

    return puntoy


def bound_and_crop(img, c):

    # Obtengo coordenadas de contorno
    x, y, w, h = cv2.boundingRect(c)
    # Recorto imagen en base a contorno
    img_crop = img[y:y+h, x:x+w]
    return img_crop


def crop_and_save_subpreg(img_pregunta_crop, puntoy, i, dir_output, folder, estudiante, q):
    cropped_img_sub = img_pregunta_crop[puntoy[i]:
                                        puntoy[i+1],]

    # id_img = f'{page}_{n}'
    file_out = str(
        dir_output / f'{folder}/{estudiante}_p{q}_{i+1}.jpg')
    # print(file_out)
    cv2.imwrite(file_out, cropped_img_sub)


def get_pregunta_inicial_pagina(pages, p):

    if pages[p] > pages[1-p]:
        q_base = max([int(re.search(r'(\d+)', k).group(0))
                      for k, v in dic_pagina.items() if v == pages[p]])
        print(f'q max: {q_base}')

    # si es la pág más baja del cuardenillo
    elif (pages[p] < pages[1-p]) & (pages[p] != 1):
        q_base = min([int(re.search(r'(\d+)', k).group(0))
                      for k, v in dic_pagina.items() if v == pages[p]])
        print(f'q min: {q_base}')

    else:  # Para la portada
        q = '_'


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


def poblar_diccionario_preguntas(q, diccionario, nivel='cuadernillo',
                                 pag=None, page=None):

    if nivel == 'cuadernillo':
        print(pag)
        hoja_cuadernillo = re.search(r'_(\d+)', pag.name).group(1)
        diccionario[f'p{q}'] = hoja_cuadernillo
    elif nivel == 'pagina':
        diccionario[f'p{q}'] = page

    return diccionario


def partir_imagen_por_mitad(img_crop):
    # Buscamos punto medio de imagen para dividirla en las dos
    # páginas del cuadernillo
    punto_medio = int(np.round(img_crop.shape[1] / 2, 1))

    img_p1 = img_crop[:, :punto_medio]  # página izquierda
    img_p2 = img_crop[:, punto_medio:]  # página derecha

    return img_p1, img_p2


def get_current_pages_cuadernillo(num_pag, pages):

    if num_pag == 0:
        pass
    # si num_pag es par y no es la primera página
    elif (num_pag % 2 == 0) & (num_pag != 0):
        pages = (pages[1]-1, pages[0] + 1)
    elif num_pag % 2 == 1:
        pages = (pages[1]+1, pages[0] - 1)

    return pages


def get_contornos_grandes(mask, pages, p):

    # Obtengo contornos
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Me quedo contornos grandes
    big_contours = [
        i for i in contours if cv2.contourArea(i) > 30000]
    #  print([i[0][0][1] for i in big_contours] )

    #  print(f'página actual: {pages[p]}')

    if pages[p] < pages[1-p]:
        # revertimos orden de contornos cuando es la página baja
        # del cuadernillo
        big_contours = big_contours[::-1]

    return big_contours


def procesamiento_antiguo(media_img):
    '''Función en desuso. Procesaba imagen para detección de contornos'''

    gray = cv2.cvtColor(media_img, cv2.COLOR_BGR2GRAY)  # convert roi into gray
    # Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
    # Canny=cv2.Canny(Blur,10,50) #apply canny to roi
    _, It = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    sx = cv2.Sobel(It, cv2.CV_32F, 1, 0)
    sy = cv2.Sobel(It, cv2.CV_32F, 0, 1)
    m = cv2.magnitude(sx, sy)
    m = cv2.normalize(m, None, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U)
    m = cv2.ximgproc.thinning(m, None, cv2.ximgproc.THINNING_GUOHALL)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    m = cv2.dilate(m, kernel, iterations=2)
