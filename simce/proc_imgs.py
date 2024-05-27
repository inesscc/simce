# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:22:37 2024

@author: jeconchao
"""
import numpy as np
import cv2
from itertools import chain
from simce.config import dir_output, regex_estudiante, dir_tabla_99, \
    dir_input, n_pixeles_entre_lineas, dir_estudiantes, dir_padres
from simce.errors import anotar_error
# from simce.apoyo_proc_imgs import get_subpreguntas_completo

import pandas as pd
import re
from dotenv import load_dotenv

from simce.utils import get_mask_naranjo
import json
from simce.config import dir_insumos
load_dotenv()

VALID_INPUT = {'cuadernillo', 'pagina'}


def get_insumos(tipo_cuadernillo):
    with open(dir_insumos / 'insumos.json') as f:
        insumos = json.load(f)

    # Seleccionamos insumos para el tipo de cuadernillo que estamos trabajando
    insumos_usar = insumos[tipo_cuadernillo]

    n_pages = insumos_usar['n_pages']
    n_preguntas = insumos_usar['n_preguntas']
    subpreg_x_preg = insumos_usar['subpreg_x_preg']
    dic_cuadernillo = insumos_usar['dic_cuadernillo']
    dic_pagina = insumos_usar['dic_pagina']
    n_subpreg_tot = insumos_usar['n_subpreg_tot']

    return n_pages, n_preguntas, subpreg_x_preg, dic_cuadernillo, dic_pagina, n_subpreg_tot


def select_directorio(tipo_cuadernillo):
    '''Selecciona directorio de datos según si se está procesando el cuadernillo
    de padres o de estudiantes'''

    if tipo_cuadernillo == 'estudiantes':
        directorio_imagenes = dir_estudiantes
    elif tipo_cuadernillo == 'padres':
        directorio_imagenes = dir_padres

    return directorio_imagenes


def get_subpreguntas(tipo_cuadernillo, para_entrenamiento=True, filter_rbd=None, filter_estudiante=None,
                     filter_rbd_int=False, muestra=False):
    '''
    Obtiene las cada una de las subpreguntas obtenidas de la función get_tablas_99(). Esto considera dos
    casos: si es para predicción obtendrá las imágenes de todas las sospechas de doble marca de la tabla
    de origen y si es para entrenamiento, además considerará aproximadamente un 20% extra de marcas
    normales (depende de variable para_entrenamiento. Variable global IS_TRAINING define esto).
    Función exporta imágenes para cada subpregunta de la tabla de entrenamiento o predicción.

    Args:
        tipo_cuadernillo (str): define si se está procesando para estudiantes o padres. Esto también
        se utiliza para definir las rutas a consultar

        para_entrenamiento (bool): define si el procesamiento se está realizando para generar una base de
        entrenamiento o de predicción


    Returns:
        None

    '''
    # Obtenemos directorio de imágenes (padres o estudiantes)
    directorio_imagenes = select_directorio(tipo_cuadernillo)

    # Definimos tabla a utilizar para seleccionar subpreguntas
    if para_entrenamiento:
        nombre_tabla_casos99 = f'casos_99_entrenamiento_compilados_{tipo_cuadernillo}.csv'
    else:
        nombre_tabla_casos99 = f'casos_99_compilados_{tipo_cuadernillo}.csv'

    df99 = pd.read_csv(
        dir_tabla_99 / nombre_tabla_casos99, dtype={'rbd_ruta': 'string'}).sort_values('ruta_imagen')

    if muestra:

        rbd_disp = {i.name for i in directorio_imagenes.iterdir()}
        df99 = df99[(df99.rbd_ruta.isin(rbd_disp))]

    # Si queremos correr función para rbd específico
    if filter_rbd:
        # Si queremos correr función desde un rbd en adelante
        if filter_rbd_int:
            df99 = df99[(df99.rbd_ruta.astype(int).ge(filter_rbd))]

        # Si queremos correr función solo para el rbd ingresado
        else:
            df99 = df99[(df99.rbd_ruta.eq(filter_rbd))]

    if filter_estudiante:
        df99 = df99[df99.serie.eq(filter_estudiante)]

    dir_preg99 = [dir_input / i for i in df99.ruta_imagen]

    n_pages, n_preguntas, subpreg_x_preg, dic_cuadernillo, dic_pagina, n_subpreg_tot = get_insumos(
        tipo_cuadernillo)

    for num, rbd in enumerate(dir_preg99):

        pregunta_selec = re.search(r'p(\d{1,2})', df99.iloc[num].preguntas).group(0)

        print('############################')
        print(rbd)
        print(f'{num=}')
        print(f'{pregunta_selec=}')

        print('############################')
        print('\n')

        estudiante = re.search(regex_estudiante, str(rbd)).group(1)

        # páginas del cuardenillo
        pagina_pregunta = dic_pagina[pregunta_selec]

        pages = get_pages(pagina_pregunta, n_pages)

        dir_output_rbd = (dir_output / f'{directorio_imagenes.name}/{rbd.parent.name}')
        dir_output_rbd.mkdir(exist_ok=True, parents=True)

        if not rbd.is_file():

            preg_error = dir_output_rbd / f'{estudiante}'
            anotar_error(pregunta=str(preg_error),
                         error=f'No existen archivos disponibles para estudiante serie {preg_error.name}',
                         nivel_error='Estudiante')
            continue

        # Para cada imagen del cuadernillo de un estudiante (2 pág x img):

        # Obtengo carpeta del rbd y archivo del estudiante a
        # partir del path:
        file = rbd.name

        print(f'{file=}')

        # Creamos directorio si no existe

        # Leemos imagen
        img_preg = cv2.imread(str(rbd), 1)
        img_crop = recorte_imagen(img_preg, 0, 200, 50, 160)
        # Eliminamos franjas negras en caso de existir
        img_sin_franja = eliminar_franjas_negras(img_crop)

        # Recortamos info innecesaria de imagen

        # Divimos imagen en dos páginas del cuadernillo
        paginas_cuadernillo = partir_imagen_por_mitad(img_sin_franja)

        # {k: v for k, v dic_pagina.items() if }

        # Seleccionamos página que nos interesa, basado en diccionario de páginas
        media_img = paginas_cuadernillo[pages.index(pagina_pregunta)]

        # Detecto recuadros naranjos
        mask_naranjo = get_mask_naranjo(media_img)

        # Obtengo contornos
        big_contours = get_contornos_grandes(mask_naranjo)

        q_base = get_pregunta_inicial_pagina(dic_pagina, pagina_pregunta)
        pregunta_selec_int = int(re.search(r'\d+', pregunta_selec).group(0))

        # Obtengo coordenadas de contornos y corto imagen
        elemento_img_pregunta = big_contours[pregunta_selec_int - q_base]
        img_pregunta = bound_and_crop(media_img, elemento_img_pregunta)

        try:

            if subpreg_x_preg[pregunta_selec] == 1:
                print('Pregunta no cuenta con subpreguntas, se guardará imagen')
                file_out = str(
                    dir_output_rbd / f'{estudiante}_{pregunta_selec}.jpg')
                n_subpreg = 1
                cv2.imwrite(file_out, img_pregunta)
                continue

        # exportamos preguntas válidas:

            subpreg_selec = df99.iloc[num].preguntas.split('_')[1]
            print(f'{subpreg_selec=}')
            # Obtenemos subpreguntas:
            img_pregunta_crop = recorte_imagen(
                img_pregunta)
            #  print(q)
            img_crop_col = get_mask_naranjo(img_pregunta_crop,
                                            lower_color=np.array(
                                                [0, 114, 139]),
                                            upper_color=np.array([23, 255, 255]))

            lineas_horizontales = obtener_puntos(
                img_crop_col, minLineLength=250)

            n_subpreg = len(lineas_horizontales) - 1

            try:

                file_out = str(
                    dir_output_rbd /
                    f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}.jpg')
                crop_and_save_subpreg(img_pregunta_crop, lineas_horizontales,
                                      i=int(subpreg_selec)-1, file_out=file_out)

            # Si hay error en procesamiento subpregunta
            except Exception as e:

                preg_error = str(
                    dir_output_rbd / f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}')
                anotar_error(
                    pregunta=preg_error,
                    error='Subregunta no pudo ser procesada',
                    nivel_error='Subpregunta',
                    e=e)

                continue

            if n_subpreg != subpreg_x_preg[pregunta_selec]:

                preg_error = str(dir_output_rbd / f'{estudiante}')

                dic_dif = get_subpregs_distintas(subpreg_x_preg, dir_output_rbd, estudiante)

                error = f'N° de subpreguntas incorrecto para estudiante {estudiante},\
        se encontraron {n_subpreg} subpreguntas {dic_dif}'

                anotar_error(
                    pregunta=preg_error, error=error, nivel_error='Estudiante')

                # Si hay error en procesamiento pregunta
        except Exception as e:

            preg_error = str(dir_output_rbd / f'{estudiante}_{pregunta_selec}')
            anotar_error(
                pregunta=preg_error, error='Pregunta no pudo ser procesada', e=e,
                nivel_error='Pregunta')

            continue

    return 'Éxito!'


def get_pages(pagina_pregunta, n_pages):
    pages_original = n_pages, 1
    pages = (pages_original[0] - (pagina_pregunta - 1), pages_original[1] + (pagina_pregunta - 1))
    if pagina_pregunta % 2 == 0:
        pages = pages[1], pages[0]

    return pages


def get_subpregs_distintas(subpreg_x_preg, dir_output_rbd, estudiante):
    df = pd.DataFrame(
        [str(i) for i in dir_output_rbd.iterdir() if estudiante in str(i)], columns=['ruta'])

    df['preg'] = df.ruta.str.extract(r'p(\d{1,2})').astype(int)
    df['subpreg'] = df.ruta.str.extract(r'p(\d{1,2}_\d{1,2})')

    df_resumen = pd.DataFrame(df.preg.value_counts().sort_values()
                              .sort_index().astype(int))

    df_resumen.index = 'p'+df_resumen.index.astype('string')

    df_resumen['baseline'] = subpreg_x_preg
    df_resumen = df_resumen.rename(columns={'count': 'origen'})
    dic_dif = df_resumen[df_resumen['origen'].ne(df_resumen.baseline)].to_dict()
    return dic_dif


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
    Funcion que identifica lineas para obtener puntos en el eje "y" para realizar el recorte a
    subpreguntas

    Args:
        img_crop_canny (_type_): _description_

    Returns:
        lines: _description_
    """
    # obteniendo lineas
    lines = cv2.HoughLinesP(img_crop_canny, 1, np.pi/180,
                            threshold=threshold, minLineLength=minLineLength)

    if lines is not None:

        indices_ordenados = np.argsort(lines[:, :, 1].flatten())
        lines_sorted = lines[indices_ordenados]

        puntoy = list(set(chain.from_iterable(lines_sorted[:, :, 1].tolist())))
        puntoy.append(img_crop_canny.shape[0])
        puntoy = sorted(puntoy)

        # print(puntoy)

        y = []
        for i in range(len(puntoy)-1):
            if puntoy[i+1] - puntoy[i] < n_pixeles_entre_lineas:
                y.append(i+1)

        # print(puntoy)
        # print(y)

        for index in sorted(y, reverse=True):
            del puntoy[index]

        return puntoy
    else:
        # Pregunta no cuenta con subpreguntas
        return None


def bound_and_crop(img, c):

    # Obtengo coordenadas de contorno
    x, y, w, h = cv2.boundingRect(c)
    # Recorto imagen en base a contorno
    img_crop = img[y:y+h, x:x+w]
    return img_crop


def crop_and_save_subpreg(img_pregunta_crop, lineas_horizontales, i, file_out, verbose=False):
    cropped_img_sub = img_pregunta_crop[lineas_horizontales[i]:
                                        lineas_horizontales[i+1],]

    # print(file_out)
    cv2.imwrite(file_out, cropped_img_sub)
    if verbose:
        print(f'{file_out} guardado!')


def get_pregunta_inicial_pagina(dic_pagina, pagina_pregunta):
    if pagina_pregunta != 1 and (pagina_pregunta in dic_pagina.values()):
        q_base = min([int(re.search(r'\d+', k).group(0))
                      for k, v in dic_pagina.items() if v == pagina_pregunta])

    else:  # Para la portada
        q_base = 0

    return q_base


def partir_imagen_por_mitad(img_crop):
    # Buscamos punto medio de imagen para dividirla en las dos
    # páginas del cuadernillo
    punto_medio = int(np.round(img_crop.shape[1] / 2, 1))

    img_p1 = img_crop[:, :punto_medio]  # página izquierda
    img_p2 = img_crop[:, punto_medio:]  # página derecha

    return img_p1, img_p2


def get_contornos_grandes(mask):

    # Obtengo contornos
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Me quedo contornos grandes
    big_contours = [
        i for i in contours if cv2.contourArea(i) > 30000]

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
