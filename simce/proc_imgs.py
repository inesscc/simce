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


def get_subpreguntas(tipo_cuadernillo, filter_rbd=None, filter_estudiante=None,
                     filter_rbd_int=False, nivel=None, muestra=False):

    directorio_imagenes = select_directorio(tipo_cuadernillo)

    df99 = pd.read_csv(
        dir_tabla_99 / f'casos_99_sample_compilados_{tipo_cuadernillo}.csv').sort_values('ruta_imagen')

    if muestra:
        df99['rbd'] = df99.ruta_imagen.str.extract(r'(\d{5})')

        rbd_disp = {i.name for i in directorio_imagenes.iterdir()}
        df99 = df99[(df99.rbd.isin(rbd_disp))]

    dir_preg99 = [dir_input / i for i in df99.ruta_imagen]

    n_pages, n_preguntas, subpreg_x_preg, dic_cuadernillo, dic_pagina, n_subpreg_tot = get_insumos(
        tipo_cuadernillo)

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

        pregunta_selec = re.search('p(\d{1,2})', df99.iloc[num].preguntas).group(0)

        if not filter_estudiante:
            print('############################')
            print(rbd)
            print(f'{num=}')
            print(f'{pregunta_selec=}')

            print('############################')
            print('\n')
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

        n_subpreg_acum = 0

        # páginas del cuardenillo
        pages = (n_pages, 1)

        dir_output_rbd = (dir_output / f'{directorio_imagenes.name}/{rbd.parent.name}')
        dir_output_rbd.mkdir(exist_ok=True, parents=True)

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

            # Leemos imagen
            img_preg = cv2.imread(str(pag), 1)
            img_crop = recorte_imagen(img_preg, 0, 200, 50, 160)
            # Eliminamos franjas negras en caso de existir
            img_sin_franja = eliminar_franjas_negras(img_crop)

            # Recortamos info innecesaria de imagen

            # Divimos imagen en dos páginas del cuadernillo
            img_p1, img_p2 = partir_imagen_por_mitad(img_sin_franja)

            # Obtenemos la pregunta desde la cual comienza la página

            # Para cada una de las dos imágenes del cuadernillo
            for p, media_img in enumerate([img_p1, img_p2]):

                # Si página actual no corresponde a la página que nos interesa, saltamos loop
                if dic_pagina[pregunta_selec] != pages[p]:
                    continue

                # Detecto recuadros naranjos
                mask_naranjo = get_mask_naranjo(media_img)

                # Obtengo contornos
                big_contours = get_contornos_grandes(mask_naranjo)

                q_base = get_pregunta_inicial_pagina(dic_pagina, pages, p)

                # Para cada contorno de pregunta:
                for num_preg, c in enumerate(big_contours):

                    q = q_base + num_preg

                    # Si la pregunta seleccionada no es la que nos interesa, seguimos
                    if f'p{q}' != pregunta_selec:
                        continue

                    # Obtengo coordenadas de contornos y corto imagen
                    img_pregunta = bound_and_crop(media_img, c)

                    # Si quiero poblar el diccionario de cuadernillo o página:
                    if nivel:
                        diccionario_nivel = poblar_diccionario_preguntas(q, diccionario_nivel,
                                                                         nivel=nivel,
                                                                         pag=pag, page=pages[p])
                        continue

                    if subpreg_x_preg[f'p{q}'] == 1:
                        print('Pregunta no cuenta con subpreguntas, se guardará imagen')
                        file_out = str(
                            dir_output_rbd / f'{estudiante}_p{q}.jpg')
                        n_subpreg_acum += 1
                        cv2.imwrite(file_out, img_pregunta)

                    # exportamos preguntas válidas:
                    elif q not in [0, 1]:

                        try:
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

                            n_subpreg_acum += n_subpreg

                            for i in range(n_subpreg):
                                try:

                                    if (i+1) != int(subpreg_selec):
                                        continue

                                    file_out = str(
                                        dir_output_rbd / f'{estudiante}_p{q}_{i+1}.jpg')
                                    crop_and_save_subpreg(img_pregunta_crop, lineas_horizontales,
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

                        # Si hay error en procesamiento pregunta
                        except Exception as e:

                            preg_error = str(anotar_error / f'{estudiante}_p{q}')
                            anotar_error(
                                pregunta=preg_error, error='Pregunta no pudo ser procesada', e=e)

                            continue

            if n_subpreg_acum != subpreg_x_preg[pregunta_selec]:

                preg_error = str(dir_output_rbd / f'{estudiante}')

                dic_dif = get_subpregs_distintas(subpreg_x_preg, folder, estudiante)

                error = f'N° de subpreguntas incorrecto para estudiante {estudiante},\
    se encontraron {n_subpreg_acum} subpreguntas {dic_dif}'

                anotar_error(
                    pregunta=preg_error, error=error)

    if nivel:
        return diccionario_nivel

    else:
        return 'Éxito!'


def get_subpregs_distintas(subpreg_x_preg, folder, estudiante):
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


def get_pregunta_inicial_pagina(dic_pagina, pages, p):
    if pages[p] != 1 and (pages[p] in dic_pagina.values()):
        q_base = min([int(re.search(r'(\d+)', k).group(0))
                      for k, v in dic_pagina.items() if v == pages[p]])

    else:  # Para la portada
        q_base = 0

    return q_base


def calcular_pregunta_actual(pages, p, q_base):

    # si es la pág + alta del cuadernillo:
    if pages[p] > pages[1-p]:
        q_base -= 1

    # si es la pág más baja del cuardenillo
    elif (pages[p] < pages[1-p]) & (pages[p] != 1):
        q_base += 1

    else:  # Para la portada
        q_base = 1

    return q_base


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


def get_contornos_grandes(mask):

    # Obtengo contornos
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Me quedo contornos grandes
    big_contours = [
        i for i in contours if cv2.contourArea(i) > 30000]
    #  print([i[0][0][1] for i in big_contours] )

    #  print(f'página actual: {pages[p]}')

    # Revertimos orden
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
