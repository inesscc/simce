# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:22:37 2024

@author: jeconchao
"""
import numpy as np
import cv2
from itertools import chain
from config.proc_img import regex_estudiante, n_pixeles_entre_lineas 
from simce.errors import anotar_error
# from simce.apoyo_proc_imgs import get_subpreguntas_completo
import pandas as pd
import re
from dotenv import load_dotenv
from simce.utils import get_mask_imagen, eliminar_o_rellenar_manchas
import json
import os
load_dotenv()

VALID_INPUT = {'cuadernillo', 'pagina'}


def get_insumos(tipo_cuadernillo:str, dir_insumos:os.PathLike)-> tuple:
    '''Carga insumos obtenidos en el [módulo de generación de insumos](../generar_insumos_img)

    Args:
        tipo_cuadernillo: tipo de cuadernillo a revisar: "estudiantes" o "padres".

        dir_insumos: directorio en el que se encuentran datos de insumos.
    
    Returns:
        insumos_total: tupla que contiene cada uno de los insumos, es decir:
            n_pages, n_preguntas, subpreg_x_preg, dic_cuadernillo, dic_pagina, n_subpreg_tot.


    '''
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

    insumos_total = n_pages, n_preguntas, subpreg_x_preg, dic_cuadernillo, dic_pagina, n_subpreg_tot

    return insumos_total




def dejar_solo_recuadros_subpregunta(img_pregunta: np.ndarray)-> np.ndarray:
    ''' Toma imagen recortada de una pregunta completa y vuelve a recortarla, usando máscaras
        de forma que solo queden los recuadros de las respuestas de la pregunta, dejando fuera el texto.
    
        Args:
            img_pregunta: imagen de una pregunta completa.

        Returns:
            img_recuadros: imagen de recuadros de una pregunta completa
    
    '''
    img_pregunta_crop = img_pregunta[60:-30, 30:-30]
    mask_recuadro = get_mask_imagen(img_pregunta_crop,
                           lower_color=np.array([0, 0, 224]),
                           upper_color=np.array([179, 11, 255]),
                           eliminar_manchas=None, iters=0, revert=False)

    # Si existen columnas blancas, las eliminamos:
    mean_col = mask_recuadro.mean(axis=0)
    mask_recuadro[:, np.where(mean_col > 200)] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mask_dilate = cv2.dilate(mask_recuadro, kernel, iterations=2)

    # Eliminamos manchas verticales y horizontales:
    lim_vert = mask_recuadro.mean(axis=0).mean()
    morph_vert = eliminar_o_rellenar_manchas(mask_dilate, orientacion='vertical', 
                                             limite=lim_vert, rellenar=False )
    lim_hor = mask_recuadro.mean(axis=1).mean()
    morph_hor = eliminar_o_rellenar_manchas(morph_vert, orientacion='horizontal',
                                             limite=lim_hor, rellenar=False )

    nonzero = cv2.findNonZero(morph_hor)

    img_recuadros = bound_and_crop(img_pregunta_crop, nonzero, buffer=70)


    return img_recuadros



def get_mascara_lineas_horizontales(img_recuadros:np.ndarray)->np.ndarray:
    ''' Genera máscara que detecta líneas horizontales que separan subpreguntas, para una imagen de recuadros de una pregunta
    completa.
    
        Args:
            img_recuadros: imagen de recuadros de una pregunta completa.
        Returns:
            mask_lineas_horizontales: máscara que contiene detección de líneas horizontales entre subpreguntas.
    
    '''

    px_naranjo = get_mask_imagen(img_recuadros,
                                   lower_color=np.array(
                                       [0, 111, 109]),
                                   upper_color=np.array([18, 255, 255]),
                                   iters=1, revert=True)

    px_azul = get_mask_imagen(img_recuadros, 
                                   lower_color=np.array([0, 0, 0]),
                                     upper_color=np.array([114, 255, 255]),
                                     eliminar_manchas=None, iters=0)
    
    px_negro = get_mask_imagen(img_recuadros, 
                                lower_color=np.array([0, 0, 204]),
                                    upper_color=np.array([179, 255, 255]),
                                    eliminar_manchas=None, iters=0)
    
    idx_naranjo = np.where(px_naranjo == 0)
    idx_azul = np.where(px_azul == 0)
    idx_negro =  np.where(px_negro == 0)

    gray = cv2.cvtColor(img_recuadros, cv2.COLOR_BGR2GRAY)
    gray[idx_azul ] = 255
    gray[idx_negro] = 255
    gray[idx_naranjo ] = 0

    gray2 = gray.copy() 

    mean_value = np.mean(gray)    
    # Replace values above the mean with 255
    gray2[(gray2 > mean_value*.95) ] = 255

    # Replace values below the mean with 0
    gray2[(gray2 < mean_value*.95) ] = 0

    gray_limpio = eliminar_o_rellenar_manchas(gray2, 
                                                       orientacion='horizontal',
                                                         limite=100, rellenar=False)[:-10, :-10]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))

    gray_dilated = cv2.dilate(gray_limpio, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))

    gray_dilated2 = cv2.dilate(gray_dilated, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    gray_eroded = cv2.erode(gray_dilated2, kernel, iterations=2)


    gray_limpio2 = eliminar_o_rellenar_manchas(gray_eroded, 
                                                       orientacion='horizontal',
                                                         limite=140, rellenar=True)
    

    gray_limpio3 = eliminar_o_rellenar_manchas(gray_limpio2, 
                                                       orientacion='horizontal',
                                                         limite=220, rellenar=False)

    mask_lineas_horizontales = cv2.bitwise_not(gray_limpio3)

    

    return mask_lineas_horizontales


def save_pregunta_completa(img_recuadros:np.ndarray,
                            dir_subpreg_rbd:os.PathLike,
                            estudiante:str, pregunta_selec:str):
    '''Guarda pregunta completa en casos de preguntas que no tienen subpreguntas.
        **No retorna nada**

    Args:
        img_recuadros: imagen de recuadros de una pregunta completa.
        dir_subpreg_rbd: directorio donde se guarda imágenes de subpreguntas y preguntas
        estudiante: identificador del estudiante asociado a pregunta
        pregunta_selec: pregunta siendo exportada actualmente


    
    '''
    print('Pregunta no cuenta con subpreguntas, se guardará imagen')
    file_out = str(
        dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}.jpg')
    
    # Si la pregunta es más larga que ancha, la dejamos a lo ancho:
    if img_recuadros.shape[0] > img_recuadros.shape[1]:
        img_recuadros = cv2.rotate(img_recuadros, cv2.ROTATE_90_CLOCKWISE)

    #n_subpreg = 1
    cv2.imwrite(file_out, img_recuadros)
    

def get_subpreguntas(tipo_cuadernillo: str, directorios:list[os.PathLike],
                     curso:str='4b', filter_rbd:None|list[str]=None,
                     filter_estudiante:None|list[str]=None,
                     muestra:bool=False):
    '''
    Recorta cada una de las subpreguntas obtenidas en el [módulo de procesamiento de tablas de doble marca](../proc_tabla_99).
    Obtendrá las imágenes de todas las sospechas de doble marca de la tabla
    de origen. Función exporta imágenes para cada subpregunta de la tabla de predicción. **No retorna nada.**

    Args:
        tipo_cuadernillo: define si se está procesando para estudiantes o padres. Esto también
        se utiliza para definir las rutas a consultar

        directorios: list con directorios del proyecto

        curso: string que identifica a curso actualmente siendo procesado.

        filter_rbd: permite filtrar uno o más RBDs específicos y solo realizar la operación sobre estos.

        filter_estudiante: permite filtrar uno o más estudiantes específicos y solo realizar la operación sobre estos.


    '''
    # Obtenemos directorio de imágenes (padres o estudiantes)
    
    directorio_imagenes = directorios[f'dir_{tipo_cuadernillo}']
    dir_tabla_99 = directorios['dir_tabla_99']
    dir_input = directorios['dir_input']
    dir_subpreg = directorios['dir_subpreg']
    

    # Definimos tabla a utilizar para seleccionar subpreguntas

    nombre_tabla_casos99 = f'casos_99_compilados_{curso}_{tipo_cuadernillo}.csv'

    df99 = pd.read_csv(
        dir_tabla_99 / nombre_tabla_casos99, dtype={'rbd_ruta': 'string'}).sort_values('ruta_imagen')



    # Si queremos correr función para rbd específicos
    if filter_rbd:

        df99 = df99[(df99.rbd_ruta.isin(filter_rbd))]

    if filter_estudiante:
        if isinstance(filter_estudiante, int):
            filter_estudiante = [filter_estudiante]
        df99 = df99[df99.serie.isin(filter_estudiante)]
    df99.ruta_imagen = df99.ruta_imagen.str.replace('\\', '/')
    dir_preg99 = [dir_input / i for i in df99.ruta_imagen]

    n_pages, _, subpreg_x_preg, _, dic_pagina, _ = get_insumos(
        tipo_cuadernillo, dir_insumos=directorios['dir_insumos'])

    for num, rbd in enumerate(dir_preg99):

        pregunta_selec = re.search(r'p(\d{1,2})', df99.iloc[num].preguntas).group(0)

        print('############################')
        print(rbd)
        print(f'{num=}')
        print(f'{pregunta_selec=}')

        print('############################')
        print('\n')

        estudiante = re.search(f'({regex_estudiante})', str(rbd)).group(1)

        # páginas del cuardenillo
        pagina_pregunta = dic_pagina[pregunta_selec]

        pages = get_pages(pagina_pregunta, n_pages)

        dir_subpreg_rbd = (dir_subpreg / f'{directorio_imagenes.name}/{rbd.parent.name}')
        dir_subpreg_rbd.mkdir(exist_ok=True, parents=True)

        if not rbd.is_file():

            preg_error = dir_subpreg_rbd / f'{estudiante}'
            anotar_error(pregunta = str(preg_error),
                         error = f'No existen archivos disponibles para serie {preg_error.name}',
                         nivel_error = tipo_cuadernillo)
            continue

        # Para cada imagen del cuadernillo de un estudiante (2 pág x img):

        # Obtengo carpeta del rbd y archivo del estudiante a
        # partir del path:
        file = rbd.name

        print(f'{file=}')

        # Creamos directorio si no existe

        # Leemos imagen
        img_preg = cv2.imread(str(rbd), 1)
        img_crop = recorte_imagen(img_preg, 0, 150, 50, 160)
        # Eliminamos franjas negras en caso de existir
        img_sin_franja = eliminar_franjas_negras(img_crop)

        # Recortamos info innecesaria de imagen

        # Divimos imagen en dos páginas del cuadernillo
        paginas_cuadernillo = partir_imagen_por_mitad(img_sin_franja)

        # {k: v for k, v dic_pagina.items() if }

        # Seleccionamos página que nos interesa, basado en diccionario de páginas
        media_img = paginas_cuadernillo[pages.index(pagina_pregunta)]

        # Detecto recuadros naranjos
        mask_naranjo = get_mask_imagen(media_img)

        # Obtengo contornos
        big_contours = get_contornos_grandes(mask_naranjo)

        q_base = get_pregunta_inicial_pagina(dic_pagina, pagina_pregunta)
        pregunta_selec_int = int(re.search(r'\d+', pregunta_selec).group(0))
        try:
            # Obtengo coordenadas de contornos y corto imagen
            elemento_img_pregunta = big_contours[pregunta_selec_int - q_base]
            img_pregunta = bound_and_crop(media_img, elemento_img_pregunta)
            
            img_pregunta_recuadros = dejar_solo_recuadros_subpregunta(img_pregunta)

            # Exportamos pregunta si no tiene subpreguntas:
            if subpreg_x_preg[pregunta_selec] == 1:
                save_pregunta_completa(img_pregunta_recuadros, dir_subpreg_rbd, estudiante, pregunta_selec)
                continue


            subpreg_selec = df99.iloc[num].preguntas.split('_')[1]
            print(f'{subpreg_selec=}')
            # Obtenemos subpreguntas:


            mask_lineas_horizontales = get_mascara_lineas_horizontales(img_pregunta_recuadros)
            
            lineas_horizontales = obtener_lineas_horizontales(
                mask_lineas_horizontales, minLineLength=np.round(mask_lineas_horizontales.shape[1] * .6))

            n_subpreg = len(lineas_horizontales) - 1

            if n_subpreg != subpreg_x_preg[pregunta_selec]:

                preg_error = str(dir_subpreg_rbd / f'{estudiante}')

                dic_dif = get_subpregs_distintas(subpreg_x_preg, dir_subpreg_rbd, estudiante)

                error = f'N° de subpreguntas incorrecto para serie {estudiante},\
        se encontraron {n_subpreg} subpreguntas {dic_dif}'

                anotar_error(
                    pregunta=preg_error, error=error, nivel_error = tipo_cuadernillo)

            try:

                file_out = str(
                    dir_subpreg_rbd /
                    f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}.jpg')
                crop_and_save_subpreg(img_pregunta_recuadros, lineas_horizontales,
                                      i=int(subpreg_selec)-1, file_out=file_out)

            # Si hay error en procesamiento subpregunta
            except Exception as e:

                preg_error = str(
                    dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}')
                anotar_error(
                    pregunta=preg_error,
                    error='Subregunta no pudo ser procesada',
                    nivel_error='Subpregunta',
                    e=e)

                continue



                # Si hay error en procesamiento pregunta
        except Exception as e:
            print(e)

            preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}')
            anotar_error(
                pregunta=preg_error, error='Pregunta no pudo ser procesada', e=e,
                nivel_error='Pregunta')

            continue

    return 'Éxito!'


def get_pages(pagina_pregunta, n_pages):
    """"""

    pages_original = n_pages, 1
    pages = (pages_original[0] - (pagina_pregunta - 1), pages_original[1] + (pagina_pregunta - 1))
    if pagina_pregunta % 2 == 0:
        pages = pages[1], pages[0]

    return pages


def get_subpregs_distintas(subpreg_x_preg, dir_subpreg_rbd, estudiante):
    df = pd.DataFrame(
        [str(i) for i in dir_subpreg_rbd.iterdir() if estudiante in str(i)], columns=['ruta'])

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
    im2 = get_mask_imagen(img_preg, lower_color=np.array([0, 0, 241]),
                          upper_color=np.array([179, 255, 255]), iters=2)

    im2[:,:100] = 255
    im2[:,-100:] = 255
    
    # mean_row = im2.mean(axis=1)
    # idx_low_rows = np.where(mean_row == 0)[0]
    # im2[idx_low_rows, :] = 255

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

def obtener_lineas_horizontales(mask_lineas_rellenas, threshold=100, minLineLength=200):
    """
    Funcion que identifica lineas para obtener puntos en el eje "y" para realizar el recorte a
    subpreguntas

    Args:
        img_crop_canny (_type_): _description_

    Returns:
        lines: _description_
    """
    # obteniendo lineas


    lines = cv2.HoughLinesP(mask_lineas_rellenas, 1, np.pi/180,
                            threshold=threshold, minLineLength=minLineLength)
    


    if lines is not None:

        indices_ordenados = np.argsort(lines[:, :, 1].flatten())
        lines_sorted = lines[indices_ordenados]

        puntoy = list(set(chain.from_iterable(lines_sorted[:, :, 1].tolist())))
        puntoy.append(mask_lineas_rellenas.shape[0])
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


def bound_and_crop(img, c, buffer=0, buffer_extra_lados=0):

    # Obtengo coordenadas de contorno
    x, y, w, h = cv2.boundingRect(c)
    # Recorto imagen en base a contorno
    img_crop = img[max(y-buffer, 0):y+h+buffer, max(0, x-buffer-buffer_extra_lados):x+w+buffer]
    return img_crop


def crop_and_save_subpreg(img_pregunta_crop, lineas_horizontales, i, file_out, verbose=False):
    img_subrptas = img_pregunta_crop[lineas_horizontales[i]:
                                        lineas_horizontales[i+1]]
    print(file_out)
    print(img_subrptas.shape)
     # Si la subpregunta es más larga que ancha, la rotamos a lo ancho:
    if img_subrptas.shape[0] > img_subrptas.shape[1]:
        img_subrptas = cv2.rotate(img_subrptas, cv2.ROTATE_90_CLOCKWISE)

    # print(file_out)
    cv2.imwrite(file_out, img_subrptas)
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


def get_contornos_grandes(mask, limit_area=30000):

    # Obtengo contornos
    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    # Me quedo contornos grandes
    big_contours = [
        i for i in contours if cv2.contourArea(i) > limit_area]

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


# def select_directorio(tipo_cuadernillo:str, directorios:os.PathLike):
#     '''Selecciona directorio de datos según si se está procesando el cuadernillo
#     de padres o de estudiantes
    
#     Args:
#         tipo_cuadernillo: "estudiantes" o "padres"
#     '''

#     if tipo_cuadernillo == 'estudiantes':
#         directorio_imagenes = directorios['dir_estudiantes']

#     elif tipo_cuadernillo == 'padres':
#         directorio_imagenes = directorios['dir_padres']

#     return directorio_imagenes

# def borrar_texto_oscuro(gray, limite=200):
#     ''' Función que toma imagen recortada de una pregunta completa y vuelve a recortarla, usando máscaras
#         de forma que solo queden los recuadros de las respuestas de la pregunta, dejando fuera el texto.
    
#         Args:
#             img_pregunta: imagen de una pregunta completa.
#         Returns:
#             img_recuadro: imagen de recuadros de una pregunta completa
    
#     '''
#     gray_mean = gray.mean(axis=1) 
#     gray_mean_mul10 = np.append(gray_mean, [gray_mean.mean()]* (10 - gray.shape[0] % 10))
#     dark_areas = gray_mean_mul10.reshape(-1, 10).mean(axis=1)
#     pixeles_borrar = np.where(dark_areas< limite)[0] * 10

#     for p in pixeles_borrar:
#         gray[p:p+10] = 255
#     return gray