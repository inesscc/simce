# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:46:31 2024

@author: jeconchao
"""

import numpy as np
import cv2
from config.proc_img import regex_estudiante, regex_hoja_cuadernillo, nombre_col_campo_bd, \
             IGNORAR_PRIMERA_PAGINA, nombre_tabla_para_insumos, \
          n_filas_ignorar_tabla_insumos, nombre_col_val_permitidos
from simce.utils import timing
import pandas as pd
import re
from simce.utils import get_mask_imagen
import simce.proc_imgs as proc
import json
import os
import argparse

def get_n_paginas(directorio_imagenes: str)->int:
    '''Obtiene el n° de páginas totales del cuadernillo SIMCE. Para esto obtiene
    todos los archivos de un alumno y multiplica el total de archivos por 2.
    
    Args:
        directorio_imagenes: string que contiene el directorio en que se encuentran imágenes
            de cuadernillos.

    Returns:
        n_pages: n° de páginas totales del cuadernillo.
    '''
    rbds = list(directorio_imagenes.iterdir())
    rbd1 = rbds[0]

    estudiantes_rbd = {re.search(f'({regex_estudiante})', str(i)).group(1)
                       for i in rbd1.rglob('*jpg')}
    n_files = len(list(rbd1.glob(f'{estudiantes_rbd.pop()}*')))
    n_pages = n_files * 2

    return n_pages


def calcular_pregunta_actual(pages: tuple[int, int], p: int, dic_q: dict)-> int:
    '''Método programático para obtener pregunta del cuadernillo que se está
    procesando. Dado que siempre una página tiene preguntas que vienen en orden
    ascendente y la otra en orden descendente (por la lógica de cuadernillo), hubo
    que incorporar esto en el algoritmo

    Args:
        pages: tupla que contiene la página izquierda y la página derecha de la página del
                cuadernillo que se está procesando. Ejemplo: (10,3) para la página 2 del cuadernillo
                estudiantes 2023

        p: integer que toma valor 0 ó 1. Si es 0 es la primera página del cuadernillo, si es  1, es
            la segunda.

        dic_q: diccionario que contiene dos llaves, q_bajo y q_alto. q_bajo es la pregunta actual desde el lado
                bajo y q_alto es la pregunta actual desde el lado alto del cuadernillo.

    Returns:
        q: pregunta actual siendo procesada

    '''

    # si es la pág + alta del cuadernillo:
    if pages[p] > pages[1-p]:
        dic_q['q_alto'] -= 1
        return dic_q['q_alto']
    # si es la pág más baja del cuardenillo
    elif (pages[p] < pages[1-p]) & (pages[p] != 1):
        dic_q['q_bajo'] += 1
        return dic_q['q_bajo']
    else:  # Para la portada
        return 0


def generar_diccionarios_x_pagina(n_pages: int, n_preguntas:int, directorio_imagenes:os.PathLike,
                              nivel:str, args: list,
                              filter_rbd:None|list|str=None, filter_estudiante:None|list|str=None,
                             ignorar_primera_pagina:bool=True,
                             )->dict:
    '''
    Función similar a get_subpreguntas() en el módulo de [procesamiento de imágenes](proc_imgs.md) diseñada para obtener
    todas las preguntas en uno o más cuadernillos específicos.
    Se utiliza para insumar los diccionarios automáticos, en particular, preguntas por página del cuadernillo y 
    preguntas por imagen del cuadernillo. 

    Args:
        n_pages: n° de páginas que tiene el cuestionario

        n_preguntas: n° de preguntas que tiene el cuestionario

        directorio_imagenes: directorio desde el que se recogen imágenes a procesar

        nivel: variable que se utiliza cuando se generan insumos.

        filter_rbd: permite filtrar uno o más RBDs específicos y solo realizar la operación sobre estos.

        filter_estudiante: permite filtrar uno o más estudiantes específicos y solo realizar la operación sobre estos.

        ignorar_primera_pagina: booleano que indica si se debe ignorar la primera página a la hora de 
            generar los diccionarios automáticos (en general primera página contiene ejemplos que debemos ignorar.)

        

    Returns:
        diccionario_nivel: diccionario donde las llaves son cada una de las preguntas del cuadernillo y los 
            valores son el n° de página o imagen que le corresponde, según si nivel es "página" o "cuadernillo",
            respectivamente 

    '''


    # Si queremos correr función para rbd específico
    if filter_rbd:

        # Si queremos correr función solo para el rbd ingresado

        if isinstance(filter_rbd, str):
            filter_rbd = [filter_rbd]
        directorios = [i for i in directorio_imagenes.iterdir() if i.name in filter_rbd]
    else:
        directorios = directorio_imagenes.iterdir()

    # Permite armar diccionario con mapeo pregunta -> página cuadernillo (archivo input)
    diccionario_nivel = dict()

    for num, rbd in enumerate(directorios):
        if not filter_estudiante:
            
            print('############################')
            print(rbd)
            print(num)
            print('############################')

        estudiantes_rbd = {re.search(f'({regex_estudiante})', str(i)).group(1)
                           for i in rbd.iterdir()}

        # Si queremos correr función para un estudiante específico:
        if filter_estudiante:
            if isinstance(filter_estudiante, str):
                filter_estudiante = [filter_estudiante]
            estudiantes_rbd = {
                i for i in estudiantes_rbd if i in filter_estudiante}

        for estudiante in estudiantes_rbd:


            # páginas del cuardenillo
            pages = (n_pages, 1)

            dic_q = {
                # pregunta inicial páginas bajas
                'q_bajo': 0,
                # pregunta inicial páginas altas
                'q_alto': n_preguntas + 1}


            # Para cada imagen del cuadernillo de un estudiante (2 pág x img):
            for num_pag, dir_pag in enumerate(sorted(list(rbd.glob(f'{estudiante}*')))):
                # Creamos directorio para guardar imágenes

                # Obtenemos páginas del cuadernillo actual:
                pages = get_paginas_actuales_cuadernillo(num_pag, pages)

                # Obtengo carpeta del rbd y archivo del estudiante a
                # partir del path:
                file = dir_pag.parts[-1]
                if args.verbose:
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



                    # Para cada contorno de pregunta:
                    for pregunta in (big_contours):

                        # Obtengo n° de pregunta en base a lógica de cuadernillo:
                        q = calcular_pregunta_actual(pages, p, dic_q) # Ojo, función actualiza dic_q



                        diccionario_nivel = poblar_diccionario_preguntas(q, diccionario_nivel,
                                                                            nivel=nivel,
                                                                            dir_pag=dir_pag,
                                                                            page=pages[p])



    return diccionario_nivel



def get_paginas_actuales_cuadernillo(num_pag:int, paginas_anteriores: tuple[int, int])->tuple[int, int]:
    '''Método programático para obtener páginas del cuadernillo que se están
    procesando en la imagen actualmente abierta. Dado que siempre una página tiene preguntas
    que vienen en orden ascendente y la otra en orden descendente (por la lógica de cuadernillo), se incorpora
    esto en el algoritmo. Se actualiza en cada iteración del loop.

    Examples:
        >>> get_paginas_actuales_cuadernillo(3, (10, 3))
        (4, 9)

    Args:
        num_pag: número de imagen del cuadernillo que se está procesando. Parte en 0.

        paginas_anteriores: tupla que contiene páginas del cuadernillo en la iteración anterior.
            Ejemplo: (10,3) para la página 2 del cuadernillo estudiantes 2023


    Returns:
        paginas_actuales: tupla actualizada con páginas del cuadernillo siendo procesadas actualmente


    '''

    if num_pag == 0:
        paginas_actuales = paginas_anteriores
    # si num_pag es par y no es la primera página
    elif (num_pag % 2 == 0):
        paginas_actuales = (paginas_anteriores[1]-1, paginas_anteriores[0] + 1)
    elif num_pag % 2 == 1:
        paginas_actuales = (paginas_anteriores[1]+1, paginas_anteriores[0] - 1)

    return paginas_actuales


def poblar_diccionario_preguntas(q: int, dic_paginas:dict, nivel:str='cuadernillo',
                                 dir_pag:None|os.PathLike=None, page:int|None=None)->dict:
    '''Función va poblando un diccionario que, para cada pregunta del cuestionario, indica a qué página
    del cuadernillo pertenece o a qué imagen pertenece, si el nivel es página o cuadernillo,
    respectivamente.

    Por ejemplo, si usamos el diccionario de estudiantes 2023, buscamos el valor asociado a p21, nos dirá
    que esta se encuentra en la imagen 3 del cuadernillo (nivel cuadernillo) o en la página 10 del
    cuadernillo (nivel página).

    Args:
        q: pregunta siendo poblada actualmente en el diccionario. Por ejemplo, 2, 14.

        dic_paginas: diccionario siendo poblado, puede ser a nivel de cuadernillo (imagen) o de
            página del cuadernillo

        nivel: nivel del diccionario que estamos poblando: cuadernillo o página.

        dir_pag: directorio de imagen siendo procesada actualmente (solo se usa si es a
                                                                                   nivel cuadernillo)

        page: página del cuadernillo siendo procesada actualmente (solo se usa si es a nivel página)

    Returns:
        dic_paginas: diccionario actualizado con la pregunta q.


    '''

    if nivel == 'cuadernillo':
        # print(dir_pag)
        hoja_cuadernillo = re.search(regex_hoja_cuadernillo, dir_pag.name).group(1)
        dic_paginas[f'p{q}'] = hoja_cuadernillo
    elif nivel == 'pagina':
        dic_paginas[f'p{q}'] = page

    return dic_paginas


def get_preg_por_hoja(n_pages:int, n_preguntas:int,
                       directorio_imagenes:os.PathLike, args:argparse.Namespace,
                         nivel:str='cuadernillo'
                         )->dict:
    '''Función que puebla diccionario completo que mapea preguntas del cuestionario a su hoja o imagen
    correspondiente en el cuadernillo. Utiliza como insumo el número de páginas del cuadernillo y el n°
    de preguntas del cuestionario.


    Args:
        n_pages: número de páginas del cuadernillo

        n_preguntas: número de preguntas del cuadernillo

        directorio_imagenes: directorio donde se encuentran imágenes del tipo de
        cuadernillo que se está procesando (padres o estudiantes).

        nivel: indica si se está obteniendo diccionario a nivel cuadernillo o página.


    Returns:
        diccionario_nivel: diccionario donde las llaves son cada una de las preguntas del cuadernillo y los 
            valores son el n° de página o imagen que le corresponde, según si nivel es "página" o "cuadernillo",
            respectivamente.


    '''

    if nivel not in proc.VALID_INPUT:
        raise ValueError(f"nivel debe ser uno de los siguientes valores: {proc.VALID_INPUT}")

    primer_est = re.search(
        f'({regex_estudiante})',
        # primer estudiante del primer rbd:
        str(next(next(directorio_imagenes.iterdir()).iterdir()))).group(1)

    diccionario_nivel = generar_diccionarios_x_pagina(n_pages, n_preguntas, directorio_imagenes,
                                    filter_estudiante=primer_est, nivel=nivel, args=args,
                                    ignorar_primera_pagina=IGNORAR_PRIMERA_PAGINA)
    return diccionario_nivel




def get_subpreg_x_preg(df_preguntas: pd.DataFrame)-> dict:
    '''Función que puebla diccionario que mapea para cada pregunta, cuántas subpreguntas tiene asociada.


    Args:
        df_preguntas: DataFrame en que cada celda es una subpregunta para un SIMCE específico.


    Returns:
        subpreg_x_preg: diccionario donde las llaves son cada una de las preguntas del cuadernillo y los 
            valores son el n° de subpreguntas asociados a esa pregunta.


    '''
    
    df_preguntas['preg'] = df_preguntas[nombre_col_campo_bd].str.extract('^p(\d+)').astype(int)
    subpreg_x_preg = df_preguntas['preg'].value_counts().sort_index()
    subpreg_x_preg.index = 'p' + subpreg_x_preg.index.astype('string') 
    subpreg_x_preg = subpreg_x_preg.to_dict()
    return subpreg_x_preg


def get_recuadros_x_subpreg(value: str)->int:
    '''Extrae de las celdas el número de opciones posibles en cada pregunta. Se excluyen del string las opciones
        Vacío y doble marca, que no representan recuadros en la práctica.

    Examples:
        >>> get_recuadros_x_subpreg(
        "0: vacio
        1: Nada capaz
        2: Poco capaz
        3: Bastante capaz
        4: Muy capaz
        99: doble marca")
        4
    
    Args:
        value: string de la celda que contiene valores posibles de subpregunta, proveniente de Excel

    Returns:
        n_recuadros_x_subpreg: N° de recuadros en subpregunta para la que se está calculando. 

    '''
    list_valores = value.split('\n')

    n_recuadros_x_subpreg = len([i for i in list_valores if not re.search('(vac[ií]o)|99', i, re.IGNORECASE)])

    return n_recuadros_x_subpreg


def generar_insumos(tipo_cuadernillo:str, directorios:dict[str, os.PathLike],
                    args:argparse.Namespace)-> dict[str, str]:
    '''
    Función principal. Genera todos los insumos para un tipo de cuadernillo específico (estudiantes o padres), es decir:
        n° de páginas, n° de preguntas, n° de subpreguntas, imagen asociada a cada pregunta, página asociada a cada pregunta,
        n° de subpreguntas en cada pregunta, n° de recuadros en cada pregunta.

    Args:
        tipo_cuadernillo: toma valor estudiantes o padres según el cuadernillo al que se le generen los insumos

        directorios: diccionario que contiene directorios del proyecto.

    Returns:
        insumos_tipo_cuadernillo: json-like que contiene cada uno de los insumos para el cuadernillo ingresado.
    '''


    directorio_imagenes = directorios[f'dir_{tipo_cuadernillo}']

    # El nombre de la carpeta que refiere a padres o estudiantes es también el nombre de la hoja en el Excel
    sheet_name = directorio_imagenes.name

    df_para_insumos = pd.read_excel(directorios['dir_input'] / nombre_tabla_para_insumos,
                    skiprows=n_filas_ignorar_tabla_insumos, sheet_name=sheet_name)
    df_para_insumos = df_para_insumos[df_para_insumos[nombre_col_campo_bd].notnull()]
    df_preguntas = df_para_insumos[df_para_insumos[nombre_col_campo_bd].str.contains('p\d+')].copy()

    n_pages = get_n_paginas(directorio_imagenes)
    n_preguntas = df_para_insumos[nombre_col_campo_bd].str.extract('(p\d+)').nunique().iloc[0]
    dic_cuadernillo = get_preg_por_hoja(n_pages, n_preguntas,
                                         directorio_imagenes , args=args, nivel='cuadernillo')
    dic_pagina = get_preg_por_hoja(n_pages, n_preguntas,
                                    directorio_imagenes, args=args, nivel='pagina')
    subpreg_x_preg = get_subpreg_x_preg(df_preguntas)
    n_subpreg_tot = df_para_insumos[nombre_col_campo_bd].str.contains('^p\d').sum()
    n_recuadros_x_subpreg = (df_preguntas
                                .set_index('p'+df_preguntas.preg.astype('string'))[nombre_col_val_permitidos]
                                .apply(lambda x: get_recuadros_x_subpreg(x))
                                .to_dict())
    insumos_tipo_cuadernillo = {'n_pages': n_pages,
                                'n_preguntas': str(n_preguntas),
                                'n_subpreg_tot': str(n_subpreg_tot),
                                'dic_cuadernillo': dic_cuadernillo,
                                'dic_pagina': dic_pagina,
                                'subpreg_x_preg': subpreg_x_preg,
                                'n_recuadros_x_subpreg': n_recuadros_x_subpreg}

    return insumos_tipo_cuadernillo


@timing
def generar_insumos_total(directorios:dict[str, os.PathLike], args:argparse.Namespace):
    '''Corre [función que genera insumos](../generar_insumos_img#simce.generar_insumos_img.generar_insumos) para 
    estudiantes y padres y luego los exporta como json. **Solo retorna un print que confirma que datos se exportaron**.

    Args:
        directorios: diccionario que contiene directorios del proyecto.

    
    '''
    print('Generando insumos estudiantes...')

    insumos_est = generar_insumos(tipo_cuadernillo='estudiantes', directorios=directorios,
                                  args=args)
    print('Generando insumos padres...')

    insumos_padres = generar_insumos(tipo_cuadernillo='padres', directorios=directorios, args=args)

    insumos = {'estudiantes': insumos_est,
               'padres': insumos_padres}

    dir_insumos = directorios['dir_insumos']
    with open(dir_insumos / 'insumos.json', 'w') as fp:
        json.dump(insumos, fp)


    return print('Insumos generados exitosamente!')



# def get_n_preguntas(directorio_imagenes: str, ignorar_primera_pagina:bool=True)->int:
#     '''Obtiene el n° de páginas totales del cuadernillo SIMCE. Para esto obtiene
#     todos los archivos de un alumno y multiplica el total de archivos por 2.
    
#     Args:
#         directorio_imagenes: string que contiene el directorio en que se encuentran imágenes
#             de cuadernillos.
#         ignorar_primera_pagina: 
            
#     Returns:
#         n_pages: n° de páginas totales del cuadernillo.
#     '''

#     rbds = list(directorio_imagenes.iterdir())
#     rbd1 = rbds[0]

#     estudiantes_rbd = {re.search(f'({regex_estudiante})', str(i)).group(1)
#                        for i in rbd1.rglob('*jpg')}

#     total_imagenes = 0
#     for n, file in enumerate(sorted(list(rbd1.glob(f'{estudiantes_rbd.pop()}*')))):

#         img_preg = cv2.imread(str(file), 1)

#         img_crop = proc.recorte_imagen(img_preg, 0, 200, 50, 160)
#         # Eliminamos franjas negras en caso de existir
#         img_sin_franja = proc.eliminar_franjas_negras(img_crop)

#         # Divimos imagen en dos páginas del cuadernillo
#         img_p1, img_p2 = proc.partir_imagen_por_mitad(img_sin_franja)

#         for p, media_img in enumerate([img_p1, img_p2]):

#             # Importante, nos saltamos primera página, ya que no contiene preguntas
#             if n == 0 and p == 1 and ignorar_primera_pagina:
#                 continue

#             mask_naranjo = get_mask_imagen(media_img)

#             # Find my contours
#             big_contours = proc.get_contornos_grandes(mask_naranjo)

#             total_imagenes += len(big_contours)
#             print(total_imagenes)

#     return total_imagenes 


# def get_baseline(n_pages, n_preguntas, directorio_imagenes, dic_pagina, directorios):
#     rbds = set()
#     paths = []
#     dir_subpreg = directorios['dir_subpreg']

#     for rbd in (islice(directorio_imagenes.iterdir(), 2)):
#         print(rbd)
#         paths.extend(list(rbd.iterdir()))
#         rbds.update([rbd.name])

#     get_subpreguntas_completo(n_pages, n_preguntas, directorio_imagenes,
#                               dic_pagina=dic_pagina, filter_rbd=rbds,
#                               ignorar_primera_pagina=IGNORAR_PRIMERA_PAGINA,
#                               dir_subpreg=dir_subpreg)

#     dir_subpreg_rbd = (dir_subpreg / f'{directorio_imagenes.name}')
#     rutas_output = [dir_subpreg_rbd / i for i in rbds]

#     rutas_output_total = []

#     for ruta in rutas_output:
#         rutas_output_total.extend(list(ruta.iterdir()))

#     df = pd.DataFrame([str(i) for i in rutas_output_total], columns=['ruta'])

#     df['est'] = df.ruta.str.extract(f'({regex_estudiante})')
#     df['preg'] = df.ruta.str.extract(r'p(\d{1,2})').astype(int)
#     df['subpreg'] = df.ruta.str.extract(r'p(\d{1,2}_\d{1,2})')
#     # n° mediano de subpreguntas por pregunta, de acuerdo a datos obtenidos de
#     # alumnos en primeros 3 colegios
#     df_resumen = (df.groupby(['est']).preg.value_counts()
#                   .groupby('preg').median().sort_values()
#                   .sort_index().astype(int))

#     df_resumen.index = 'p'+df_resumen.index.astype('string')

#     # Eliminamos archivos creados para generar insumos
#     [rmtree(str(i)) for i in rutas_output]

#     return df_resumen

if __name__ == '__main__':
    generar_insumos_total()
