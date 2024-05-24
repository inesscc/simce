# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:52:13 2024

@author: jeconchao
"""


import re
import cv2
import numpy as np
from simce.utils import get_mask_naranjo
from simce.config import regex_estudiante, n_preg_ignoradas_estudiantes, n_preg_ignoradas_padres
import simce.proc_imgs as proc


def get_n_paginas(directorio_imagenes):
    rbds = list(directorio_imagenes.iterdir())
    rbd1 = rbds[0]

    estudiantes_rbd = {re.search(regex_estudiante, str(i)).group(1)
                       for i in rbd1.rglob('*jpg')}
    n_files = len(list(rbd1.glob(f'{estudiantes_rbd.pop()}*')))
    n_pages = n_files * 2

    return n_pages


def get_n_preguntas(directorio_imagenes, tipo_cuadernillo):
    rbds = list(directorio_imagenes.iterdir())
    rbd1 = rbds[0]

    estudiantes_rbd = {re.search(regex_estudiante, str(i)).group(1)
                       for i in rbd1.rglob('*jpg')}

    total_imagenes = 0
    for file in (rbd1.glob(f'{estudiantes_rbd.pop()}*')):

        img_preg = cv2.imread(str(file), 1)

        img_crop = proc.recorte_imagen(img_preg, 0, 200, 50, 160)
        # Eliminamos franjas negras en caso de existir
        img_sin_franja = proc.eliminar_franjas_negras(img_crop)

        # Divimos imagen en dos páginas del cuadernillo
        img_p1, img_p2 = proc.partir_imagen_por_mitad(img_sin_franja)
        print(file)
        for p, media_img in enumerate([img_p1, img_p2]):

            mask_naranjo = get_mask_naranjo(media_img)

            # Find my contours
            big_contours = proc.get_contornos_grandes(mask_naranjo)

            total_imagenes += len(big_contours)

    # Quitamos del cálculo preguntas que son ignoradas:
    if tipo_cuadernillo == 'estudiantes':
        n_preg_ignoradas = n_preg_ignoradas_estudiantes
    elif tipo_cuadernillo == 'padres':
        n_preg_ignoradas = n_preg_ignoradas_padres

    return total_imagenes - n_preg_ignoradas  # Eliminamos 2 preguntas de portada
