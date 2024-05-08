# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:52:13 2024

@author: jeconchao
"""


from simce.config import dir_estudiantes
import re
import cv2
import numpy as np

def get_n_paginas():
    rbds = list(dir_estudiantes.iterdir())
    rbd1 = rbds[0]

    estudiantes_rbd = {re.search(r'\d{7}', str(i)).group(0)
                       for i in rbd1.iterdir()}
    n_files = len(list(rbd1.glob(f'{estudiantes_rbd.pop()}*')))
    n_pages = n_files * 2

    return n_pages


def get_n_preguntas():
    rbds = list(dir_estudiantes.iterdir())
    rbd1 = rbds[0]

    estudiantes_rbd = {re.search(r'\d{7}', str(i)).group(0)
                       for i in rbd1.iterdir()}

    total_imagenes = 0
    for file in (rbd1.glob(f'{estudiantes_rbd.pop()}*')):

        img_preg = cv2.imread(str(file), 1)

        x, y = img_preg.shape[:2]
        img_crop = img_preg[40:x - 200, 50:y-160]

        punto_medio = int(np.round(img_crop.shape[1] / 2, 1))

        img_p1 = img_crop[:, :punto_medio]
        img_p2 = img_crop[:, punto_medio:]

        for p, media_img in enumerate([img_p1, img_p2]):

            gray = cv2.cvtColor(media_img, cv2.COLOR_BGR2GRAY)  # convert roi into gray
            Blur = cv2.GaussianBlur(gray, (5, 5), 1)  # apply blur to roi
            # Canny=cv2.Canny(Blur,10,50) #apply canny to roi
            _, It = cv2.threshold(Blur, 0, 255, cv2.THRESH_OTSU)
            sx = cv2.Sobel(It, cv2.CV_32F, 1, 0)
            sy = cv2.Sobel(It, cv2.CV_32F, 0, 1)
            m = cv2.magnitude(sx, sy)
            m = cv2.normalize(m, None, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U)
            m = cv2.ximgproc.thinning(m, None, cv2.ximgproc.THINNING_GUOHALL)

            # Find my contours
            contours = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
            print(len(big_contours))
            total_imagenes += len(big_contours)
    return total_imagenes - 2  # Eliminamos 2 preguntas de portada
