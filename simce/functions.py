from itertools import chain
import numpy as np
import cv2
import pandas as pd
from os import getcwd, scandir
from os.path import abspath

# Funciones ---------------------------------------------------------------------

### Procesamiento imagen

def recorte_imagen(img_preg, x0 =130, x1= 30, y0 = 50, y1=50):
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
    
    x,y = img_preg.shape[:2]
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


### Procesamiento sub-pregunta

def obtener_puntos(img_crop_canny, threshold = 100, minLineLength = 100):
    """
    Funcion que identifica lineas para obtener puntos en el eje "y" para realizar el recorte a subpreguntas

    Args:
        img_crop_canny (_type_): _description_

    Returns:
        lines: _description_
    """
    # obteniendo lineas
    lines = cv2.HoughLinesP(img_crop_canny, 1, np.pi/180, threshold= threshold, minLineLength = minLineLength)
    
    indices_ordenados = np.argsort(lines[:, :, 1].flatten())
    lines_sorted = lines[indices_ordenados]
    
    puntoy = list(set(chain.from_iterable(lines_sorted[:, :,1].tolist())))
    puntoy.append(img_crop_canny.shape[0])
    puntoy = sorted(puntoy)
    
    print(puntoy)
    
    y = []
    for i in range(len(puntoy)-1):
        if puntoy[i+1]- puntoy[i]<35:
            y.append(i+1)

    print(puntoy)
    print(y)
    
    for index in sorted(y, reverse=True):
        del puntoy[index]
    
    return puntoy


def ls(ruta = getcwd()):
    """Funcion para obtener la ruta de los archivos dentro de la carpeta indicada."""
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]