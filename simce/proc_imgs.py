# -*- coding: utf-8 -*-
"""
Created on Thu May  2 17:22:37 2024

@author: jeconchao
"""
import numpy as np
import cv2
from itertools import chain

def get_mask_naranjo(media_img, lower_color=np.array([13, 52, 0]), upper_color = np.array([29, 255, 255])):
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

    # Crea una máscara binaria donde los píxeles de la imagen que están dentro del rango de color especificado son blancos, y todos los demás píxeles son negros.
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask = cv2.dilate(mask, kernel, iterations=2)
    # Devuelve la máscara
    return mask






      
      
      

def recorte_imagen(img_preg, x0 =130, x1= 20, y0 = 50, y1=50):
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

def obtener_puntos(img_crop_canny, threshold = 100, minLineLength = 200):
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
    
    #print(puntoy)
    
    y = []
    for i in range(len(puntoy)-1):
        if puntoy[i+1]- puntoy[i]<27:
            y.append(i+1)

   # print(puntoy)
    #print(y)
    
    for index in sorted(y, reverse=True):
        del puntoy[index]
    
    return puntoy


def procesamiento_antiguo(media_img):
    '''Función en desuso. Procesaba imagen para detección de contornos'''
    
    gray = cv2.cvtColor(media_img, cv2.COLOR_BGR2GRAY) #convert roi into gray
    #Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
    #Canny=cv2.Canny(Blur,10,50) #apply canny to roi
    _,It = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    sx = cv2.Sobel(It,cv2.CV_32F,1,0)
    sy = cv2.Sobel(It,cv2.CV_32F,0,1)
    m = cv2.magnitude(sx,sy)
    m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
    m = cv2.ximgproc.thinning(m,None,cv2.ximgproc.THINNING_GUOHALL)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    m = cv2.dilate(m, kernel, iterations=2)
