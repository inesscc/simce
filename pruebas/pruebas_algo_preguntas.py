# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 16:59:23 2024

@author: jeconchao
"""

from simce.config import dir_estudiantes
from itertools import chain
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from pathlib import Path
from time import time
from simce.utils import dic_img_preg
import pandas as pd


now = time()
e1 = Path('data/input/cuestionario_estudiantes/09955/')
est = [i for i in e1.iterdir() if '4272468' in str(i)]
n_pages = len(est) * 2
n_preguntas = 29 # OJO, está hardcodeado, se podría hacer una función que obtenga automático
pages = (n_pages, 1)
q1 = 0
q2 = n_preguntas +1

for n, preg in enumerate(est):
    file_dir = re.sub(r'data\\input\\cuestionario_estudiantes\\', '', str(preg))
    folder, file = file_dir.split('\\')
    
    file_no_ext = Path(file).with_suffix('')
    id_est = re.search('\d+',f'{file_no_ext}').group(0)
    Path(f'data/output/{folder}').mkdir(exist_ok=True)
    page = str(file_no_ext)[-1]
    
    img_preg = cv2.imread(str(preg),1)
    
    
    
    x,y = img_preg.shape[:2]
    img_crop = img_preg[40:x - 200, 50:y-160]
    
    punto_medio = int(np.round(img_crop.shape[1] / 2, 1))
    
    img_p1 = img_crop[:, :punto_medio] 
    img_p2 = img_crop[:, punto_medio:]
    
    if (n % 2 == 0) & (n != 0): # si n es par y no es la primera página
        pages = (pages[1]-1, pages[0] + 1) 
    elif n % 2 == 1:
        pages = (pages[1]+1, pages[0] - 1)
    print(pages)



    for p, media_img in enumerate([img_p1, img_p2]):
        print(media_img.shape)
        
        gray = cv2.cvtColor(media_img, cv2.COLOR_BGR2GRAY) #convert roi into gray
        Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
       # Canny=cv2.Canny(Blur,10,50) #apply canny to roi
        _,It = cv2.threshold(Blur,0,255,cv2.THRESH_OTSU)
        sx = cv2.Sobel(It,cv2.CV_32F,1,0)
        sy = cv2.Sobel(It,cv2.CV_32F,0,1)
        m = cv2.magnitude(sx,sy)
        m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
        m = cv2.ximgproc.thinning(m,None,cv2.ximgproc.THINNING_GUOHALL)
        
        #Find my contours
        contours =cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
        print([i[0][0][1] for i in big_contours] )

       # big_contours_sizes = [cv2.contourArea(i) for i in big_contours]
        
      #  big_contours_sort = [i for _, i in sorted(zip(big_contours_sizes, big_contours))]
          
        print(f'página actual: {pages[p]}')
        
        if pages[p] < pages[1-p]:
            # revertimos orden de contornos cuando es la página baja del cuadernillo
           big_contours = big_contours[::-1]
    
        for c in (big_contours):
            print(n)
            x,y,w,h= cv2.boundingRect(c)
            cropped_img=media_img[y:y+h, x:x+w]
            
            if pages[p] > pages[1-p]: # si es la pág más alta del cuadernillo
                q2 -= 1
                q = q2
            elif (pages[p] < pages[1-p]) & (pages[p] != 1): # si es la pág más baja del cuardenillo
                q1 += 1
                q = q1
            else: # Para la portada
                q = '_'
            
    
           # id_img = f'{page}_{n}'
            n += 1
            file_out = f'data/output/{folder}/{id_est}_p{q}.jpg'
            print(file_out)
            cv2.imwrite(file_out, cropped_img)
            
#%%

cv2.imshow("Detected Lines", cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
