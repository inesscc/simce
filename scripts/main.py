# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:34:05 2024

@author: jeconchao
"""

from simce.utils import crear_directorios, get_n_paginas, get_n_preguntas, get_mask_naranjo
from simce.config import dir_estudiantes
import cv2
from pathlib import Path
import re
import numpy as np
# Creamos directorios
crear_directorios()

n_pages = get_n_paginas()
n_preguntas = get_n_preguntas() 




#%%
for num, rbd in enumerate(dir_estudiantes.iterdir()):
    print(rbd)

    estudiantes_rbd = {re.search('\d{7}',str(i)).group(0) 
                       for i in rbd.iterdir()}
    
    for n, estudiante in enumerate(estudiantes_rbd):
        # páginas del cuardenillo
        pages = (n_pages, 1)
        # pregunta inicial páginas bajas
        q1 = 0
        # pregunta inicial páginas altas
        q2 = n_preguntas + 1
        for num_pag, pag in enumerate(rbd.glob(f'{estudiante}*')):
             file_dir = re.sub(r'data\\input\\cuestionario_estudiantes\\', '', str(pag))
             folder, file = file_dir.split('\\')
            
          #   if int(folder) == 10013 and estudiante == '4274572':
             print('file:', file)
             print(f'num_pag: {num_pag}')
            # print(pages)
             file_no_ext = Path(file).with_suffix('')
     
             Path(f'data/output/{folder}').mkdir(exist_ok=True)
             page = str(file_no_ext)[-1]
             
             img_preg = cv2.imread(str(pag),1)
             
             x,y = img_preg.shape[:2]
             img_crop = img_preg[:x - 200, 50:y-160]
             
             punto_medio = int(np.round(img_crop.shape[1] / 2, 1))
             
             img_p1 = img_crop[:, :punto_medio] 
             img_p2 = img_crop[:, punto_medio:]
             
             if (num_pag % 2 == 0) & (num_pag != 0): # si n es par y no es la primera página
                 pages = (pages[1]-1, pages[0] + 1) 
             elif num_pag % 2 == 1:
                 pages = (pages[1]+1, pages[0] - 1)
             print(pages)
             
             
             for p, media_img in enumerate([img_p1, img_p2]):
            #     print(media_img.shape)
                 
                #  gray = cv2.cvtColor(media_img, cv2.COLOR_BGR2GRAY) #convert roi into gray
                #  Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
                # # Canny=cv2.Canny(Blur,10,50) #apply canny to roi
                #  _,It = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
                #  sx = cv2.Sobel(It,cv2.CV_32F,1,0)
                #  sy = cv2.Sobel(It,cv2.CV_32F,0,1)
                #  m = cv2.magnitude(sx,sy)
                #  m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
                #  m = cv2.ximgproc.thinning(m,None,cv2.ximgproc.THINNING_GUOHALL)
                #  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                #  m = cv2.dilate(m, kernel, iterations=2)
                 m = get_mask_naranjo(media_img)
                 
                 #Find my contours
                 contours =cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
                 big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
               #  print([i[0][0][1] for i in big_contours] )

                   
               #  print(f'página actual: {pages[p]}')
                 
                 if pages[p] < pages[1-p]:
                     # revertimos orden de contornos cuando es la página baja del cuadernillo
                    big_contours = big_contours[::-1]
             
                 for c in (big_contours):
                     
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
                     file_out = f'data/output/{folder}/{estudiante}_p{q}.jpg'
                     print(file_out)
                     cv2.imwrite(file_out, cropped_img)
         

#%%
import pandas as pd
from pathlib import Path
errores_procesamiento = dict()
for folder in Path('data/output/').iterdir():
    s = pd.Series([re.match('\d+', i.name).group(0) for i in folder.iterdir()]).value_counts()
    if s.min() > 30:
        print(folder)
        errores_procesamiento.update({folder: s[s<30].index})
        
        print('mín: ', s.min())
        print('máx: ', s.max())

#%%
folder = '10007'
e2 = Path(f'data/output/{folder}')
s = pd.Series([re.match('\d+', i.name).group(0) for i in e2.iterdir()]).value_counts()
print(s)

#%%

e3 = Path(f'data/output')
n
for n,i in enumerate(e3.rglob('*')):
    pass
    
#%%

for contour in big_contours:
   cv2.drawContours(media_img, contour, -1, (0, 255, 0), 3)
cv2.imshow("Detected Lines", cv2.resize(media_img, (900, 900)))

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

# def apply_approx(cnt):

#     epsilon = 0.45*cv2.arcLength(cnt,True)
#     approx = cv2.approxPolyDP(cnt,epsilon,True)
#     return approx 

#%%


contours =cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
big_contours = [i for i in contours if cv2.contourArea(i) > 30000]



for cnt in big_contours:
    # epsilon = 0.1*cv2.arcLength(cnt,True)
    # approx = cv2.approxPolyDP(cnt,epsilon,True)
    # print(approx)
    cv2.drawContours(media_img, [cv2.boundingRect(cnt)], 0, (0, 255, 0), 3)
   
   
cv2.imshow("Detected Lines", cv2.resize(media_img, (900, 900)))
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
cv2.convexHull(cnt)

#%%
cnt = big_contours[0]
hull = cv2.convexHull(cnt)
for contour in big_contours:
    print(cv2.convexHull(contour).shape)

#%%


mask_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# Find contours in the mask
contours = cv2.findContours(mask_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
