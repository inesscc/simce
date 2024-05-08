# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:54:01 2024

@author: jeconchao
"""

from simce.proc_imgs import get_mask_naranjo, recorte_imagen

from simce.config import dir_estudiantes
from simce.utils import crear_directorios, get_n_paginas, get_n_preguntas

import cv2
from pathlib import Path
import re
import numpy as np
import simce.proc_imgs as proc
# Creamos directorios
crear_directorios()

n_pages = get_n_paginas()
n_preguntas = get_n_preguntas() 
revisar_pregunta = []


#%% Preguntas
for num, rbd in enumerate(dir_estudiantes.iterdir()):

    print('#################################')
    print(rbd)
    print(num)
    print('#################################')
    estudiantes_rbd = {re.search('\d{7}',str(i)).group(0) 
                       for i in rbd.iterdir()}
#    if rbd.name == '09963':
    for n, estudiante in enumerate(estudiantes_rbd):
        
        if estudiante == '4274380':
            # páginas del cuardenillo
            pages = (n_pages, 1)
            # pregunta inicial páginas bajas
            q1 = 0
            # pregunta inicial páginas altas
            q2 = n_preguntas + 1
            
            # Para cada imagen del cuadernillo de un estudiante (2 páginas por imagen):
            for num_pag, pag in enumerate(rbd.glob(f'{estudiante}*')):
                
                # Obtengo carpeta del rbd y archivo del estudiante a partir del path:
                 folder, file = (pag.parts[-2], pag.parts[-1])
                
               #  if int(folder) == 93 and estudiante == '4003683':
                 print('file:', file)
                 print(f'num_pag: {num_pag}')
                # print(pages)
                # Quitamos extensión al archivo
                 file_no_ext = Path(file).with_suffix('')
                 # Creamos directorio si no existe
                 Path(f'data/output_preg/{folder}').mkdir(exist_ok=True, parents=True)
                 
                 # Obtenemos página del archivo
                 page = re.search('\d+$',str(file_no_ext)).group(0)
                 
                 # Leemos imagen
                 img_preg = cv2.imread(str(pag),1)
                 
                 # Recortamos info innecesaria de imagen
                 img_crop = recorte_imagen(img_preg, 0, 200, 50, 160)
    
                 # Buscamos punto medio de imagen para dividirla en las dos páginas del cuadernillo
                 punto_medio = int(np.round(img_crop.shape[1] / 2, 1))
                 
                 img_p1 = img_crop[:, :punto_medio] # página izquierda
                 img_p2 = img_crop[:, punto_medio:] # página derecha
                 
                 # Obtenemos páginas del cuadernillo actual:
                 if (num_pag % 2 == 0) & (num_pag != 0): # si n es par y no es la primera página
                     pages = (pages[1]-1, pages[0] + 1) 
                 elif num_pag % 2 == 1:
                     pages = (pages[1]+1, pages[0] - 1)
                 print(pages)
                 
                 
                 # Para cada una de las dos imágenes del cuadernillo
                 for p, media_img in enumerate([img_p1, img_p2]):
                     
                     # Detecto recuadros naranjos
                     m = get_mask_naranjo(media_img)
                     
                     # Obtengo contornos
                     contours =cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
                     # Me quedo contornos grandes
                     big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
                   #  print([i[0][0][1] for i in big_contours] )
    
                       
                   #  print(f'página actual: {pages[p]}')
                     
                     if pages[p] < pages[1-p]:
                         # revertimos orden de contornos cuando es la página baja del cuadernillo
                        big_contours = big_contours[::-1]
                 
                     for c in (big_contours):
                         # Obtengo coordenadas de contornos
                         x,y,w,h= cv2.boundingRect(c)
                         img_pregunta = media_img[y:y+h, x:x+w]
                         
                         # Obtengo n° de pregunta en base a lógica de cuadernillo:
                         if pages[p] > pages[1-p]: # si es la pág más alta del cuadernillo
                             q2 -= 1
                             q = q2
                         elif (pages[p] < pages[1-p]) & (pages[p] != 1): # si es la pág más baja del cuardenillo
                             q1 += 1
                             q = q1
                         else: # Para la portada
                             q = '_'
                       
                         
    
                         file_out = f'data/output_preg/{folder}/{estudiante}_p{q}.jpg'
                         print(file_out)
                         cv2.imwrite(file_out, img_pregunta)
                         
#%%
cv2.imshow("Detected Lines",cv2.resize(m, (900, 900)))
cv2.waitKey(0)
cv2.destroyAllWindows()

     
#%% Chequeos

import pandas as pd
from pathlib import Path
errores_procesamiento = dict()

for folder in Path('data/output_preg/').iterdir():
    s = pd.Series([re.match('\d+', i.name).group(0) for i in folder.iterdir()]).value_counts()
    if s.min() < 30 or s.max() > 30:
        print(folder)
        errores_procesamiento.update({folder: s[s<30].index})
        
        print('mín: ', s.min())
        print('máx: ', s.max())
