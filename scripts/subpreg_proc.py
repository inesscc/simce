import numpy as np
import cv2
import pandas 
from time import time

import simce.functions as func


# Procesamiento a subpreguntas----------------------------------------------

## ejemplo
preguntas_ejemplo = func.ls(r'data\output\09952')
folder = 'Subpreguntas'                           ### EDITAR RUTA ###

now = time()
revisar_pregunta = []           # revisar en que preguntas se cae

for pregunta in preguntas_ejemplo:
    
    id_pregunta = pregunta.split('\\')            ### ELIMINAR SACANDO ID ###
    preg = id_pregunta[-1].replace('.jpg', '')    ### EDITAR: NOMBRE PREGUNTA ###
    print('Revisando pregunta ' + id_pregunta[3])
    
    try:
        img_preg = cv2.imread(pregunta,1)
        img_crop = func.recorte_imagen(img_preg)
        img_crop_col = func.procesamiento_color(img_crop)
        
        puntoy = func.obtener_puntos(img_crop_col)
        
        n = 1 # id subpregunta
        try:
            for i in range(len(puntoy)-1):
                cropped_img = img_crop[puntoy[i]:puntoy[i+1],]
                
                page = 'Subpreg'                                   ### EDITAR RUTA ###
                id_img = f'{id_pregunta[2]}_{preg}_{page}_{n}'     
                n += 1
                file_out = f'data/output/{folder}/{id_img}.jpg'    ### EDITAR RUTA ###
                print(file_out)
                
                cv2.imwrite(file_out, cropped_img)
                
        except Exception as e:
            print('Ups, ocurrió un error al recortar la imagen con subpregunta ' + str(n))
            print(e)
            revisar_pregunta.append(pregunta+ '__'+ str(n))
            
    except Exception as e:
        
        print('Ups, ocurrió un error con la pregunta' + pregunta)
        print(e)
        revisar_pregunta.append(pregunta)
        
print(time() - now)
print(revisar_pregunta)