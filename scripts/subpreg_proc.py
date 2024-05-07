import numpy as np
import cv2
import pandas 
from time import time
from simce.utils import ls
import simce.proc_imgs as proc
from simce.proc_imgs import get_mask_naranjo


# Procesamiento a subpreguntas----------------------------------------------

## ejemplo
preguntas_ejemplo = ls(r'data\output\09952')
folder = 'Subpreguntas'                           ### EDITAR RUTA ###

now = time()
revisar_pregunta = []           # revisar en que preguntas se cae

for pregunta in preguntas_ejemplo:
    
    id_pregunta = pregunta.split('\\')            ### ELIMINAR SACANDO ID ###
    preg = id_pregunta[-1].replace('.jpg', '')    ### EDITAR: NOMBRE PREGUNTA ###
    print('Revisando pregunta ' + id_pregunta[3])
    
    try:
        img_preg = cv2.imread(pregunta,1)
        img_crop = proc.recorte_imagen(img_preg)
        img_crop_col = get_mask_naranjo(img_crop, lower_color=np.array([0, 114, 139]), upper_color = np.array([17, 255, 255]))
        
        puntoy = proc.obtener_puntos(img_crop_col)
        
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