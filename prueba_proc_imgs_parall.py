
import os
import re
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count, Manager
import time

from itertools import chain

from config.proc_img import dir_subpreg, regex_estudiante, dir_tabla_99, \
    dir_input, n_pixeles_entre_lineas, dir_estudiantes, dir_padres, regex_extraer_rbd_de_ruta

from simce.errors import agregar_error, escribir_errores
from simce.utils import get_mask_imagen

from simce.proc_imgs import select_directorio, get_insumos, get_pages, get_subpregs_distintas, eliminar_franjas_negras, recorte_imagen, \
    obtener_lineas_horizontales, bound_and_crop, crop_and_save_subpreg, get_pregunta_inicial_pagina, \
    partir_imagen_por_mitad, get_contornos_grandes, dejar_solo_recuadros_subpregunta, get_mascara_lineas_horizontales

import json
from config.proc_img import dir_insumos

from dotenv import load_dotenv
load_dotenv()

VALID_INPUT = {'cuadernillo', 'pagina'}

## procesamiento imagen ----------------------------------

def process_single_image(df99, num, rbd, directorio_imagenes, dic_pagina, n_pages, subpreg_x_preg, 
                         dir_subpreg, tipo_cuadernillo, regex_estudiante, queue):
    
    """
    Procesamiento de una sola imagen
    """
    # from pathlib import Path
    # df99 = df99[df99['serie'] == 4077894] #data\input_raw\CP\02748\4077894_4.jpg
    # num = 1
    # rbd = Path('data/input_raw/CP/02748/4077894_4.jpg')
    
    pregunta_selec = re.search(r'p(\d{1,2})', df99.iloc[num].preguntas).group(0)
    estudiante = re.search(regex_estudiante, str(rbd)).group(1)
    pagina_pregunta = dic_pagina[pregunta_selec]
    pages = get_pages(pagina_pregunta, n_pages)
    
    dir_subpreg_rbd = (dir_subpreg /f'{rbd.parent.parent.name}'/ f'{rbd.parent.name}')
    
    dir_subpreg_rbd.mkdir(exist_ok=True, parents=True)

    if not rbd.is_file():
        preg_error = dir_subpreg_rbd / f'{estudiante}'
        agregar_error(queue= queue,
                      pregunta=str(preg_error),
                      error=f'No existen archivos disponibles para serie {preg_error.name}',
                      nivel_error=tipo_cuadernillo)
        return 'Ocurrió un error'

    file = rbd.name
    print(f'{file=}')
    # Leemos imagen
    img_preg = cv2.imread(str(rbd), 1) 
    img_crop = recorte_imagen(img_preg, 0, 150, 50, 160)
    
    # Eliminamos franjas negras en caso de existir
    img_sin_franja = eliminar_franjas_negras(img_crop)
    
    # Divimos imagen en dos páginas del cuadernillo
    paginas_cuadernillo = partir_imagen_por_mitad(img_sin_franja)
    
    # Seleccionamos página que nos interesa, basado en diccionario de páginas
    media_img = paginas_cuadernillo[pages.index(pagina_pregunta)]
    
    if media_img is None:
        print(f"Error: No se pudo cargar la imagen")
        agregar_error(queue= queue,
                      pregunta=str(dir_subpreg_rbd / f'{estudiante}'),
                      error=f'No se pudo cargar la mitad de la imagen',
                      nivel_error=tipo_cuadernillo)
        
        return 'Ocurrió un error'
    
    else:
        # Detecto recuadros naranjos
        try:
            mask_naranjo = get_mask_imagen(media_img)
        
            # Obtengo contornos
            big_contours = get_contornos_grandes(mask_naranjo)
            
            q_base = get_pregunta_inicial_pagina(dic_pagina, pagina_pregunta)
            pregunta_selec_int = int(re.search(r'\d+', pregunta_selec).group(0))

            try:
                # Obtengo coordenadas de contornos y corto imagen
                elemento_img_pregunta = big_contours[pregunta_selec_int - q_base]
                img_pregunta = bound_and_crop(media_img, elemento_img_pregunta)

                img_pregunta_recuadros = dejar_solo_recuadros_subpregunta(img_pregunta)
                
                # Exportamos pregunta si no tiene subpreguntas:
                if subpreg_x_preg[pregunta_selec] == 1:
                    print('Pregunta no cuenta con subpreguntas, se guardará imagen')
                    file_out = str(
                        dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}.jpg')

                    n_subpreg = 1
                    cv2.imwrite(file_out, img_pregunta_recuadros)
                    
                    return 'Éxito!'

                subpreg_selec = df99.iloc[num].preguntas.split('_')[1]
                print(f'{subpreg_selec=}')
                
                # Obtenemos subpreguntas:
                #img_pregunta_crop = recorte_imagen(img_pregunta)
                # img_crop_col = get_mask_imagen(img_pregunta_recuadros,
                #                                lower_color=np.array(
                #                                    [0, 111, 109]),
                #                                upper_color=np.array([18, 255, 255]))

                img_crop_col = get_mascara_lineas_horizontales(img_pregunta_recuadros)
                
                lineas_horizontales = obtener_lineas_horizontales(
                    img_crop_col, minLineLength=np.round(img_crop_col.shape[1] * .6))
                
                n_subpreg = len(lineas_horizontales) - 1

                if n_subpreg != subpreg_x_preg[pregunta_selec]:
                    preg_error = str(dir_subpreg_rbd / f'{estudiante}')
                    dic_dif = get_subpregs_distintas(subpreg_x_preg, dir_subpreg_rbd, estudiante)
                    error = f'N° de subpreguntas incorrecto para serie {estudiante}, se encontraron {n_subpreg} subpreguntas {dic_dif}'
                    agregar_error(queue= queue, pregunta=preg_error, error=error, nivel_error=tipo_cuadernillo)
            

                try:
                    file_out = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}.jpg')
                    crop_and_save_subpreg(img_pregunta_recuadros, lineas_horizontales,
                                          i=int(subpreg_selec)-1, file_out=file_out)
                
                # Si hay error en procesamiento subpregunta
                except Exception as e:
                    preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}')
                    agregar_error(queue= queue,
                                pregunta=preg_error, 
                                error='Subregunta no pudo ser procesada',
                                nivel_error='Subpregunta', 
                                )
                    return 'Ups, ocurrio un error en la subpregunta'


            except Exception as e:
                preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}')
                agregar_error(queue= queue, pregunta=preg_error, error='Pregunta no pudo ser procesada', nivel_error='Pregunta')
                return
            
        except Exception as e:
            print('Ocurrio un error con la mascara')
            preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}')
            agregar_error(queue= queue, pregunta=preg_error, error='Ocurrio un error con la mascara', nivel_error='Pregunta')
            
    return 'Éxito!'


## division en bloques --------------------

def process_image_block(image_block):
    queue, df99, directorio_imagenes, dic_pagina, n_pages, subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante = image_block

    for num, rbd in enumerate(directorio_imagenes):
        print(num)
        process_single_image(df99, num, rbd, directorio_imagenes, dic_pagina, n_pages,
                             subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante,
                             queue)

#process_image_block(image_blocks[0])
#df99[df99['serie'] == 4077894]


def process_general(directorio_imagenes, tipo_cuadernillo, para_entrenamiento, 
         regex_estudiante, dir_tabla_99, dir_input, dir_subpreg,
         muestra, filter_rbd, filter_rbd_int, filter_estudiante, queue):
    
    if para_entrenamiento:
        nombre_tabla_casos99 = f'casos_99_entrenamiento_compilados_{tipo_cuadernillo}.csv'
    else:
        nombre_tabla_casos99 = f'casos_99_compilados_{tipo_cuadernillo}.csv'
    df99 = pd.read_csv(dir_tabla_99 / nombre_tabla_casos99, dtype={'rbd_ruta': 'string'}).sort_values('ruta_imagen')

    # Filtrar 
    if muestra:
        rbd_disp = {i.name for i in directorio_imagenes.iterdir()}
        df99 = df99[(df99.rbd_ruta.isin(rbd_disp))]

    if filter_rbd:
        if filter_rbd_int:
            df99 = df99[(df99.rbd_ruta.astype(int).ge(filter_rbd))]
        else:
            df99 = df99[(df99.rbd_ruta.eq(filter_rbd))]

    if filter_estudiante:
        if isinstance(filter_estudiante, int):
            filter_estudiante = [filter_estudiante]
        df99 = df99[df99.serie.isin(filter_estudiante)]
        
    df99.ruta_imagen = df99.ruta_imagen.str.replace('\\', '/')
    dir_preg99 = [dir_input / i for i in df99.ruta_imagen]

    n_pages, n_preguntas, subpreg_x_preg, dic_cuadernillo, dic_pagina, n_subpreg_tot = get_insumos(tipo_cuadernillo)

    # Dividir en bloques para procesamiento paralelo
    num_workers = 20 #cpu_count() -1
    print('###########')
    print(num_workers)
    print('###########')
    
    block_size = len(dir_preg99) // num_workers
    image_blocks = [(queue, df99[i:i + block_size], dir_preg99[i:i + block_size], dic_pagina, n_pages, 
                     subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante) for i in range(0, len(dir_preg99), block_size)]

    # Usar multiprocessing Pool
    with Pool(num_workers) as pool:
        print('-------------')
        pool.map(process_image_block, image_blocks)

    return 'Éxito!'



if __name__ == "__main__":
    
    tipo_cuadernillo = 'estudiantes'
    directorio_imagenes = select_directorio(tipo_cuadernillo)
    para_entrenamiento = True  
    muestra = False  
    filter_rbd = None  
    filter_rbd_int = None 
    filter_estudiante = None  
    regex_estudiante = regex_estudiante 
    dir_tabla_99 = dir_tabla_99
    dir_input = dir_input
    dir_subpreg = dir_subpreg 
    
    manager = Manager()
    queue = manager.Queue()
    
    inicio = time.time()
    process_general(directorio_imagenes, tipo_cuadernillo, para_entrenamiento, 
                    regex_estudiante, dir_tabla_99, dir_input, dir_subpreg,
                    muestra, filter_rbd, filter_rbd_int, filter_estudiante,queue)
    
    fin = time.time() - inicio
    print(f"Tiempo de procesamiento: {fin:.2f}")
    
    inicio2 = time.time()
    escribir_errores(queue)
    fin2 = time.time() - inicio2
    print(f"Tiempo en escribir errores: {fin2:.2f}")
    
    