
import os
import re
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
from itertools import chain
from simce.errors import agregar_error
from simce.utils import get_mask_imagen
from simce.proc_imgs import get_insumos, get_pages, get_subpregs_distintas, eliminar_franjas_negras, recorte_imagen, \
    obtener_lineas_horizontales, bound_and_crop, crop_and_save_subpreg, get_pregunta_inicial_pagina, save_pregunta_completa, \
    partir_imagen_por_mitad, get_contornos_grandes, dejar_solo_recuadros_subpregunta, get_mascara_lineas_horizontales
from dotenv import load_dotenv
load_dotenv()
from simce.utils import timing
VALID_INPUT = {'cuadernillo', 'pagina'}

## procesamiento imagen ----------------------------------

def process_single_image(preguntas, num: int, rbd, dic_pagina:dict, n_pages: int, subpreg_x_preg: dict, 
                         dir_subpreg, tipo_cuadernillo:str, regex_estudiante:str, queue):
    
    """
    Genera el recorte de una pregunta/subpregunta. Primero verificamos que la imagen no posea franjas negras en ningun costado del cuestionario.
    Posteriormente, dividimos la imagen en dos paginas del cuadernillo y seleccionamos la pagina de la pregunta a recortar. \
        En aquella pagina, detectamos los recuadros naranjos y realizamos un primer recorte según las coordenadas obtenidas. \
            Luego identificaremos los recuadros blancos y volvemos a cortar la imagen.
    
    Si la pregunta seleccionada no posee subpreguntas, nos quedamos con el recorte de recuadros blancos y guardamos la imagen.
    Si la pregunta seleccionada posee subpreguntas, identificaremos las lineas horizontales y \
        guardaremos la imagen con la sección de la subpregunta de interes. 

    Args:
        - preguntas (pd.series): lista con preguntas a recortar
        - num (int): Id de pregunta a recortar
        - rbd: Ruta de la pregunta a recortar
        - dict_pagina (dict): Diccionario con mapeo de preguntas en la pagina del cuestionario
        - n_pages (int): Cantidad de paginas que posee el cuestionario en total
        - subpreg_x_preg (dict): Insumo con cantidad de subpreguntas por pregunta.
        - dir_subpreg (): Directorio general en donde se guardarán las preguntas recortadas 
        - tipo_cuadernillo (str): define si se está procesando para estudiantes o padres.
        - regex_estudiante (str): Expresion regular que nos ayuda a identificar el n° de serie del cuestionario
        - queue (multiprocessing.Manager().Queue()): Cola de tareas gestionada por Manager() para intercambiar datos entre procesos de forma segura.
    
    Returns:
        
        
    """
    # from pathlib import Path
    # df99 = df99[df99['serie'] == 4077894] #data\input_raw\CP\02748\4077894_4.jpg
    # num = 1
    # rbd = Path('data/input_raw/CP/02748/4077894_4.jpg')
    
    pregunta_selec = re.search(r'p(\d{1,2})', preguntas.iloc[num]).group(0)          # seleccion de pregunta
    estudiante = re.search(f'({regex_estudiante})', str(rbd)).group(1)               # serie estudiante
    pagina_pregunta = dic_pagina[pregunta_selec]                                     # ubicacion pregunta
    pages = get_pages(pagina_pregunta, n_pages)
    
    dir_subpreg_rbd = (dir_subpreg /f'{rbd.parent.parent.name}'/ f'{rbd.parent.name}')  # obtencion path pregunta
    
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
                    save_pregunta_completa(img_pregunta_recuadros, dir_subpreg_rbd, estudiante, pregunta_selec)
                    
                    return 'Éxito!'

                subpreg_selec = preguntas.iloc[num].split('_')[1]
                print(f'{subpreg_selec=}')
                
                # Obtenemos subpreguntas:
                #img_pregunta_crop = recorte_imagen(img_pregunta)
                # img_crop_col = get_mask_imagen(img_pregunta_recuadros,
                #                                lower_color=np.array(
                #                                    [0, 111, 109]),
                #                                upper_color=np.array([18, 255, 255]))

                # Obtenemos lineas horizontales:
                img_crop_col = get_mascara_lineas_horizontales(img_pregunta_recuadros)
                
                lineas_horizontales = obtener_lineas_horizontales(
                    img_crop_col, minLineLength=np.round(img_crop_col.shape[1] * .6))
                
                n_subpreg = len(lineas_horizontales) - 1

                if n_subpreg != subpreg_x_preg[pregunta_selec]:
                    preg_error = str(dir_subpreg_rbd / f'{estudiante}')
                    dic_dif = get_subpregs_distintas(subpreg_x_preg, dir_subpreg_rbd, estudiante)
                    error = f'N° de subpreguntas incorrecto para serie {estudiante}, se encontraron {n_subpreg} subpreguntas {dic_dif}'
                    agregar_error(queue= queue, pregunta=preg_error, error=error, nivel_error=tipo_cuadernillo)
            
                # Realizamos recorte y guardado de subpregunta
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

def process_image_block(image_block:list):
    """
    Envia a cada bloque la función process_single_image(), la cual se encarga de realizar los recortes a cada una de las imagenes disponibles en los bloques.
    
    Args:
        - image_block (list): lista con los objetos a usar en cada uno de los bloques, posee los insumos, path de las imagenes y queue.
        
    Returns:
        
    """
    queue, preguntas, directorio_imagenes, dic_pagina, n_pages, subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante = image_block

    for num, rbd in enumerate(directorio_imagenes):
        process_single_image(preguntas, num, rbd, dic_pagina, n_pages,
                             subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante,
                             queue)

#process_image_block(image_blocks[0])
#df99[df99['serie'] == 4077894]

@timing
def process_general(dirs:dict, regex_estudiante: str, queue, curso: str, tipo_cuadernillo: str,
                    muestra= None, filter_rbd= None, filter_rbd_int= False, filter_estudiante= None):
    
    """
    Genera el recorte de preguntas/subpreguntas, de forma paralelizada, para los registros obtenidos de la función get_tablas_99().
    Para ello, se utilizan los insumos disponibles get_insumos(), se cuenta el total de CPUs para realizar el procesamiento simultaneo, \
        se divide las preguntas equitativamente para que cada nucleo posea aproximadamente la misma cantidad de preguntas a recortar, \
            y finalmente se aplica process_image_block() para realizar el procesamiento de cada pregunta en los diferentes bloques disponibles \
                (aplicamos la misma funcion de procesamiento en cada uno de los bloques). 
    
    Args:
        - dirs (dict): Diccionario con los directorios a usar
        - regex_estudiante (str): Expresion regular que nos ayuda a identificar el n° de serie del cuestionario  
        - queue (multiprocessing.Manager().Queue()): Cola de tareas gestionada por Manager() para intercambiar datos entre procesos de forma segura.
        - curso (str): Nombre carpeta que identifica el curso en procesamiento.
        - tipo_cuadernillo (str): define si se está procesando para estudiantes o padres. Esto también se utiliza para definir las rutas a consultar

    Returns:
        Retorna un string el cual indica si el procesamiento de imagenes se realizó correctamente.
    """
    
    
    print(f'Procesando cuadernillo {tipo_cuadernillo}')
    


    nombre_tabla_casos99 = f'casos_99_compilados_{curso}_{tipo_cuadernillo}.csv'
    df99 = pd.read_csv(dirs['dir_tabla_99'] / nombre_tabla_casos99, dtype={'rbd_ruta': 'string'}).sort_values('ruta_imagen')

    # Filtrar 
    if muestra:
        rbd_disp = {i.name for i in dirs[f'dir_{tipo_cuadernillo}'].iterdir()}
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
    dir_preg99 = [dirs['dir_input'] / i for i in df99.ruta_imagen]

    n_pages, _, subpreg_x_preg, _, dic_pagina, _ = get_insumos(tipo_cuadernillo,
                                                                dir_insumos=dirs['dir_insumos'])

    # Dividir en bloques para procesamiento paralelo
    num_workers = cpu_count() -1
    print('###########')
    print(f'Cantidad de CPUs a usar {num_workers}')
    print('###########')
    
    block_size = len(dir_preg99) // num_workers
    print(f'## Cantidad de preguntas en cada bloque: {block_size}')
    
    image_blocks = [(queue, df99[i:i + block_size].preguntas, dir_preg99[i:i + block_size], dic_pagina, n_pages, 
                     subpreg_x_preg, dirs['dir_subpreg'], tipo_cuadernillo, regex_estudiante) for i in range(0, len(dir_preg99), block_size)]

    # Usar multiprocessing Pool
    with Pool(num_workers) as pool:
        print('-------------')
        pool.map(process_image_block, image_blocks)

    return 'Éxito!'

