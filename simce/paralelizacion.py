
from os import PathLike
import re
import cv2
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from simce.errors import agregar_error
from simce.utils import get_mask_imagen
from config.proc_img import n_pixeles_entre_lineas
from simce.proc_imgs import get_insumos, get_pages_cuadernillo, get_subpregs_distintas, eliminar_franjas_negras, recorte_imagen, \
    obtener_lineas_horizontales, bound_and_crop, crop_and_save_subpreg, get_pregunta_inicial_pagina, save_pregunta_completa, \
    partir_imagen_por_mitad, get_contornos_grandes, dejar_solo_recuadros_subpregunta, get_mascara_lineas_horizontales
from dotenv import load_dotenv
load_dotenv()
from simce.utils import timing
import argparse
from pathlib import Path
from multiprocessing import Queue
VALID_INPUT = {'cuadernillo', 'pagina'}

#files = [i.name for i in Path('data/input_raw').rglob('*.jpg')]
## procesamiento imagen ----------------------------------

def process_single_image(preguntas:pd.Series, num: int, rbd:PathLike, dic_pagina:dict,
                          n_pages: int, subpreg_x_preg: dict, 
                         dir_subpreg:PathLike, tipo_cuadernillo:str, regex_estudiante:str, 
                         args:argparse.Namespace, queue:Queue):
    
    """
    Genera el recorte de una pregunta/subpregunta. Primero se verifica que la imagen no posea franjas negras en
    ningún costado del cuestionario. Posteriormente, se divide la imagen en dos páginas del cuadernillo y
    se selecciona la página de la pregunta a recortar. En esta, se detectan los recuadros de las preguntas y
    se realiza un primer recorte según las coordenadas obtenidas. Luego se identifican los recuadros de respuesta y
    se vuelve a cortar la imagen.
    
    Si la pregunta seleccionada no posee subpreguntas, se obtiene el recorte de recuadros de respuesta 
    y se guarda la imagen.

    Si la pregunta seleccionada posee subpreguntas, se identifican las líneas horizontales separadoras de subpreguntas
    y se procede a guardar la imagen con la sección de la subpregunta de interés. 

    **No retorna nada**

    Args:
        preguntas: lista con preguntas a recortar
        num: ID de pregunta a recortar
        rbd: Ruta de la pregunta a recortar
        dic_pagina: Diccionario con mapeo de preguntas en la página del cuestionario
        n_pages: Cantidad de páginas que posee el cuestionario en total
        subpreg_x_preg: Insumo con cantidad de subpreguntas por pregunta.
        dir_subpreg: Directorio general en donde se guardarán las preguntas recortadas 
        tipo_cuadernillo: define si se está procesando para estudiantes o padres.
        regex_estudiante: Expresión regular que nos ayuda a identificar el n° de serie del cuestionario
        queue: Cola de tareas gestionada por Manager() para intercambiar datos entre procesos de forma segura.

        
        
    """
    # from pathlib import Path
    # df99 = df99[df99['serie'] == 4077894] #data\input_raw\CP\02748\4077894_4.jpg
    # num = 1
    # rbd = Path('data/input_raw/CP/02748/4077894_4.jpg')
    
    pregunta_selec = re.search(r'p(\d{1,2})', preguntas.iloc[num]).group(0)          # seleccion de pregunta
    estudiante = re.search(f'({regex_estudiante})', str(rbd)).group(1)               # serie estudiante
    pagina_pregunta = dic_pagina[pregunta_selec]                                     # ubicacion pregunta
    pages = get_pages_cuadernillo(pagina_pregunta, n_pages)
    
    dir_subpreg_rbd = (dir_subpreg /f'{rbd.parent.parent.name}'/ f'{rbd.parent.name}')  # obtencion path pregunta
    
    dir_subpreg_rbd.mkdir(exist_ok=True, parents=True)

    if not rbd.is_file():
        preg_error = dir_subpreg_rbd / f'{estudiante}'
        agregar_error(queue= queue,
                      pregunta=str(preg_error),
                      error=f'No existen archivos disponibles para serie {preg_error.name}',
                      nivel_error=tipo_cuadernillo)
        return 'Ocurrió un error: archivo no existe'

    file = rbd.name

    # if file not in files:
    #     return ''

    if args.verbose:
        print(f'{file=}')
    # Leemos imagen
    img_completa = cv2.imread(str(rbd), 1) 
    img_completa_crop = recorte_imagen(img_completa, 0, 150, 50, 160)
    
    # Eliminamos franjas negras en caso de existir
    img_completa_sin_franja = eliminar_franjas_negras(img_completa_crop)
    
    # Divimos imagen en dos páginas del cuadernillo
    paginas_cuadernillo = partir_imagen_por_mitad(img_completa_sin_franja)
    
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

                img_recuadros_pregunta = dejar_solo_recuadros_subpregunta(img_pregunta)
                
                # Exportamos pregunta si no tiene subpreguntas:
                if subpreg_x_preg[pregunta_selec] == 1:
                    save_pregunta_completa(img_recuadros_pregunta, dir_subpreg_rbd, estudiante, pregunta_selec)
                    
                    return 'Éxito!'

                subpreg_selec = preguntas.iloc[num].split('_')[1]
                if args.verbose:
                    print(f'{subpreg_selec=}')
                
                # Obtenemos subpreguntas:
                #img_pregunta_crop = recorte_imagen(img_pregunta)
                # img_crop_col = get_mask_imagen(img_pregunta_recuadros,
                #                                lower_color=np.array(
                #                                    [0, 111, 109]),
                #                                upper_color=np.array([18, 255, 255]))

                # Obtenemos lineas horizontales:
                mask_lineas_horizontales = get_mascara_lineas_horizontales(img_recuadros_pregunta)
                
                lineas_horizontales = obtener_lineas_horizontales(
                    mask_lineas_horizontales, n_pixeles_entre_lineas=n_pixeles_entre_lineas,
                      minLineLength=np.round(mask_lineas_horizontales.shape[1] * .6))
                #print(lineas_horizontales)
                
                n_subpreg = len(lineas_horizontales) - 1

                if n_subpreg != subpreg_x_preg[pregunta_selec]:
                    preg_error = str(dir_subpreg_rbd / f'{estudiante}')
                    dic_dif = get_subpregs_distintas(subpreg_x_preg, dir_subpreg_rbd, estudiante)
                    error = f'N° de subpreguntas incorrecto para serie {estudiante}, se encontraron {n_subpreg} subpreguntas {dic_dif}'
                    agregar_error(queue= queue, pregunta=preg_error, error=error, nivel_error=tipo_cuadernillo)
            
                # Realizamos recorte y guardado de subpregunta
                try:
                    file_out = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}.jpg')
                    crop_and_save_subpreg(img_recuadros_pregunta, lineas_horizontales,
                                          i=int(subpreg_selec)-1, file_out=file_out, verbose=args.verbose)
                
                # Si hay error en procesamiento subpregunta
                except Exception as e:
                    print(e)
                    preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}_{int(subpreg_selec)}')
                    agregar_error(queue= queue,
                                pregunta=preg_error, 
                                error='Subregunta no pudo ser procesada',
                                nivel_error='Subpregunta', 
                                )
                    return 'Ups, ocurrio un error en la subpregunta'


            except Exception as e:
                print(e)
                preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}')
                agregar_error(queue= queue, pregunta=preg_error, error='Pregunta no pudo ser procesada', nivel_error='Pregunta')
                return
            
        except Exception as e:
            print('Ocurrió un error con la máscara')
            print(e)
            preg_error = str(dir_subpreg_rbd / f'{estudiante}_{pregunta_selec}')
            agregar_error(queue= queue, pregunta=preg_error, error='Ocurrio un error con la mascara', nivel_error='Pregunta')
            
    print('Éxito!')


## división en bloques --------------------

def process_image_block(image_block:list):
    """
    Envía a cada bloque la función [process_single_image()](../paralelizacion#simce.paralelizacion.process_single_image),
    la que se encarga de realizar los recortes a cada una de las imágenes disponibles en los bloques. **No retorna nada**
    
    Args:
        image_block: lista con los objetos a usar en cada uno de los bloques, posee los insumos, 
            path de las imagenes y queue.

    """
    queue, preguntas, directorio_imagenes, dic_pagina, n_pages, \
          subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante, args = image_block

    for num, rbd in enumerate(directorio_imagenes):
        process_single_image(preguntas, num, rbd, dic_pagina, n_pages,
                             subpreg_x_preg, dir_subpreg, tipo_cuadernillo, regex_estudiante, args,
                             queue)


@timing
def process_general(dirs:dict[str, PathLike], regex_estudiante: str, queue:Queue, curso: str, tipo_cuadernillo: str,
                    args:argparse.Namespace, filter_rbd: None|str|list[str]= None, 
                    filter_estudiante: None|str|list[str]= None):
    
    """
    Genera el recorte de preguntas/subpreguntas, de forma paralelizada, para los registros obtenidos de la 
        [función que genera tabla de imágenes con dobles marcas](../proc_tabla_99#simce.proc_tabla_99.get_tablas_99).
        Para ello, se utilizan los insumos generados en
        la [función de obtención de insumos](../generar_insumos_img#simce.generar_insumos_img.generar_insumos),
        se cuenta 
        el total de CPUs para realizar el procesamiento simultáneo, se dividen las preguntas equitativamente
        para que cada núcleo posea aproximadamente la misma cantidad de preguntas a recortar, y finalmente
        se aplica la [función que procesa bloque de imágenes](../paralelizacion#simce.paralelizacion.process_image_block)
          para realizar el procesamiento de cada pregunta en los diferentes 
        bloques disponibles (se le aplica la misma funcion de procesamiento en cada uno de los bloques).
        **No retorna nada**.
    
    Args:
        dirs: Diccionario con los directorios a usar

        regex_estudiante: Expresion regular que nos ayuda a identificar el n° de serie del cuestionario 

        queue: Cola de tareas gestionada por Manager() para intercambiar datos entre procesos de forma segura.

        curso: Nombre carpeta que identifica el curso en procesamiento.

        tipo_cuadernillo: define si se está procesando para estudiantes o padres. Esto también se utiliza para definir las rutas a consultar
        args: argumentos enviados desde la línea de comandos.

        filter_rbd: permite filtrar uno o más RBDs específicos y solo realizar la operación sobre estos.

        filter_estudiante: permite filtrar uno o más estudiantes específicos y solo realizar la operación sobre estos.

    """
    
    
    print(f'Procesando cuadernillo {tipo_cuadernillo}')
    


    nombre_tabla_casos99 = f'casos_99_compilados_{curso}_{tipo_cuadernillo}.csv'
    df99 = pd.read_csv(dirs['dir_tabla_99'] / nombre_tabla_casos99, dtype={'rbd_ruta': 'string'}).sort_values('ruta_imagen')



    if filter_rbd:

        df99 = df99[(df99.rbd_ruta.isin(filter_rbd))]

    if filter_estudiante:
        if isinstance(filter_estudiante, int):
            filter_estudiante = [filter_estudiante]
        df99 = df99[df99.serie.isin(filter_estudiante)]
        
    df99.ruta_imagen = df99.ruta_imagen.str.replace('\\', '/')
    dir_preg99 = [dirs['dir_img_bruta'] / i for i in df99.ruta_imagen]

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
                     subpreg_x_preg, dirs['dir_subpreg'], tipo_cuadernillo, regex_estudiante, args) for i in range(0, len(dir_preg99), block_size)]

    # Usar multiprocessing Pool
    with Pool(num_workers) as pool:
        print('-------------')
        pool.map(process_image_block, image_blocks)

    print('Éxito!')

