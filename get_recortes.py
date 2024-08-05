from simce.generar_insumos_img import generar_insumos_total
from simce.proc_tabla_99 import get_tablas_99_total
from simce.preparar_modelamiento import gen_pred_set
from simce.utils import crear_directorios
from config.proc_img import get_directorios, regex_estudiante
from simce.paralelizacion import process_general
from simce.errors import escribir_errores
from multiprocessing import Manager
import argparse

def main(args):
    if args.curso:
        CURSO = args.curso
    else:
        from config.proc_img import  CURSO
    directorios = get_directorios(curso=CURSO)
    crear_directorios(directorios)
    # 1.  Generar insumos para procesamiento
    generar_insumos_total(directorios) 
    # 2. Generar tablas con dobles marcas
    get_tablas_99_total(directorios=directorios)

    dirs = get_directorios()
    muestra = False  
    filter_rbd = None  
    filter_rbd_int = None 
    filter_estudiante = None  

    
    manager = Manager()
    queue = manager.Queue()
    

    process_general(dirs['dir_padres'], dirs, 
                    regex_estudiante, muestra, filter_rbd, filter_rbd_int, 
                    filter_estudiante,queue, curso=CURSO,  tipo_cuadernillo='padres')
    
    process_general(dirs['dir_estudiantes'], dirs, 
                regex_estudiante, muestra, filter_rbd, filter_rbd_int, 
                filter_estudiante,queue, curso=CURSO,  tipo_cuadernillo='estudiantes')
    

    escribir_errores(queue)

    gen_pred_set(directorios, curso=CURSO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Módulo de recorte de imágenes. Aquí se obtiene el recorte de cada subpregunta y se generan tablas con las imágenes \
                                   a predecir')
    parser.add_argument('--curso', help='(opcional) identificador del curso a predecir')

    args = parser.parse_args()

    main(args)

