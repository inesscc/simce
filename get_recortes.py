from simce.generar_insumos_img import generar_insumos_total
from simce.proc_tabla_99 import get_tablas_99_total
from simce.preparar_modelamiento import gen_pred_set
from simce.utils import crear_directorios, timing
from config.proc_img import get_directorios, regex_estudiante
from simce.paralelizacion import process_general
from simce.errors import escribir_errores
from multiprocessing import Manager
import argparse

@timing
def main(args):
    """
    Función principal que realiza el recorte de las preguntas de forma paralelizada. 
    Primero obtenemos los directorios e insumos a usar, generamos las tablas con dobles marcas y \
        realizamos el recorte de imagenes paralelizado.

    Args:
        args.curso (str): identificador del curso a predecir
    """

    if args.curso:
        CURSO = args.curso
    else:
        from config.proc_img import  CURSO
        
    dirs = get_directorios(curso=CURSO)
    crear_directorios(dirs)

    # 1.  Generar insumos para procesamiento
    generar_insumos_total(dirs, args=args) 
    # 2. Generar tablas con dobles marcas
    get_tablas_99_total(directorios=dirs)

    
    manager = Manager()             # Objeto para gestionar datos compartidos entre procesos
    queue = manager.Queue()         # Cola de tareas
    
    process_general(dirs = dirs, regex_estudiante= regex_estudiante, 
                    queue = queue, curso=CURSO, args=args, tipo_cuadernillo='padres',
                    filter_rbd=['04243','01487'])

    process_general(dirs = dirs, regex_estudiante= regex_estudiante, 
                    queue = queue, curso=CURSO, args=args, tipo_cuadernillo='estudiantes',
                    filter_rbd=['04243','01487'])
    

    escribir_errores(queue)

    gen_pred_set(dirs, curso=CURSO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Módulo de recorte de imágenes. Aquí se obtiene el recorte de cada subpregunta y se generan tablas con las imágenes \
                                   a predecir')
    parser.add_argument('--curso', help='(opcional) identificador del curso a predecir')

    parser.add_argument("-v", "--verbose", help="Se imprime más texto informativo si se activa",
                    action="store_true")

    args = parser.parse_args()

    main(args)

