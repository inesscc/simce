from config.proc_img import get_directorios, regex_estudiante, CURSO
from simce.proc_imgs import select_directorio
from simce.paralelizacion import process_general
from simce.errors import escribir_errores
from time import time
from multiprocessing import Manager

if __name__ == "__main__":
    dirs = get_directorios()
    tipo_cuadernillo = 'padres'
    directorio_imagenes = select_directorio(tipo_cuadernillo, directorios=dirs)
    para_entrenamiento = True  
    muestra = False  
    filter_rbd = None  
    filter_rbd_int = None 
    filter_estudiante = None  

    
    manager = Manager()
    queue = manager.Queue()
    
    inicio = time.time()
    process_general(directorio_imagenes, dirs, para_entrenamiento, 
                    regex_estudiante, muestra, filter_rbd, filter_rbd_int, 
                    filter_estudiante,queue, curso=CURSO,  tipo_cuadernillo='estudiantes')
    
    fin = time.time() - inicio
    print(f"Tiempo de procesamiento: {fin:.2f}")
    
    inicio2 = time.time()
    escribir_errores(queue)
    fin2 = time.time() - inicio2
    print(f"Tiempo en escribir errores: {fin2:.2f}")