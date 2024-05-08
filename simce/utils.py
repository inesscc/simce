# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:54:35 2024

@author: jeconchao
"""


from simce.config import dir_data
from os import getcwd, scandir
from os.path import abspath


def crear_directorios():

    (dir_data / 'output').mkdir(exist_ok=True, parents=True)

    (dir_data / 'input/cuestionario_estudiantes').mkdir(exist_ok=True, parents=True)
    (dir_data / 'input/cuestionario_padres').mkdir(exist_ok=True)


def ls(ruta=getcwd()):
    """Funcion para obtener la ruta de los archivos dentro de la carpeta indicada."""
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]
