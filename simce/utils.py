# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:54:35 2024

@author: jeconchao
"""


from os import getcwd, scandir
from os.path import abspath
import cv2
import numpy as np
import pandas as pd
# from config.proc_img import dir_estudiantes, dir_subpreg, dir_tabla_99, dir_input, dir_padres, dir_insumos
import json
import torch
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from functools import wraps
from time import time
from PIL import Image
import random

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' %
              (f.__name__, args, kw, te-ts))
        return result
    return wrap





def ls(ruta=getcwd()):
    """Funcion para obtener la ruta de los archivos dentro de la carpeta indicada."""
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]


def get_mask_imagen(media_img, lower_color=np.array([13, 11, 0]), upper_color=np.array([29, 255, 255]),
                    iters=4, eliminar_manchas='horizontal', revert=False):
    """
    Genera una máscara binaria para una imagen dada, basada en un rango de color en el espacio de color HSV.

    Args:
    media_img (np.ndarray): La imagen de entrada en formato BGR.
    lower_color (np.ndarray, optional): El límite inferior del rango de color en formato HSV. Por defecto es np.array([13, 31, 0]), que corresponde al color naranjo.
    upper_color (np.ndarray, optional): El límite superior del rango de color en formato HSV. Por defecto es np.array([29, 255, 255]), que corresponde al color naranjo.

    Returns:
    mask (numpy.ndarray): Una máscara binaria donde los píxeles de la imagen que están dentro del rango de color especificado son blancos, y todos los demás píxeles son negros.
    """
    # Convierte la imagen de entrada de BGR a HSV
    hsv = cv2.cvtColor(media_img, cv2.COLOR_BGR2HSV)

    # Crea una máscara binaria donde los píxeles de la imagen que están dentro del rango de color
    # especificado son blancos, y todos los demás píxeles son negros.
    mask = cv2.inRange(hsv, lower_color, upper_color)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=iters)

    if eliminar_manchas:
        
        if eliminar_manchas == 'vertical':
            mask = eliminar_o_rellenar_manchas(mask, orientacion='vertical', limite=50)
        elif eliminar_manchas == 'horizontal':

            mask = eliminar_o_rellenar_manchas(mask, orientacion='horizontal', limite=100)

        else:
            return print('Valor inválido para eliminar manchas')
    if revert:
        mask = cv2.bitwise_not(mask)

    return mask


def eliminar_o_rellenar_manchas(mask, orientacion, limite, rellenar=False):
        
        mask = mask.copy()
        
        if rellenar:
            val_replace = 255
        else:
            val_replace = 0

    
        if orientacion == 'vertical':
            axis = 0
            # Calculamos la media de cada columna
            mean_col = mask.mean(axis=axis)

            if rellenar:
                comparison = mean_col > limite
            else:
                comparison = mean_col < limite
            # Si la media es menor a 100, reemplazamos con 0 (negro):
            # Esto permite eliminar manchas de color que a veces se dan
            idx_low_rows = np.where(comparison)[0]
            mask[:, idx_low_rows] = val_replace
        elif orientacion == 'horizontal':
            axis = 1
            # Calculamos la media de cada fila:
            mean_row = mask.mean(axis=axis)

            if rellenar:
                comparison = mean_row >= limite
            else:
                comparison = mean_row <= limite
            # Si la media es menor a 100, reemplazamos con 0 (negro):
            # Esto permite eliminar manchas de color que a veces se dan
            idx_low_rows = np.where(comparison)[0]
            mask[idx_low_rows, :] = val_replace
        else:
            return print('Valor inválido para eliminar manchas')
        
        return mask

class RandomRotation:
    def __init__(self, degrees, p):
        self.degrees = degrees
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            if self.degrees % 90 == 0:  # Only handle multiples of 90 degrees
                rotations = (self.degrees // 90) % 4
                for _ in range(rotations):
                    x = x.transpose(Image.ROTATE_90)
            else:
                print("This implementation only supports rotations that are multiples of 90 degrees.")
        return x

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
