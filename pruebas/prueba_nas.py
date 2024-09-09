import cv2
import os
from time import time
from pathlib import Path
password = 'agencia.2024'
user_name = '.\Agencia'
os.system(rf"NET USE P: \\10.10.100.28\4b_2023" )
os.listdir('P:/CE/00001/4000029_2.jpg')
files = [i.name for i in Path('data/input_proc/4b/subpreg_recortadas').rglob('*.jpg')]
len((files))

n = time()
im = cv2.imread('P:/CE/00001/4000029_2.jpg')
print(time() - n)
im

import pandas as pd
a = pd.read_parquet('data/output/predicciones/predicciones_modelo.parquet')
b = pd.read_csv('data/input_modelamiento/data_pred.csv')
a.shape
b.shape
b.ruta_imagen.duplicated().sum()


df = pd.read_excel('problemas_datos.xlsx')
df.head(100)