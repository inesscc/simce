import cv2
import os
from time import time
from pathlib import Path
password = 'agencia.2024'
user_name = '.\Agencia'
os.system(rf"NET USE P: \\10.10.100.28\4b_2023" )
os.listdir('P:/CE/00001/4000029_2.jpg')
files = [i.name for i in Path('data/input_raw').rglob('*.jpg')]
len((files))

n = time()
im = cv2.imread('P:/CE/00001/4000029_2.jpg')
print(time() - n)
im
