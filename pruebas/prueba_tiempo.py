import os
import cv2
from time import time

n = time()
img = cv2.imread('data/input_raw/CE/00003/4000081_1.jpg')
print(time() - n)