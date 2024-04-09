# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:28:56 2024

@author: jeconchao
"""

from simce.config import dir_est
from itertools import chain
import numpy as np
import cv2

p1 = list(chain.from_iterable([[j for j in i.iterdir() if '_1' in j.name] for i in dir_est.iterdir()]))
p1_sample = p1[:10]


dst = p1_sample[0]
for i in range(len(image_data)):
    if i == 0:
        pass
    else:
        alpha = 1.0/(i + 1)
        beta = 1.0 - alpha
        dst = cv2.addWeighted(image_data[i], alpha, dst, beta, 0.0)
 
# Save blended image
cv2.imwrite('weather_forecast.png', dst)