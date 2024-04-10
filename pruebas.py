# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:28:56 2024

@author: jeconchao
"""

from simce.config import dir_est
from itertools import chain
import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from pathlib import Path
from time import time
x_int_left = 193
x_int_right = 3240
y_upp = 37
y_low = 1720

p1 = list(chain.from_iterable(
    [[str(j) for j in i.iterdir() if '_1' in j.name] for i in dir_est.iterdir()]))
p1_sample = p1[:100]




p1_sample_images = [cv2.imread(img,1) for img in p1_sample]
dims_minimas = np.array([[j for j in i.shape]  for i in p1_sample_images]).min(axis=0)[:2]

p1_sample_images_resize = [cv2.resize(img, (dims_minimas[0], dims_minimas[1])) 
                           for img in p1_sample_images ]

#%%




cv2.imshow('rpueba',cv2.resize(img_crop, (1800, 900)))
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 


#%%

def crop_img(img_preg):
    gray = cv2.cvtColor(img_preg, cv2.COLOR_BGR2GRAY) #convert roi into gray
    Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
    Canny=cv2.Canny(Blur,10,50) #apply canny to roi

    #Find my contours
    contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

    contours_sizes = [cv2.contourArea(i) for i in contours]
    max_contour = contours_sizes.index(max(contours_sizes))
    x,y,w,h= cv2.boundingRect(contours[max_contour])
    img_crop=img_preg[y+10:y-10+h, x+10:x-10+w]
    #print(img_crop.shape)
    return img_crop
    
now = time()
e1 = Path('data/input/cuestionario_estudiantes/09952/')
for n, preg in enumerate(e1.iterdir()):
    
    #if str(preg) == 'data\\input\\cuestionario_estudiantes\\09952\\4272352_2.jpg':
         
        page = re.search('_([^_]*)$', preg.with_suffix('').name).group(1)
        
        print(preg)
        img_preg = cv2.imread(str(preg),1)
        
        img_crop = crop_img(img_preg)
        
        

            
        gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) #convert roi into gray
        Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
        Canny=cv2.Canny(Blur,10,50) #apply canny to roi
        
        #Find my contours
        contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
        big_contours_sizes = [cv2.contourArea(i) for i in big_contours]
        
        big_contours_sort = [i for _, i in sorted(zip(big_contours_sizes, big_contours))]
        
        
        
        file_dir = re.sub(r'data\\input\\cuestionario_estudiantes\\', '', str(preg))
    
        for n, c in enumerate(big_contours_sort):
            x,y,w,h= cv2.boundingRect(c)
            cropped_img=img_crop[y:y+h, x:x+w]
            
            folder, file = file_dir.split('\\')
            file_no_ext = Path(file).with_suffix('')
    
            Path(f'data/output/{folder}').mkdir(exist_ok=True)
            cv2.imwrite(f'data/output/{folder}/{file_no_ext}_{n}.jpg',cropped_img)



print(time() - now)


#%%
import pandas as pd
e2 = Path(f'data/output/{folder}')
pd.Series([re.match('\d+', i.name).group(0) for i in e2.iterdir()]).value_counts()

#%%
dst = p1_sample_images_resize[0]
dst.shape
dst_crop = dst[ 50:2700, 130:2070]
dst_crop.shape
#%%
#cv2.imwrite('prueba.png', dst_crop)
img_preg.shape
cv2.imshow('rpueba',cv2.resize(img_crop, (1800, 900)))
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 
#%%
# Load image and HSV color threshold
dst_crop =  cv2.resize(dst_crop, (1800, 900))
original = dst_crop.copy()
image = cv2.cvtColor(dst_crop, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 239], dtype="uint8")
upper = np.array([0, 0, 255], dtype="uint8")
mask = cv2.inRange(image, lower, upper)
detected = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow('detected', detected)
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 

#%%

gray = cv2.cvtColor(dst_crop, cv2.COLOR_BGR2GRAY) #convert roi into gray
Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
Canny=cv2.Canny(Blur,10,50) #apply canny to roi

#Find my contours
contours =cv2.findContours(Canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
big_contours_sizes = [cv2.contourArea(i) for i in big_contours]
big_contours_sort = [i for _, i in sorted(zip(big_contours_sizes, big_contours))]
#%%
file_dir = re.sub(r'data\\input\\cuestionario_estudiantes\\', '', p1_sample[0])

for n, c in enumerate(big_contours_sort):
    x,y,w,h= cv2.boundingRect(c)
    cropped_img=dst_crop[y:y+h, x:x+w]
    
    folder, file = file_dir.split('\\')
    file_no_ext = Path(file).with_suffix('')

    Path(f'data/output/{folder}').mkdir(exist_ok=True)
    cv2.imwrite(f'data/output/{folder}/{file_no_ext}_{n}.jpg',cropped_img)
    

#%%
x,y,w,h= cv2.boundingRect(big_contours_sort[4])
cropped_img=dst_crop[y:y+h, x:x+w]
cv2.imshow('Roi Rect ONLY',cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
cv2.drawContours(dst_crop,contours2,-1,(0,255,0),2)
cv2.imshow('Roi Rect ONLY',dst_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

#Loop through my contours to find rectangles and put them in a list, so i can view them individually later.
cntrRect = []
for i in contours2:
        epsilon = 0.05*cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i,epsilon,True)
        if len(approx) == 4:
            cv2.drawContours(dst_crop,cntrRect,-1,(0,255,0),2)
            cv2.imshow('Roi Rect ONLY',dst_crop)
            cntrRect.append(approx)

cv2.waitKey(0)
cv2.destroyAllWindows()




#%%
# Remove noise
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# Find contours and find total area
cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
area = 0

for c in cnts:
    area += cv2.contourArea(c)
    cv2.drawContours(original,[c], 0, (0,0,0), 2)

print(area)
cv2.imshow('mask', mask)
cv2.imshow('original', original)
cv2.imshow('opening', opening)
cv2.imshow('detected', detected)


cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
cv2.imshow("Python Logo", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%
for n, img in enumerate(p1_sample_images_resize):
    print(img.shape)
    if n != 0:
        alpha = 1.0/(n + 1)
        beta = 1.0 - alpha
        dst = cv2.addWeighted(img, alpha, dst, beta, 0.0)
 
# Save blended image
cv2.imwrite('prueba.png', dst)
