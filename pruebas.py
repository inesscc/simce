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
from simce.utils import dic_img_preg
import pandas as pd

# p1 = list(chain.from_iterable(
#     [[str(j) for j in i.iterdir() if '_1' in j.name] for i in dir_est.iterdir()]))
# p1_sample = p1[:100]




# p1_sample_images = [cv2.imread(img,1) for img in p1_sample]
# dims_minimas = np.array([[j for j in i.shape]  for i in p1_sample_images]).min(axis=0)[:2]

# p1_sample_images_resize = [cv2.resize(img, (dims_minimas[0], dims_minimas[1])) 
#                            for img in p1_sample_images ]

#%%





#%%



now = time()
e1 = Path('data/input/cuestionario_estudiantes/09955/')
for preg in (e1.iterdir()):
    file_dir = re.sub(r'data\\input\\cuestionario_estudiantes\\', '', str(preg))
    folder, file = file_dir.split('\\')
    file_no_ext = Path(file).with_suffix('')
    id_est = re.search('\d+',f'{file_no_ext}').group(0)
    Path(f'data/output/{folder}').mkdir(exist_ok=True)
    page = str(file_no_ext)[-1]
        
        

   # if str(preg) == 'data\\input\\cuestionario_estudiantes\\09955\\4272452_3.jpg':
        
     

       # print(preg)
    img_preg = cv2.imread(str(preg),1)
    
    x,y = img_preg.shape[:2]
    img_crop = img_preg[40:x - 200, 50:y-160]

    punto_medio = int(np.round(img_crop.shape[1] / 2, 1))
    
    img_p1 = img_crop[:, :punto_medio] 
    img_p2 = img_crop[:, punto_medio:]
    
    n = 0
    for media_img in [img_p1, img_p2]:
        print(media_img.shape)
        
        gray = cv2.cvtColor(media_img, cv2.COLOR_BGR2GRAY) #convert roi into gray
        Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
       # Canny=cv2.Canny(Blur,10,50) #apply canny to roi
        _,It = cv2.threshold(Blur,0,255,cv2.THRESH_OTSU)
        sx = cv2.Sobel(It,cv2.CV_32F,1,0)
        sy = cv2.Sobel(It,cv2.CV_32F,0,1)
        m = cv2.magnitude(sx,sy)
        m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
        m = cv2.ximgproc.thinning(m,None,cv2.ximgproc.THINNING_GUOHALL)
        
        #Find my contours
        contours =cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
        big_contours = [i for i in contours if cv2.contourArea(i) > 30000]
        big_contours_sizes = [cv2.contourArea(i) for i in big_contours]
        
        big_contours_sort = [i for _, i in sorted(zip(big_contours_sizes, big_contours))]
        
        
        

    
        for c in (big_contours_sort):
            print(n)
            x,y,w,h= cv2.boundingRect(c)
            cropped_img=media_img[y:y+h, x:x+w]
            

            id_img = f'{page}_{n}'
            n += 1
            file_out = f'data/output/{folder}/{id_est}_{dic_img_preg[id_img]}.jpg'
            print(file_out)
            cv2.imwrite(file_out, cropped_img)
            
      #  break



print(time() - now)

#%%

n = 0

gray = cv2.cvtColor(img_p1, cv2.COLOR_BGR2GRAY) #convert roi into gray
_,It = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
sx = cv2.Sobel(It,cv2.CV_32F,1,0)
sy = cv2.Sobel(It,cv2.CV_32F,0,1)
m = cv2.magnitude(sx,sy)
m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
m = cv2.ximgproc.thinning(m,None,cv2.ximgproc.THINNING_GUOHALL)
#Blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
#Canny=cv2.Canny(Blur,10,50) #apply canny to roi

#Find my contours
contours =cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]

big_contours = [i for i in contours if cv2.contourArea(i) > 25000]
big_contours_sizes = [cv2.contourArea(i) for i in big_contours]


big_contours_sort = [i for _, i in sorted(zip(big_contours_sizes, big_contours))]

    
cv2.imshow('rpueba',cv2.resize(m, (900, 900)))
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 


#%% Cropped images

gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY) #convert roi into gray
blur=cv2.GaussianBlur(gray,(5,5),1) #apply blur to roi
canny=cv2.Canny(blur,10,50) #apply canny to roi
sx = cv2.Sobel(canny,cv2.CV_32F,1,0)
sy = cv2.Sobel(canny,cv2.CV_32F,0,1)
m = cv2.magnitude(sx,sy)
m = cv2.normalize(m,None,0.,255.,cv2.NORM_MINMAX,cv2.CV_8U)
m = cv2.ximgproc.thinning(m,None,cv2.ximgproc.THINNING_GUOHALL)
 # Threshold using Otsu's method
#_, thresholded = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY)

# Detect lines using HoughLinesP
lines = cv2.HoughLinesP(m, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)


#%%



# Display the result
cv2.imshow("Detected Lines", cv2.resize(m, (900, 900)))
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

for c in (big_contours_sort):


    print(n)
    x,y,w,h= cv2.boundingRect(c)
    cropped_img=img_p1[y:y+h, x:x+w]
    

    id_img = f'{page}_{n}'
    n += 1
    file_out = f'data/output/{folder}/{id_est}_{dic_img_preg[id_img]}.jpg'
    print(file_out)
    cv2.imwrite(file_out, cropped_img)



#%%

cv2.imshow('rpueba',cv2.resize(cropped_img, (900, 900)))
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 

#%%

e2 = Path(f'data/output/{folder}')
pd.Series([re.match('\d+', i.name).group(0) for i in e2.iterdir()]).value_counts()
#%%


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
