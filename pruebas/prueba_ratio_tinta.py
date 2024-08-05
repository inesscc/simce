import cv2
from PIL import Image
from config.proc_img import get_directorios
from simce.proc_imgs import bound_and_crop
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
from torchvision import tv_tensors
import numpy as np
from simce.utils import get_mask_imagen
def plot(imgs, col_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    print(f'{num_rows=}')
    print(f'{num_cols=}')
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = transforms.ToImage()(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = transforms.ToDtype(torch.uint8, scale=True)(img)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
 

    if col_title is not None:
        for col_idx in range(num_cols):
            print(col_title[col_idx])
            axs[0, col_idx].set_title(col_title[col_idx])

    plt.tight_layout()


def get_recuadros(mask):
     # Define the border width in pixels
    top, bottom, left, right = [3]*4

    # Create a border around the image
    bordered_mask = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=0).astype(np.uint8)

    



    contours, _ = cv2.findContours(bordered_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    big_contours = [
        i for i in contours if 600 < cv2.contourArea(i) < 2000 ]


    for contour in big_contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        cv2.rectangle(bordered_mask, (x, y), (x+w, y+h), 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bordered_mask2 = cv2.morphologyEx(bordered_mask, cv2.MORPH_CLOSE, kernel, iterations=3)


    contours2, _ = cv2.findContours(bordered_mask2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    big_contours2 = [
        i for i in contours2 if 600 < cv2.contourArea(i) < 2000 ]

    
    for contour in big_contours2:
        x, y, w, h = cv2.boundingRect(contour)
        
        cv2.rectangle(bordered_mask2, (x, y), (x+w, y+h), 255, -1)

    if len(big_contours2) >= len(big_contours):

        return bordered_mask2, big_contours2
    else:

        return bordered_mask, big_contours
    


ruta = test_final[test_final.ratio_tinta.isnull()].iloc[4].ruta_imagen_output
ruta = test_final.sort_values('indice_tinta_top1').iloc[-10].ruta_imagen_output
'data/input_proc/4b/subpreg_recortadas/base/CP/01168/4028778_p1.jpg'
def get_indices_tinta(ruta):

    #bgr_img = cv2.imread(ruta)[20:-20, 15:-15]
    bgr_img = cv2.imread(ruta)

    mask_blanco = get_mask_imagen(bgr_img, lower_color=np.array([0,31,0]),
                                  upper_color=np.array([179, 255, 255]), iters=1,
                                    eliminar_manchas=False, revert=True)
    
    mask_blanco_fill, contornos_og = get_recuadros(mask_blanco)

    if 'CE' in ruta:
        # Detectamos grises y negros si es cuestionario de estudiantes
        mask_tinta = get_mask_imagen(bgr_img, lower_color=np.array([0,0,225]),
                                    upper_color=np.array([179, 255, 255]), iters=1, eliminar_manchas=False,
                                    revert=True)

    elif 'CP' in ruta:
        # Detectamos tinta azul si es cuestionario de padres
        mask_tinta = get_mask_imagen(bgr_img, lower_color=np.array([67,46,0]),
                                    upper_color=np.array([156, 255, 255]), iters=1, eliminar_manchas=False)
        if mask_tinta.mean() < 0.7:
            # Detectamos grises y negros si no detectamos tinta azul
            mask_tinta = get_mask_imagen(bgr_img, lower_color=np.array([0,0,225]),
                                        upper_color=np.array([179, 255, 255]), iters=1, eliminar_manchas=False,
                                        revert=True)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_tinta = cv2.morphologyEx(mask_tinta, cv2.MORPH_CLOSE, kernel, iterations=2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    mask_tinta = cv2.erode(mask_tinta, kernel, iterations=1)
    
    idx_blanco = np.where(mask_tinta == 255)
    mask_blanco_fill[idx_blanco] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_blanco_fill = cv2.morphologyEx(mask_blanco_fill, cv2.MORPH_CLOSE, kernel, iterations=2)


    axis = 0
    # Calculamos la media de cada columna:
    sum_blanco = np.sum(mask_blanco_fill == 255, axis=axis)

    # Si la media es menor a 100, reemplazamos con 0 (negro):
    # Esto permite eliminar manchas de color que a veces se dan
    idx_low_rows = np.where(sum_blanco < 12)[0]
    mask_blanco_fill[:, idx_low_rows] = 0

    # Creo contorno en torno a contornos originales, para no distorsionar
    for contour in contornos_og:
        x, y, w, h = cv2.boundingRect(contour)
        
        cv2.rectangle(mask_blanco_fill, (x, y), (x+w, y+h), 0, 4)


    # Define the border width in pixels
    top, bottom, left, right = [3]*4

    # Create a border around the image
    bordered_mask = cv2.copyMakeBorder(mask_blanco_fill, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=0).astype(np.uint8)
    img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY) /255
    #bordered_mask = cv2.bitwise_not(bordered_mask)


    contours, _ = cv2.findContours(bordered_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    big_contours = [
        i for i in contours if 250 < cv2.contourArea(i) < 2600 ]
    #len(big_contours)
    rect_img = img.copy()
    bordered_rect_img = cv2.copyMakeBorder(rect_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=1)

    indices = []
    intensidades = []

    for contour in big_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Ajustamos x y h, porque transformaciones están generando recuadros más pequeños
        x = x - 3
        y = y - 3

        ratio_dims = w / h
        if ratio_dims > 5 or ratio_dims < .2:
            continue

        elif  ratio_dims > 1.15 or ratio_dims < .85:
            dif_px = np.abs(w - h) 
            px_cortar = int(np.floor(dif_px/2))
            # Si el ratio es mayor a 1, el recuadro es más ancho de lo que debería:
            if ratio_dims > 1:   
                w = w - px_cortar
                x = x - px_cortar
            # Si el ratio es menor a 1, el recuadro es más alto de lo que debería:
            else:
                h = h - px_cortar
                y = y + px_cortar
        
        cv2.rectangle(bordered_rect_img, (x, y), (x+w, y+h), 0, 3)
        img_crop = bordered_rect_img[y+3:y+h-3, x+3:x+w-3]
        idx_blanco = np.where(img_crop > 0.9)
        img_crop[idx_blanco] = 1
        

        intensidad_promedio = 1- img_crop[img_crop != 1].mean()
        indice = 1 - img_crop.mean()
        indices.append(np.round(indice, 3))
        intensidades.append(np.round(intensidad_promedio, 3))
    print(indices)
    


    indices_relevantes = sorted(indices, reverse = True)[:2]
    intensidades_relevantes =   sorted([i for i in intensidades if not pd.isna(i)], reverse = True)[:2]

    return indices_relevantes, intensidades_relevantes

# Show the resulting contours on the image
ruta
cv2.imshow('Contours', bordered_rect_img)
cv2.waitKey(1) 
cv2.destroyAllWindows()
cv2.waitKey(1) 
ruta

dir_train_test = get_directorios('dir_train_test')
test = pd.read_csv(dir_train_test / 'test.csv')
i = 1

row = test[test.falsa_sospecha.eq(1)].iloc[i]
#ruta = problemas2.ruta_imagen_output.iloc[2]



pd.set_option("display.max_colwidth", None)
#t = train.sample(3000, random_state=42).reset_index(drop=True)
#t['indices_tinta'] = t.ruta_imagen_output.apply(lambda x: get_indices_tinta(x))
#test = test[test.ruta_imagen_output.str.contains('base')]
test['indices'] = test.ruta_imagen_output.apply(lambda x: get_indices_tinta(x))

split = pd.DataFrame(test['indices'].tolist(), columns = ['indice_tinta', 'indice_intensidad'])
test_final = test.copy()
for col in split.columns:
    col_split = pd.DataFrame(split[col].tolist(), columns = [f'{col}_top1', f'{col}_top2'])
    test_final = pd.concat([test_final, col_split], axis = 1)
#test_final.sort_values('indice_intensidad_top2')
#test_final[test_final.indice_intensidad_top2.isnull()].ruta_imagen_output.iloc[0]
#test_final = pd.concat([test, split], axis = 1)
#test_final[['ruta_imagen_output', 'indice_intensidad']]
test_final['ratio_tinta'] = test_final.indice_tinta_top1 / test_final.indice_tinta_top2
test_final['ratio_intensidad'] = test_final.indice_intensidad_top1 / test_final.indice_intensidad_top2
test_final.ratio_intensidad.describe()


test_final[['ratio_indices', 'indice_top1', 'indice_top2']]
test_final.ratio_indices = test_final.ratio_indices.replace(np.inf, -1)
#Oficialmente es doble marca, pero encontramos una marca:
problemas  = test_final[test_final.dm_sospecha.eq(1) & test_final.ratio_indices.eq(-1) & test_final.dm_final.eq(1)]
# Se encuentra solo uno menos recuadro:
problemas2 = test_final[test_final.indice_top2.isnull()]

problemas_total = pd.concat([problemas, problemas2])
problemas_total.to_excel('data/otros/problematicos.xlsx', index=False)
problemas_total.ruta_imagen_output.iloc[0]
train_final.ratio_indices.describe()
train_final[train_final.ratio_indices.ge(1.5)].ruta_imagen_output.iloc[0]

train_filter = train_final[train_final.ratio_indices.ne(-1)]
train_final.ratio_indices.eq(-1).sum()
train_filter[train_filter.ratio_indices.le(1.3) & train_filter.dm_final.eq(0)].ruta_imagen_output.iloc[0]
train_filter.dm_final.value_counts()
train_final[train_final.ratio_indices.notnull()]
train_final.iloc[0]


plot([mask_negro])
plt.show()