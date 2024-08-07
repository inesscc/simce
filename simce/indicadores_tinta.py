import cv2
import numpy as np
from simce.utils import get_mask_imagen
import pandas as pd

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
    



def calcular_indices_tinta(ruta):

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


def get_indices_tinta(dirs):
    preds = pd.read_parquet(dirs['dir_predicciones'] / 'predicciones_modelo.parquet')
    preds['indices'] = preds.ruta_imagen_output.apply(lambda x: calcular_indices_tinta(x))

    split = pd.DataFrame(preds['indices'].tolist(), columns = ['indice_tinta', 'indice_intensidad'])
    preds_final = preds.copy()
    for col in split.columns:
        col_split = pd.DataFrame(split[col].tolist(), columns = [f'{col}_top1', f'{col}_top2'])
        preds_final = pd.concat([preds_final, col_split], axis = 1)
    preds_final['ratio_tinta'] = preds_final.indice_tinta_top1 / preds_final.indice_tinta_top2
    preds_final['ratio_intensidad'] = preds_final.indice_intensidad_top1 / preds_final.indice_intensidad_top2


    preds_final.to_excel(dirs['dir_predicciones'] / 'predicciones_modelo_final.xlsx')

    print('Predicciones con insumos posteriores exportadas exitosamente!')

