import cv2
import numpy as np
from simce.utils import preparar_mascaras, eliminar_o_rellenar_manchas

import numpy as np
import pandas as pd
from os import PathLike


    


def calcular_indices_tinta(ruta:str|PathLike)-> tuple[list[float, float], list[float, float]]:
    """
    Calcula índices de tinta para una subpregunta específica.

    Args:
        ruta: ruta de la imagen a la que se le calcularán los indicadores
    
    Returns:
        indices_relevantes: lista con indicador de porcentaje de tinta de primer y segundo recuadros más altos.
         
        intensidades_relevantes: lista con indicador de intensidad de tinta de primer y segundo recuadros más altos.
    
    """

    bordered_mask, bordered_rect_img = preparar_mascaras(ruta)

    contours, _ = cv2.findContours(bordered_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    big_contours = [
        i for i in contours if 250 < cv2.contourArea(i) < 2600 ]
    #bgr_img = cv2.imread(ruta)[20:-20, 15:-15]


    porcentajes_tinta = []
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
        
        #cv2.rectangle(bordered_rect_img, (x, y), (x+w, y+h), 0, 3)
        img_crop = bordered_rect_img[y+3:y+h-3, x+3:x+w-3]
        idx_blanco = np.where(img_crop > 0.9)
        img_crop[idx_blanco] = 1
        

        intensidad_promedio = 1- img_crop[img_crop != 1].mean()
        indice = 1 - img_crop.mean()
        porcentajes_tinta.append(np.round(indice, 3))
        intensidades.append(np.round(intensidad_promedio, 3))

    


    porcentajes_relevantes = sorted(porcentajes_tinta, reverse = True)[:2]
    intensidades_relevantes =   sorted([i for i in intensidades if not pd.isna(i)], reverse = True)[:2]

    return porcentajes_relevantes, intensidades_relevantes

def get_indices_tinta_total(dirs: dict[str, PathLike]):
    """
    Toma tabla de predicciones y procede a calcular índices de tinta, que agrega a la tabla y luego
    exporta una tabla final. Los indicadores calculados son:
     
      - Ratio porcentaje de tinta: ratio entre el porcentaje relleno del recuadro con más tinta
        y el segundo recuadro con más tinta, para una subpregunta dada
       
      - Ratio de intensidad de tinta: ratio entre la intensidad promedio de la tinta del recuadro más intenso 
      y el segundo más intenso, para una subpregunta dada
       
    **No retorna nada**

    Args:
        dirs: diccionario de directorios del proyecto

    
    """
    preds = pd.read_parquet(dirs['dir_predicciones'] / 'predicciones_modelo.parquet')
    preds['indices'] = preds.ruta_imagen_output.apply(lambda x: calcular_indices_tinta(x))
    preds = preds.reset_index(drop=True)
    split = pd.DataFrame(preds['indices'].tolist(), columns = ['indice_tinta', 'indice_intensidad'])
    preds_final = preds.copy()
    for col in split.columns:
        col_split = pd.DataFrame(split[col].tolist(), columns = [f'{col}_top1', f'{col}_top2'])
        preds_final = pd.concat([preds_final, col_split], axis = 1)
    preds_final['ratio_tinta'] = preds_final.indice_tinta_top1 / preds_final.indice_tinta_top2
    preds_final['ratio_intensidad'] = preds_final.indice_intensidad_top1 / preds_final.indice_intensidad_top2


    preds_final.to_excel(dirs['dir_predicciones'] / 'predicciones_modelo_final.xlsx')

    print('Predicciones con insumos posteriores exportadas exitosamente!')

