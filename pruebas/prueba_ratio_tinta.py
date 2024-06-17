import cv2
from PIL import Image
from config.proc_img import dir_train_test
from simce.proc_imgs import bound_and_crop
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as transforms
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
import torch
from torchvision import tv_tensors
import numpy as np
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

def is_rectangle_or_square(contour, epsilon=0.02, min_aspect_ratio=0.75, max_aspect_ratio=1.25):
    # Approximate the contour
    approx = cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)
    
    # Check if the approximated contour has 4 vertices
    if len(approx) == 4:
        # Calculate the bounding rectangle
        x, y, w, h = cv2.boundingRect(approx)
        
        # Calculate the aspect ratio
        aspect_ratio = float(w) / h
        if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
            return True
    return False


train = pd.read_csv(dir_train_test / 'train.csv')
i = 1
from config.proc_img import dir_input

a = pd.read_csv(dir_input / 'CP_Final_DobleMarca.csv', sep=';')
a[a.serie.eq(4031281)].p27_1
row = train[train.falsa_sospecha.eq(1)].iloc[i]
ruta = problemas.ruta_imagen_output.iloc[9]

ruta
def get_indices_tinta(ruta):

    bgr_img = cv2.imread(ruta)
    img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY) /255

    mask = img.copy()
    idx_blanco = np.where(mask > 0.95)
    mask[idx_blanco] = 1
    mask[mask != 1] = 0 
    mask = 1 - mask

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    morph2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=1)
 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph1 = cv2.morphologyEx(morph2, cv2.MORPH_OPEN, kernel, iterations=3)




    mask_final = ((1 - morph1)* 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_final, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    big_contours = [
        i for i in contours if  cv2.contourArea(i) > 250 ]
    len(big_contours)
    rect_img = img.copy()
    indices = []

    for contour in big_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(rect_img, (x, y), (x+w, y+h), (0, 0, 0), 3)
        img_crop = bound_and_crop(rect_img, contour, buffer=-3)
        idx_blanco = np.where(img_crop > 0.9)
        img_crop[idx_blanco] = 1
        indice = 1 - img_crop.mean()
        indices.append(indice)
    print(indices)


    indices_relevantes = sorted(indices, reverse = True)[:2]  

    return indices_relevantes

pd.set_option("display.max_colwidth", None)
t = train.sample(2000, random_state=42).reset_index(drop=True)
t['indices_tinta'] = t.ruta_imagen_output.apply(lambda x: get_indices_tinta(x))

train['indices_tinta'] = train.ruta_imagen_output.apply(lambda x: get_indices_tinta(x))

split = pd.DataFrame(t['indices_tinta'].tolist(), columns = ['indice_top1', 'indice_top2'])
train_final = pd.concat([t, split], axis = 1)

train_final['ratio_indices'] = train_final.indice_top1 / train_final.indice_top2
train_final[['ratio_indices', 'indice_top1', 'indice_top2']]
train_final.ratio_indices = train_final.ratio_indices.replace(np.inf, -1)
problemas  = train_final[train_final.dm_sospecha.eq(1) & train_final.ratio_indices.eq(-1) & train_final.dm_final.eq(1)]


train_filter = train_final[train_final.ratio_indices.ne(-1)]
train_final.ratio_indices.eq(-1).sum()
train_filter[train_filter.ratio_indices.le(1.3) & train_filter.dm_final.eq(0)].ruta_imagen_output.iloc[0]
train_filter.dm_final.value_counts()
train_final[train_final.ratio_indices.notnull()]
train_final.iloc[0]
img_crop.mean()
# Show the resulting contours on the image
cv2.imshow('Contours', rect_img)
cv2.waitKey(10) 
cv2.destroyAllWindows()

plot([bgr_img])
plt.show()