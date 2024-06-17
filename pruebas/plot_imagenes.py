# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:20:44 2024

@author: jeconchao
"""
from PIL import Image

# %%
  folder = '09954'

   for folder in Path('data/output/').iterdir():

        s = pd.Series([re.match(r'\d+', i.name).group(0) for i in folder.iterdir()])
        s2 = pd.Series([re.search(r'p\d{1,2}', i.name).group(0)
                       for i in folder.iterdir()])
        s3 = pd.Series(
            [re.search(r'p\d{1,2}_\d{1,2}', i.name).group(0) for i in folder.iterdir()])
        df_check = pd.concat([s.rename('id_est'), s2.rename('preg'),
                              s3.rename('subpreg')], axis=1)

        n_est = df_check.id_est.nunique()
        subpregs = df_check.groupby('subpreg').id_est.count()

        df_check.groupby('id_est').preg.value_counts()

        nsubpreg_x_alumno = s.value_counts()

        if not nsubpreg_x_alumno[nsubpreg_x_alumno.ne(165)].empty:
            print(f'RBD {folder.name}:\n')
            print(nsubpreg_x_alumno[nsubpreg_x_alumno.ne(165)])
            print(subpregs[subpregs.ne(n_est)])
            print('\n')

    # %%

    e3 = Path('data/output')

    for n, i in enumerate(e3.rglob('*')):
        pass

    # %%

    cv2.imshow("Detected Lines", img_rptas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # %%

    # hsv_img = cv2.cvtColor(mask_naranjo,  cv2.COLOR_GRAY2BGR)
    # hsv_img = cv2.drawContours(mask_naranjo, big_contours, -1, (60, 200, 200), 3)
    cv2.imshow("Detected Lines", cv2.resize(im2, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# %%

    big_contours
    img_pregunta = bound_and_crop(media_img, c)

# %%

    img_crop = proc.recorte_imagen(cropped_img)
    img_crop_col = proc.procesamiento_color(img_crop)

    puntoy = proc.obtener_puntos(img_crop_col)

    for i in range(len(puntoy)-1):
        print(i)
        cropped_img_sub = img_crop[puntoy[i]:puntoy[i+1],]

        cv2.imshow("Detected Lines", cropped_img_sub)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # def apply_approx(cnt):

    #     epsilon = 0.45*cv2.arcLength(cnt,True)
    #     approx = cv2.approxPolyDP(cnt,epsilon,True)
    #     return approx

    # %%

    cv2.imshow("Detected Lines", cv2.resize(cropped_img, (900, 900)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# %%
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
def plot(imgs, col_title=None, titles=None, **imshow_kwargs):
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


import pandas as pd
from config.proc_img import dir_tabla_99, dir_input
from PIL import Image
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt

padres99 = f'casos_99_entrenamiento_compilados_padres.csv'
est99 = f'casos_99_entrenamiento_compilados_estudiantes.csv'
df99p = pd.read_csv(dir_tabla_99 / padres99)
import cv2
img = Image.open('data/input_proc/subpreg_recortadas/base/CP/01790/4051000_p8.jpg')

df99p.columns

def addnoise(input_image, noise_factor = 0.1):
    """transforma la imagen agredando un ruido gausiano"""
    inputs = transforms.ToTensor()(input_image)
    noise = inputs + torch.rand_like(inputs) * noise_factor
    noise = torch.clip (noise,0,1.)
    output_image = transforms.ToPILImage()
    image = output_image(noise)
    return image
def transform_img(orig_img):
    """Transformaciones a imagen para aumento de casos de sospecha de doble marca en entrenamiento"""

    trans_img = transforms.RandomHorizontalFlip(0.7)(orig_img)  # Randomly flip
    trans_img = transforms.RandomVerticalFlip(0.7)(trans_img)
    trans_img = transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)
    trans_img = addnoise(trans_img)
    trans_img = transforms.GaussianBlur(kernel_size=(3, 5), sigma=(1, 2))(trans_img)

    return trans_img


padded_imgs = [transform_img(img) for padding in (3, 10, 30, 50)]

for i in range(10):
    row = train[train.falsa_sospecha.eq(1)].iloc[i]
    img = Image.open(row.ruta_imagen_output)
    #img2 = Image.open((dir_input / row.ruta_imagen.replace('\\', '/')))
    plot([img], col_title=[row.ruta_imagen_output] )
    #plt.title(train[train.falsa_sospecha.eq(1)].ruta_imagen_output.iloc[i])
    plt.show()