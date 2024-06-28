# -*- coding: utf-8 -*-
"""
Created on Mon May 27 12:20:44 2024

@author: jeconchao
"""


import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
from pathlib import Path
import torch
dir_input = Path('data/input_raw')
def plot(imgs, col_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    # print(f'{num_rows=}')
    # print(f'{num_cols=}')
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            
            img = transforms.ToImage()(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = transforms.ToDtype(torch.uint8, scale=True)(img)
            
            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
 

    if col_title is not None:
        for col_idx in range(num_cols):
            print(col_title[col_idx])
            axs[0, col_idx].set_title(col_title[col_idx])

    plt.tight_layout()


#rev = pd.read_excel('data/otros/datos_a_revisar.xlsx')
rev = pd.read_excel('data/otros/datos_revisados.xlsx')
rev.origen.value_counts()
rev['diferente'] = rev.etiqueta_original != rev.etiqueta_final
rev.groupby('origen').diferente.sum().div(rev.groupby('origen').diferente.count()).sort_values(ascending=False)
nombre_encargado = 'juane'
#mi_rev = rev[rev.encargado.eq(nombre_encargado)]
#mi_rev = rev[rev.comentarios.notnull()]
mi_rev = rev[rev.origen.eq('doble_marca_normal') ]
#mi_rev2 = mi_rev[mi_rev.etiqueta_final.eq('-')]
for i in range(len(mi_rev)):
    if i >= 0:
        print(i)
        
        row = mi_rev.iloc[i]
        print(row.origen)
        try:
            img = Image.open(row.ruta_imagen_output)
            
            img2 = Image.open((dir_input / row.ruta_imagen.replace('\\', '/')))
        
            plot([img, img2], col_title=[row.ruta_imagen_output, row.ruta_imagen] )
            #plt.title(train[tr-ain.falsa_sospecha.eq(1)].ruta_imagen_output.iloc[i])
            plt.show()
        except Exception as e:
            print(e)
            pass


