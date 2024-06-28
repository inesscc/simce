import torch
torch.cuda.is_available()
import torchvision
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from config.proc_img import dir_modelos

print('Finished Training')
from config.proc_img import dir_tabla_99
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os.path
from torchvision.models import list_models

import PIL
import re

import mlflow

### mantenemos la carga de archivos
classification_models = list_models(module=torchvision.models)
padres99 = f'casos_99_entrenamiento_compilados_padres.csv'
est99 = f'casos_99_entrenamiento_compilados_estudiantes.csv'
df99p = pd.read_csv(dir_tabla_99 / padres99)

df99e = pd.read_csv(dir_tabla_99 / est99).sample(frac=.1, random_state=42)
df99 = pd.concat([df99e, df99p]).reset_index(drop=True)
from pathlib import Path
from config import dir_input
faltantes = df99e[~df99e.ruta_imagen.str.replace('\\', '/').apply(lambda x: (dir_input / Path(x)).is_file())].ruta_imagen.to_list()
import pickle
with open("files_faltantes", "wb") as fp:   #Pickling
   pickle.dump(faltantes, fp)


df_exist = df99[df99.ruta_imagen_output.apply(lambda x: Path(x).is_file())].reset_index()
df_exist.dm_sospecha = (df_exist.dm_sospecha == 99).astype(int)
nombre_tabla_casos99 = 'prueba_torch.csv'
df_exist.to_csv(dir_tabla_99 / 'prueba_torch.csv')

df_dup = df_exist[(df_exist['dm_sospecha'] == 1) & (df_exist['dm_final'] == 0)]


#### funciones aux para generar transformaciones

def addnoise(input_image, noise_factor = 0.1):
    """transforma la imagen agredando un ruido gausiano"""
    inputs = transforms.ToTensor()(input_image)
    noise = inputs + torch.rand_like(inputs) * noise_factor
    noise = torch.clip (noise,0,1.)
    output_image = transforms.ToPILImage()
    image = output_image(noise)
    return image

def transform_img(path_img):
    """Transformaciones a imagen para aumento de casos de sospecha de doble marca en entrenamiento"""
    orig_img = PIL.Image.open(path_img)

    trans_img = transforms.RandomHorizontalFlip(0.7)(orig_img)               # Randomly flip
    trans_img = transforms.RandomVerticalFlip(0.7)(trans_img) 
    #trans_img = transforms.ColorJitter(brightness=0, contrast=0.1, saturation=0.003, hue=0.2)(trans_img)
    trans_img = addnoise(trans_img)
    trans_img = transforms.GaussianBlur(kernel_size = (3, 5), sigma = (1, 2)) (trans_img)
    
    return trans_img
 

############# generando imagenes #############
dir_DataAug = Path('data/output/output_subpreg/DataAugmentation') ## poner ruta nueva
dir_DataAug.mkdir(exist_ok=True, parents=True)

rutas = []
for img in range(df_dup.shape[0]):
    imagen = df_dup.iloc[img]
    trans_img = transform_img(Path(imagen['ruta_imagen_output']))
    nombre_imagen = 'Transf_' + re.search(r'(\d{7,})_.*jpg', imagen['ruta_imagen_output']).group(0)
    trans_img.save(dir_DataAug / nombre_imagen)
    print('imagen_guardada :D')
    rutas.append(dir_DataAug / nombre_imagen)
    
df_dup['ruta_imagen_output'] = rutas  ## modificando ruta final output

# save .csv
pd.concat([df_exist, df_dup]).reset_index(drop=True).drop('index', axis= 1).to_csv(dir_tabla_99 / 'prueba_torch2.csv')

##########################

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, filter_sospecha=False):
        self.labels_frame = pd.read_csv(csv_file)

        if filter_sospecha:
            self.labels_frame = self.labels_frame[self.labels_frame['dm_sospecha'] == 1].reset_index(drop=True)

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.labels_frame.loc[idx, 'ruta_imagen_output'])
        image = Image.open(img_path)
        label = self.labels_frame.loc[idx,  'dm_final']
        directory = self.labels_frame.loc[idx, 'ruta_imagen_output']

        if self.transform:
            image = self.transform(image)

        return image, label, directory
    

transform = transforms.Compose([
    transforms.Resize((224, 224)),    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
    
dataset = CustomImageDataset(csv_file=dir_tabla_99 / nombre_tabla_casos99, root_dir='', transform=transform,
                             filter_sospecha=True)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

dataiter = iter(dataloader)
images, labels, dirs = next(dataiter)
dirs[0]

### ... continuar con el codigo normalmente jiji
