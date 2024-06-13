from config.proc_img import dir_tabla_99, dir_subpreg_aug, dir_train_test, SEED, dir_subpreg
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import torchvision.transforms.v2 as v2
from PIL import Image
import torch
import random
# Creamos directorio para imágenes aumentadas:

def get_img_existentes():
    dir_subpreg_aug.mkdir(parents=True, exist_ok=True)

    padres99 = f'casos_99_entrenamiento_compilados_padres.csv'
    est99 = f'casos_99_entrenamiento_compilados_estudiantes.csv'
    df99p = pd.read_csv(dir_tabla_99 / padres99)
    
    df99e = pd.read_csv(dir_tabla_99 / est99)
    df99e['falsa_sospecha'] = ((df99e['dm_sospecha'] == 1) & (df99e['dm_final'] == 0))
    est_falsa_sospecha = df99e[df99e.falsa_sospecha.eq(1)]
    otros_est = df99e[df99e.falsa_sospecha.eq(0)].sample(frac=.1, random_state=42)
    df99 = pd.concat([est_falsa_sospecha, otros_est, df99p]).reset_index(drop=True)

    df_exist = df99[df99.ruta_imagen_output.apply(lambda x: Path(x).is_file())].reset_index()
    print(f'{df_exist.shape=}')
    print(f'{df99.shape=}')
    df_exist.dm_sospecha = (df_exist.dm_sospecha == 99).astype(int)
    df_exist['falsa_sospecha'] = ((df_exist['dm_sospecha'] == 1) & (df_exist['dm_final'] == 0))

    df_sampleado = df_exist[df_exist.dm_sospecha.eq(0)]
    df_sospecha = df_exist[df_exist.dm_sospecha.ne(0)]

    return df_sospecha, df_sampleado


def gen_train_test():

    df_exist, df_sampleado = get_img_existentes()

    train, test = train_test_split(df_exist, stratify=df_exist['falsa_sospecha'], test_size=.2)

    df_aug = gen_df_aumentado(train)

    export_train_test(train, df_aug, test, df_sampleado)


############# generando imagenes #############



def gen_df_aumentado(train):


    df_aug = train[train.falsa_sospecha.eq(1)].copy()


    
    rutas = []
    df_aug_final = pd.DataFrame()
    # 5 rondas de aumentado
    for i in range(5):
        random.seed(SEED+i)
        torch.manual_seed(SEED+i)
        print(f'{i=}')
        df_aug_final = pd.concat([df_aug_final, df_aug])


        for img in range(df_aug.shape[0]):
            dir_imagen_og = Path(df_aug.iloc[img].ruta_imagen_output)

            dir_imagen_aug = Path(str(dir_imagen_og).replace(dir_subpreg.name, dir_subpreg_aug.name))
            
            trans_img = transform_img(dir_imagen_og)
            dir_imagen_aug =  Path(str(dir_imagen_aug.with_suffix('')) + f'aug_{i+1}.jpg')
            dir_imagen_aug.parent.mkdir(exist_ok=True, parents=True)
            trans_img.save(dir_imagen_aug)
            #print('imagen_guardada :D')
            rutas.append(dir_imagen_aug)
    
    df_aug_final['ruta_imagen_output'] = rutas  ## modificando ruta final output

    print('Imágenes aumentadas exportadas con éxito!')

    return df_aug_final

def export_train_test(train, df_aug, test, df_sampleado):
    dir_train_test.mkdir(parents=True, exist_ok=True)
    # save .csv
    (pd.concat([train, df_aug, df_sampleado]).reset_index()
    .rename(columns={'level_0': 'indice_original'})
    .drop('index', axis= 1).to_csv(dir_train_test / 'train.csv'))

    test[test.dm_sospecha.eq(1)].reset_index().rename(columns={'level_0': 'indice_original'}).drop('index', axis= 1).to_csv(dir_train_test / 'test.csv')
    print('Tablas de entrenamiento y test exportadas exitosamente!')

#### funciones aux para generar transformaciones

def addnoise(input_image, noise_factor = 0.1):
    """transforma la imagen agredando un ruido gausiano"""
    inputs = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(input_image)
    noise = inputs + torch.rand_like(inputs) * noise_factor
    noise = torch.clip (noise,0,1.)
    output_image = v2.ToPILImage()
    image = output_image(noise)
    return image

class RandomRotation:
    def __init__(self, degrees, p):
        self.degrees = degrees
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            if self.degrees % 90 == 0:  # Only handle multiples of 90 degrees
                rotations = (self.degrees // 90) % 4
                for _ in range(rotations):
                    x = x.transpose(Image.ROTATE_90)
            else:
                print("This implementation only supports rotations that are multiples of 90 degrees.")
        return x


def transform_img(path_img):
    """Transformaciones a imagen para aumento de casos de sospecha de doble marca en entrenamiento"""
    orig_img = Image.open(path_img)

    trans_img = v2.RandomHorizontalFlip(0.7)(orig_img)               # Randomly flip
    trans_img = v2.RandomVerticalFlip(0.7)(trans_img) 
    trans_img = RandomRotation(degrees=90, p=.5 )(trans_img)
    trans_img = v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)
    trans_img = addnoise(trans_img)
    trans_img = v2.GaussianBlur(kernel_size = (3, 5), sigma = (1, 2)) (trans_img)
    
    return trans_img




