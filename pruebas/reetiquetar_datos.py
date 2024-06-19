import pandas as pd
from config.proc_img import dir_tabla_99, SEED, dir_train_test

est99 = pd.read_csv(dir_tabla_99 / 'casos_99_entrenamiento_compilados_estudiantes.csv')
pad99 = pd.read_csv(dir_tabla_99 / 'casos_99_entrenamiento_compilados_padres.csv')

fs = pd.concat([est99, pad99])
fs = fs[fs.dm_sospecha.eq(1) & fs.dm_final.eq(0)]
fs['origen'] = 'falsa_sospecha'
fs_sample = fs.sample(800, random_state=SEED)

tinta_problematico = pd.read_excel('data/otros/problematicos.xlsx')
tinta_problematico['origen'] = 'ratio_tinta'

train = pd.read_csv(dir_train_test / 'train.csv') 
train['origen'] = 'doble_marca_normal'
not_fs = train[train.falsa_sospecha.eq(0)]
train_sample = not_fs.sample(300, random_state=SEED).drop(columns=['Unnamed: 0'])
a_revisar = (pd.concat([tinta_problematico, fs_sample, train_sample]).drop(columns=['Unnamed: 0', 'indice_original'])
             .drop_duplicates(['ruta_imagen_output']))[['origen', 'ruta_imagen_output', 'ruta_imagen', 'dm_final']]
import numpy as np
a_revisar['encargado'] = np.tile(['juane', 'klaus', 'javi', 'nacho'], len(a_revisar)//4 + 1)[:len(a_revisar)]
a_revisar = a_revisar.rename(columns={'dm_final': 'etiqueta_original'}) 
a_revisar['etiqueta_final'] = ''
a_revisar = a_revisar[['ruta_imagen_output', 'ruta_imagen', 'origen', 'encargado', 'etiqueta_original', 'etiqueta_final']]
a_revisar.to_excel('data/otros/datos_a_revisar.xlsx', index=False)