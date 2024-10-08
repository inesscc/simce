from config.proc_img import SEED, nombre_tabla_predicciones
import pandas as pd

from pathlib import Path

import config.proc_img as module_config
# Creamos directorio para imágenes aumentadas:




def get_img_existentes_pred(directorios, curso) -> pd.DataFrame:
    '''
    Genera dataframe con archivos existentes (que sí pudieron ser procesados), con la siguiente lógica:

    - Obtiene todas las falsas sospechas (dm_sospecha == 1 y dm_final == 0) de padres y estudiantes
    - Obtiene fraccion_sample de las filas que no son falsa sospecha de estudiantes (para evitar desbalancear
    clases y hacer crecer demasiado el dataset)
    '''
    
    padres99 = f'casos_99_compilados_{curso}_padres.csv'
    est99 = f'casos_99_compilados_{curso}_estudiantes.csv'

    # if (directorios['dir_tabla_99'] / padres99).is_file():
    df99p = pd.read_csv(directorios['dir_tabla_99'] / padres99)
    
    df99e = pd.read_csv(directorios['dir_tabla_99'] / est99)

    df99 = pd.concat([df99p, df99e]).reset_index(drop=True)

    # Filtramos archivos que efectivamente fue posible procesar:
    df_exist = df99[df99.ruta_imagen_output.apply(lambda x: Path(x).is_file())].reset_index()


    print(f'{df_exist.shape=}')
    print(f'{df99.shape=}')

    return df_exist




    

def gen_pred_set(directorios, curso):

    df_exist = get_img_existentes_pred(directorios, curso)
    df_exist.to_csv(directorios['dir_train_test'] / nombre_tabla_predicciones, index=False)

    return print('Tablas para predicción exportadas exitosamente!')
    

    # Traemos re-etiquetado 

    # directorios = config.init_obj('directorios', module_config, curso='4b' )
    # df_exist_curso = get_img_existentes(fraccion_sample, directorios, curso='4b' )

    # df_exist_re = incorporar_reetiquetas(df_exist_curso,directorios, '4b')

    

    # df_sospecha, df_sampleado = separar_dataframes(df_exist_re)

    # train, test = train_test_split(df_sospecha, stratify=df_sospecha['falsa_sospecha'], test_size=.2, random_state=SEED)

    # df_aug = gen_df_aumentado(train, directorios, n_augment_rounds=n_augment_rounds)

    # export_train_test(train, df_aug, test, directorios, df_sampleado=None)


############# generando imagenes #############

# def gen_train_test(n_augment_rounds, fraccion_sample, config):

#     for curso in ['4b', '8b']:

#         directorios_curso = config.init_obj('directorios', module_config, curso=curso )
#         df_exist_curso = get_img_existentes(fraccion_sample, directorios_curso, curso=curso )

#         if curso == '4b':
#             df_exist_curso = incorporar_reetiquetas(df_exist_curso,directorios_curso, curso)
        
#         # df_sospecha, df_sampleado = separar_dataframes(df_exist_curso)

#         train, test = train_test_split(df_exist_curso, stratify=df_exist_curso['falsa_sospecha'],
#                                         test_size=.2, random_state=SEED)

#         df_aug = gen_df_aumentado(train, directorios_curso, n_augment_rounds=n_augment_rounds)

#         export_train_test(train, df_aug, test, directorios_curso, curso=curso, df_sampleado=None)
#     return print('Tablas exportadas para todos los cursos exitosamente!')

# def gen_df_aumentado(train: pd.DataFrame, directorios, n_augment_rounds:int) -> pd.DataFrame:
#     '''Genera n_augment_rounds copias de cada fila del set de entrenamiento con distintas transformaciones que
#      se generarán de acuerdo con una probabilidad '''
    
#     df_aug = train[train.falsa_sospecha.eq(1)].copy()



    
#     rutas = []
#     df_aug_final = pd.DataFrame()
#     # n_augments rondas de aumentado
#     for i in range(n_augment_rounds):
#         random.seed(SEED+i)
#         torch.manual_seed(SEED+i)
#         print(f'{i=}')
#         df_aug_final = pd.concat([df_aug_final, df_aug])


#         for img in range(df_aug.shape[0]):
#             dir_imagen_og = Path(df_aug.iloc[img].ruta_imagen_output)

#             dir_imagen_aug = Path(str(dir_imagen_og).replace(directorios['dir_subpreg'].name, directorios['dir_subpreg_aug'].name))
            
#             trans_img = transform_img(dir_imagen_og, i)
#             dir_imagen_aug =  Path(str(dir_imagen_aug.with_suffix('')) + f'aug_{i+1}.jpg')
#             dir_imagen_aug.parent.mkdir(exist_ok=True, parents=True)
#             trans_img.save(dir_imagen_aug)
#             #print('imagen_guardada :D')
#             rutas.append(dir_imagen_aug)
    
#     df_aug_final['ruta_imagen_output'] = rutas  ## modificando ruta final output

#     print('Imágenes aumentadas exportadas con éxito!')

#     return df_aug_final

def export_train_test(train, df_aug, test, directorios, curso, df_sampleado=None):


    if df_sampleado:
        dfs_concat = [train, df_aug, df_sampleado]
    else:
        dfs_concat = [train, df_aug]

    (pd.concat(dfs_concat).reset_index()
    .rename(columns={'level_0': 'indice_original'})
    .drop('index', axis= 1).to_csv(directorios['dir_train_test'] / f'train_{curso}.csv'))

    (test[test.dm_sospecha.eq(1)]
     .reset_index()
     .rename(columns={'level_0': 'indice_original'})
     .drop('index', axis= 1)
     .to_csv(directorios['dir_train_test'] / f'test_{curso}.csv'))
    print('Tablas de entrenamiento y test exportadas exitosamente!')

#### funciones aux para generar transformaciones

# def random_addnoise(input_image, noise_factor = 0.1, p=.5):
#     """transforma la imagen agredando un ruido gausiano con probabilidad p"""
#     if random.random() < p:
#         inputs = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])(input_image)
#         noise = inputs + torch.rand_like(inputs) * noise_factor
#         noise = torch.clip (noise,0,1.)
#         output_image = v2.ToPILImage()
#         image = output_image(noise)
#     else:
#         image = input_image
#     return image




# def transform_img(path_img, i):
#     """Transformaciones a imagen para aumento de casos de sospecha de doble marca en entrenamiento"""
#     orig_img = Image.open(path_img)

#     if i == 0:
#         trans_img = v2.RandomHorizontalFlip(1)(orig_img)    
#        # trans_img = v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)         
#     elif i == 1:
#         trans_img = v2.RandomVerticalFlip(1)(orig_img) 
#        # trans_img = v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)
#     elif i == 2:
#         trans_img = v2.RandomHorizontalFlip(1)(orig_img)  
#         trans_img = v2.RandomVerticalFlip(1)(trans_img) 
#        # trans_img = v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)
#     elif i == 3:
#         trans_img = random_addnoise(orig_img)
#        # trans_img = v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)
#     elif i == 4:

#         trans_img = v2.GaussianBlur(kernel_size = (3, 5), sigma = (1, 2)) (orig_img)
#        # trans_img = v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.01)(trans_img)
#     elif i >= 5:
#         trans_img = v2.ColorJitter(brightness=0.08, contrast=0.15, saturation=0.15, hue=0.005)(orig_img)
#     else:

#         raise 'Función acepta hasta 5 rondas de data augmentation'

    
#     return trans_img




# from config.proc_img import dir_input
# folders_output = set([i.name for i in (dir_subpreg / 'CE').glob('*')])
# folders_input = set([i.name for i in (dir_input / 'CE').glob('*')])
# dif_folders = sorted(folders_input.difference(folders_output))
# folders_to_zip = [i for i in (dir_input / 'CE').glob('*') if i.name in dif_folders]
# import shutil
# import tempfile
# with tempfile.TemporaryDirectory() as temp_dir:
#     for folder in folders_to_zip:
#         shutil.copytree(folder, Path(temp_dir) / folder.name)

#     # Create the zip archive from the temporary directory
#     shutil.make_archive('carpetas_faltantes', 'zip', temp_dir)


# def separar_dataframes(df_exist) -> tuple[pd.DataFrame, pd.DataFrame]:
#     # Obtenemos versión final de variable falsa sospecha con todos los datos:
    

#     # Separamos entre datos que fueron obtenidos de sampleo aleatorio y datos obtenidos de sospechas de doble marca:
#     df_sampleado = df_exist[df_exist.dm_sospecha.eq(0)]
#     df_sospecha = df_exist[df_exist.dm_sospecha.ne(0)]

#     return df_sospecha, df_sampleado

# def get_img_existentes(fraccion_sample: float, directorios, curso) -> pd.DataFrame:
#     '''
#     Genera dataframe con archivos existentes (que sí pudieron ser procesados), con la siguiente lógica:

#     - Obtiene todas las falsas sospechas (dm_sospecha == 1 y dm_final == 0) de padres y estudiantes
#     - Obtiene fraccion_sample de las filas que no son falsa sospecha de estudiantes (para evitar desbalancear
#     clases y hacer crecer demasiado el dataset)
#     '''
    
#     padres99 = f'casos_99_entrenamiento_compilados_{curso}_padres.csv'
#     est99 = f'casos_99_entrenamiento_compilados_{curso}_estudiantes.csv'

#     if (directorios['dir_tabla_99'] / padres99).is_file():
#         df99p = pd.read_csv(directorios['dir_tabla_99'] / padres99)
    
#     df99e = pd.read_csv(directorios['dir_tabla_99'] / est99)


#     # Obtenemos falsas sospechas de estudiantes para filtrar casos relevantes
#     df99e['falsa_sospecha'] = ((df99e['dm_sospecha'] == 1) & (df99e['dm_final'] == 0))
#     est_falsa_sospecha = df99e[df99e.falsa_sospecha.eq(1)]

#     # Obtenemos frac_sample de los otros casos para alimentar el dataset:
#     otros_est = df99e[df99e.falsa_sospecha.eq(0)].sample(frac=fraccion_sample, random_state=SEED)

#     if (directorios['dir_tabla_99'] / padres99).is_file():
#         dfs_99 = [est_falsa_sospecha, otros_est, df99p]
#     else:
#         dfs_99 = [est_falsa_sospecha, otros_est]

#     df99 = pd.concat(dfs_99).reset_index(drop=True)

#     # Filtramos archivos que efectivamente fue posible procesar:
#     df_exist = df99[df99.ruta_imagen_output.apply(lambda x: Path(x).is_file())].reset_index()


#     print(f'{df_exist.shape=}')
#     print(f'{df99.shape=}')

#     return df_exist


# def incorporar_reetiquetas(df_exist: pd.DataFrame, directorios, curso) -> pd.DataFrame:
#     '''
#     Incorpora re-etiquetas para mejorar calidad del dataset
#     '''

#     reetiqueta = pd.read_excel(directorios['dir_insumos'] / 'datos_revisados.xlsx')
#     reetiqueta2 = pd.read_excel(directorios['dir_insumos'] / 'datos_revisados_p2_2.xlsx')
#     reetiqueta3 = pd.read_excel(directorios['dir_insumos'] / 'datos_revisados_p3.xlsx')

#     etiqueta_final =reetiqueta.set_index('ruta_imagen_output').etiqueta_final
#     data_eliminar = set(etiqueta_final[etiqueta_final.isin(['-', 99])].index)    
#     etiqueta_final = etiqueta_final[~etiqueta_final.isin(['-', 99])]
#     etiqueta_final.index = etiqueta_final.index.str.replace(str(directorios['dir_input_proc']), 
#                                                             str(directorios['dir_input_proc'] / curso))
 

#     etiqueta_final2 =reetiqueta2.set_index('ruta_imagen_output').etiqueta_final
#     data_eliminar.update(set(etiqueta_final2[etiqueta_final2.isin(['-', 99])].index)  )
#     etiqueta_final2 = etiqueta_final2[~etiqueta_final2.isin(['-', 99])]
#     etiqueta_final2.index = etiqueta_final2.index.str.replace(str(directorios['dir_input_proc']),
#                                                                str(directorios['dir_input_proc'] / curso))
    
#     etiqueta_final3 =reetiqueta3.set_index('ruta_imagen_output').etiqueta_final
#     data_eliminar.update(set(etiqueta_final3[etiqueta_final3.isin(['-', 99])].index)  )  
#     etiqueta_final3 = etiqueta_final3[~etiqueta_final3.isin(['-', 99])]
#     etiqueta_final3.index = etiqueta_final3.index.str.replace(str(directorios['dir_input_proc']), 
#                                                             str(directorios['dir_input_proc'] ))

#     df_exist['reetiqueta'] = df_exist.ruta_imagen_output.map(etiqueta_final)
#     df_exist['reetiqueta2'] = df_exist.ruta_imagen_output.map(etiqueta_final2)
#     df_exist['reetiqueta3'] = df_exist.ruta_imagen_output.map(etiqueta_final3)

#     df_exist['reetiqueta'] = df_exist.reetiqueta3.combine_first(df_exist.reetiqueta2).combine_first(df_exist.reetiqueta)
    
#     df_exist['dm_final'] = df_exist.reetiqueta.combine_first(df_exist.dm_final).astype(int)

#     df_exist['falsa_sospecha'] = ((df_exist['dm_sospecha'] == 1) & (df_exist['dm_final'] == 0))

#     # Eliminamos datos confusos para el modelo
#     df_exist_final = df_exist[~df_exist.ruta_imagen_output.isin(data_eliminar)]

#     return df_exist_final