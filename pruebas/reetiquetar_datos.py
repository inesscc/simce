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



# A revisar, parte 2 --------------

import torch
torch.cuda.is_available()
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary
from config.proc_img import dir_modelos
import mlflow
from config.proc_img import dir_train_test
import pandas as pd
from pathlib import Path
from torchvision.models import list_models
from config.parse_config import ConfigParser
from simce.utils import read_json, prepare_device
import model.model as module_arch
import data_loader.data_loaders as module_data
from trainer import Trainer
import model.metric as module_metric
from torchvision import models
from simce.modelamiento import preparar_capas_modelo

config_dict = read_json('saved/models/maxvit_t/0701_092128/config.json')
config = ConfigParser(config_dict)
device, device_ids = prepare_device(config['n_gpu'])
ruta_modelo = 'saved/models/maxvit_t/0701_092128/model_best.pt'
config2 = read_json('config/model.json')
config2 = ConfigParser(config2)
trainloader = config2.init_obj('data_loader_train', module_data, shuffle=False, batch_size=256, return_directory=True )

model_load  = config.init_obj('arch', models)
model_load = preparar_capas_modelo(model_load, config['arch']['type'])

checkpoint = torch.load(ruta_modelo)
state_dict = checkpoint['state_dict']
model_load.load_state_dict(state_dict)
model_load.to(device)


#model_load.eval()

# Initialize lists to store predictions and true labels
predictions = []
probs = []
true_labels = []
lst_directories = []
# Iterate over the test data
for n,(images, labels, directories) in enumerate(trainloader):
    print(n, '/' , len(trainloader))
    # Move the images and labels to the same device as the model
    images = images.to(device)
    labels = labels.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model_load(images)

    probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
    max_probabilities = probabilities.max(dim=1)[0]
    
    # Get the predicted class for each image
    _, predicted = torch.max(outputs.data, 1)

    # Store the predictions and true labels
    predictions.extend(predicted.tolist())
    true_labels.extend(labels.tolist())
    probs.extend(max_probabilities)
    lst_directories.extend(directories)

print('Predicciones listas!')
probs_float = [i.item() for i in probs]
import pandas as pd
test = pd.read_csv(dir_train_test / 'test.csv')
train = pd.read_csv(dir_train_test / 'train.csv')

preds = pd.DataFrame({'pred': predictions,
              'true': true_labels,
              'proba': probs_float,
              'dirs': lst_directories})
#preds['dirs'] = train.ruta_imagen_output

preds_tot = preds.merge(train, left_on='dirs', right_on='ruta_imagen_output', how='left')

preds_tot['acierto'] = preds_tot.pred == preds_tot.true


preds[preds.true.eq(1) & preds.pred.eq(0)].reset_index().dirs.apply(lambda x: Path(x).name).value_counts()

preds_tot['deciles'] = pd.qcut(preds_tot.proba, q=20)
preds_tot.groupby('deciles').acierto.mean().plot()
plt.axhline(.98, color='red')
plt.show()

preds_tot = preds_tot.sort_values('proba', ascending=False)
fp = preds_tot[preds_tot.acierto.eq(0) & preds_tot.dm_final.eq(1)].copy()
fp['origen'] = 'falso_negativo'
fn = preds_tot[preds_tot.acierto.eq(0) & preds_tot.dm_final.eq(0)].copy()
fn['origen'] = 'falso_positivo'
import numpy as np
a_revisar_p2 = pd.concat([fp.head(int(np.round(len(fp)/2))),
                            fn.head(int(np.round(len(fn)/2)))])[['origen', 'ruta_imagen_output', 'ruta_imagen', 'dm_final', 'pred']]
a_revisar_p2['encargado'] = np.tile(['juane', 'klaus', 'javi', 'nacho'], len(a_revisar_p2)//4 + 1)[:len(a_revisar_p2)]
a_revisar_p2 = a_revisar_p2.rename(columns={'dm_final': 'etiqueta_original', 'pred': 'etiqueta_predicha'}) 
a_revisar_p2['etiqueta_final'] = ''
a_revisar_p2 = a_revisar_p2[['ruta_imagen_output', 'ruta_imagen', 'origen', 'encargado', 'etiqueta_original',
                              'etiqueta_predicha', 'etiqueta_final']]
a_revisar_p2.to_excel('data/otros/datos_a_revisar_p2.xlsx', index=False)
a_revisar_p2



from config.proc_img import dir_input, dir_subpreg
folders_output = set([i.name for i in (dir_subpreg / 'CE').glob('*')])
folders_input = set([i.name for i in (dir_input / 'CE').glob('*')])
klaus = a_revisar_p2[a_revisar_p2.encargado.eq('klaus')]
files_input = (dir_input / klaus.ruta_imagen.str.replace('\\', '/').squeeze()).to_list()
files_output = klaus.ruta_imagen_output

import zipfile
output_filename = 'klaus.zip'

with zipfile.ZipFile(output_filename, 'w') as zipf:
    # Add input files to the 'input' folder
    for file_path in files_input:
        zipf.write(file_path, arcname=f'input/{file_path}')

    # Add output files to the 'output' folder
    for file_path in files_output:
        zipf.write(file_path, arcname=f'output/{file_path}')

print(f"Zip archive '{output_filename}' created successfully!")


preds_tot.to_excel('data/otros/resultados_maxvit.xlsx')