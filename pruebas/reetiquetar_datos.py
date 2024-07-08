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
import data_loader.data_loaders as module_data
from simce.modelamiento import preparar_capas_modelo
from tqdm import tqdm
## Predicciones -----------------------------------------------------


config_dict = read_json('saved/models/saved_server/efficientnetv2_m/config.json')
config = ConfigParser(config_dict)
device, device_ids = prepare_device(config['n_gpu'])



model  = config.init_obj('arch', models)
model_name = config['arch']['type']
model = preparar_capas_modelo(model, model_name)

ruta_modelo = 'saved/models//saved_server/efficientnetv2_m/model_best.pt'
checkpoint = torch.load(ruta_modelo)
state_dict = checkpoint['state_dict']
if config['n_gpu'] > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

testloader = config.init_obj('data_loader_test', module_data, model=model_name, 
                             return_directory=True)
trainloader = config.init_obj('data_loader_train', module_data, model=model_name, 
                              return_directory=True)





#model_load.eval()

# Initialize lists to store predictions and true labels
predictions = []
probs = []
true_labels = []
lst_directories = []

with torch.no_grad():
    # Iterate over the test data
    for n,(images, labels, directories) in enumerate(tqdm(trainloader)):
        # Move the images and labels to the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Make predictions
        
        outputs = model(images)

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
probs_float
import pandas as pd

test = pd.read_csv(dir_train_test / 'test.csv')
train = pd.read_csv(dir_train_test / 'train.csv')
train.dm_final.value_counts()

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
preds_tot.dirs.iloc[0]
preds_tot['origen_imagen'] = preds_tot.dirs.str.extract('(augmented|base)')
preds_tot = preds_tot[preds_tot.origen_imagen.ne('augmented')]
fp = preds_tot[preds_tot.acierto.eq(0) & preds_tot.dm_final.eq(1)].copy()
fp['origen'] = 'falso_negativo'
fn = preds_tot[preds_tot.acierto.eq(0) & preds_tot.dm_final.eq(0)].copy()
fn['origen'] = 'falso_positivo'

import numpy as np
a_revisar_p2 = pd.concat([fp, fn])[['origen', 'ruta_imagen_output', 'ruta_imagen', 'dm_final', 'pred']]

a_revisar_p2['encargado'] = np.tile(['juane', 'klaus', 'javi', 'nacho'], len(a_revisar_p2)//4 + 1)[:len(a_revisar_p2)]
a_revisar_p2 = a_revisar_p2.rename(columns={'dm_final': 'etiqueta_original', 'pred': 'etiqueta_predicha'}) 
a_revisar_p2['etiqueta_final'] = ''
a_revisar_p2 = a_revisar_p2[['ruta_imagen_output', 'ruta_imagen', 'origen', 'encargado', 'etiqueta_original',
                              'etiqueta_predicha', 'etiqueta_final']]
a_revisar_p2.to_excel('data/otros/datos_a_revisar_p2_2.xlsx', index=False)
a_revisar_p2



from config.proc_img import dir_input, dir_subpreg, dir_subpreg_aux
folders_output = set([i.name for i in (dir_subpreg / 'CE').glob('*')])
folders_input = set([i.name for i in (dir_input / 'CE').glob('*')])
klaus = rev[rev.encargado.eq('nacho')]
files_input = (dir_input / klaus.ruta_imagen.str.replace('\\', '/').squeeze()).to_list()
files_output = klaus.ruta_imagen_output.apply(lambda x: dir_subpreg_aux / ('/'.join(Path(x).parts[-4:])))

import zipfile
output_filename = 'nacho.zip'

with zipfile.ZipFile(output_filename, 'w') as zipf:
    # Add input files to the 'input' folder
    for file_path in files_input:
        zipf.write(file_path, arcname=f'input/{file_path}')

    # Add output files to the 'output' folder
    for file_path in files_output:
        zipf.write(file_path, arcname=f'output/{file_path}')

print(f"Zip archive '{output_filename}' created successfully!")


preds_tot.to_excel('data/otros/resultados_maxvit.xlsx')