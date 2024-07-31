import torch
torch.cuda.is_available()
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary

import mlflow

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
from config.proc_img import get_directorios
from config.proc_img import SEED
import numpy as np
## Predicciones -----------------------------------------------------

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)




config_dict = read_json('saved/models/saved_server/maxvit/config.json')
config = ConfigParser(config_dict)
directorios = get_directorios()
device, device_ids = prepare_device(config['n_gpu'])



model  = config.init_obj('arch', models)
model_name = config['arch']['type']
model = preparar_capas_modelo(model, model_name)

ruta_modelo = 'saved/models/saved_server/maxvit/model_best_nuevo.pt'
checkpoint = torch.load(ruta_modelo)
state_dict = checkpoint['state_dict']
if config['n_gpu'] > 1:
    model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)

model = model.to(device)

model.eval()
dir_train_test = get_directorios(filtro='dir_train_test')

testloader = config.init_obj('data_loader_test', module_data, model=model_name, 
                             return_directory=True, dir_data=dir_train_test)






#model_load.eval()

# Initialize lists to store predictions and true labels
predictions = []
probs = []
true_labels = []
lst_directories = []

with torch.no_grad():
    # Iterate over the test data
    for n,(images, labels, directories) in enumerate(tqdm(testloader)):
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

test = pd.read_csv(dir_train_test / 'test_8b.csv')
test_rev = pd.read_excel('data/otros/resultados_maxvit2_rev.xlsx')[['ruta_imagen_output', 'etiqueta_final']]
# test_rev2 = pd.read_excel('data/otros/datos_revisados_p3.xlsx')[['ruta_imagen_output', 'etiqueta_final']]
# test_rev = test_rev[~test_rev.ruta_imagen_output.isin(test_rev2.ruta_imagen_output)]
# test_rev_total = pd.concat([test_rev, test_rev2])
#test = test.merge(test_rev, on='ruta_imagen_output', how='left')
#test['dm_final2'] = test.etiqueta_final.combine_first(test.dm_final)

#train = pd.read_csv(dir_train_test / 'train.csv')

preds = pd.DataFrame({'pred': predictions,
              'true': true_labels,
              'proba': probs_float,
              'dirs': lst_directories})

#preds['dirs'] = test.ruta_imagen_output

preds_tot = preds.merge(test, left_on='dirs', right_on='ruta_imagen_output', how='left')


preds_tot[preds_tot.acierto.eq(0) & preds_tot.deciles.cat.codes.gt(10)]

preds_tot_export = preds_tot[['ruta_imagen_output', 'ruta_imagen'  'pred', 'proba']]
preds_tot_export = preds_tot_export[preds_tot_export.acierto.ne(1)]
preds_tot_export['etiqueta_final'] = ''
preds_tot_export.drop(columns=['acierto']).sort_values('proba', ascending=False).to_excel('data/otros/resultados_maxvit_8b.xlsx')

