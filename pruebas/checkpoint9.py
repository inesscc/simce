import pandas as pd
import config.proc_img as module_config
from config.parse_config import ConfigParser
from simce.utils import read_json
from config.proc_img import CURSO
import os

os.listdir('//10.10.100.28/4b_2023')
config_dict = read_json('config/model.json')
config = ConfigParser(config_dict)
d = config.init_obj('directorios', module_config, curso=str(module_config.CURSO) )
r1 = pd.read_excel(d['dir_insumos'] / 'datos_revisados.xlsx') 
r2 = pd.read_excel(d['dir_insumos'] / 'datos_revisados_p2_2.xlsx')
r3 = pd.read_excel('data/otros/datos_revisados_p3.xlsx') 


muestra = r1[r1.origen.eq('doble_marca_normal')]
muestra.etiqueta_original.ne(muestra.etiqueta_final).value_counts()

rev = pd.concat([r1, r2, r3])
rev['cambio_etiqueta'] = rev.etiqueta_original != rev.etiqueta_final
rev.cambio_etiqueta.value_counts()
rev.mostrar_ACE.value_counts()



# ----------
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
import config.proc_img as module_config
from config.proc_img import SEED
import numpy as np

## Predicciones -----------------------------------------------------


    
fix_test(Path('data/input_modelamiento'))
def testear_modelo(modelo):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)


    config_dict = read_json(f'saved/models/saved_server/{modelo}/config.json')
    config = ConfigParser(config_dict)

    device, _ = prepare_device(config['n_gpu'])



    model  = config.init_obj('arch', models)
    model_name = config['arch']['type']
    model = preparar_capas_modelo(model, model_name)

    ruta_modelo = f'saved/models/saved_server/{modelo}/model_best.pt'
    checkpoint = torch.load(ruta_modelo)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()
    dir_train_test = config.init_obj('directorios', module_config, curso='4b', filtro='dir_train_test' )

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

    test = pd.read_csv(dir_train_test / 'test.csv')
    
    test_rev = pd.read_excel('data/otros/resultados_maxvit_rev.xlsx')[['ruta_imagen_output', 'etiqueta_final']]
    test_rev2 = pd.read_excel('data/otros/datos_revisados_p3.xlsx')[['ruta_imagen_output', 'etiqueta_final']]
    test_rev = test_rev[~test_rev.ruta_imagen_output.isin(test_rev2.ruta_imagen_output)]
    test_rev_total = pd.concat([test_rev, test_rev2])
    test = test.merge(test_rev_total, on='ruta_imagen_output', how='left')
    test['dm_final2'] = test.etiqueta_final.combine_first(test.dm_final)


    preds = pd.DataFrame({'pred': predictions,
                'true': true_labels,
                'proba': probs_float,
                'dirs': lst_directories})

    #preds['dirs'] = test.ruta_imagen_output

    preds_tot = preds.merge(test, left_on='dirs', right_on='ruta_imagen_output', how='left')
    preds_tot['acierto'] = preds_tot.pred == preds_tot.dm_final
    preds_tot['acierto2'] = preds_tot.pred == preds_tot.dm_final2



    preds_tot['veintiles'] = pd.qcut(preds_tot.proba, q=20).cat.codes + 1
    return preds_tot

preds_tot_maxvit = testear_modelo('maxvit')



preds_tot[preds_tot.acierto.eq(0)].veintiles.value_counts()
import matplotlib.ticker as mtick

ax = preds_tot.groupby('veintiles').acierto2.mean().mul(100).plot(figsize=(14,7), color='blue')
#preds_tot.groupby('veintiles').acierto.mean().mul(100).plot(ax=ax, color='gray')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.axhline(99, color='red', ls='--', lw=.5)
plt.axhline(100, color='red', ls='solid', lw=1)
#plt.axvline(8, color='green')
plt.title('% de acierto según veintil de certeza del modelo', fontsize=18)
plt.xticks(range(1,21), fontsize=15)
plt.yticks(fontsize=15)
#plt.ylim(0,100)
plt.ylabel('% de acierto', fontsize=15)
plt.xlabel('veintiles', fontsize=15)
#plt.legend(['acierto original', 'acierto post-revisión', '99%', '100%'], fontsize=15)
plt.legend(['acierto post-revisión', '99%', '100%', ], fontsize=15)

plt.savefig('figura.png', bbox_inches='tight', dpi=200)
plt.show()

preds_tot[preds_tot.acierto.eq(0) & preds_tot.deciles.cat.codes.gt(10)]
preds_tot['deciles'] = preds_tot['deciles'].cat.codes
preds_tot_export = preds_tot[['ruta_imagen_output', 'ruta_imagen' ,'true', 'pred', 'acierto', 'deciles', 'proba']]
preds_tot_export = preds_tot_export[preds_tot_export.acierto.ne(1)]
preds_tot_export['etiqueta_final'] = ''
preds_tot_export.drop(columns=['acierto']).sort_values('proba', ascending=False).to_excel('data/otros/resultados_maxvit.xlsx')
preds_tot[preds_tot.deciles.cat.codes.ge(7)].acierto.mean()
preds_tot[preds_tot.deciles.cat.codes.ge(16) & preds_tot.dm_final.eq(0)].ruta_imagen_output.iloc[0]
preds_tot.deciles.value_counts()
test.dm_final.value_counts().div(test.shape[0])
preds_tot.acierto.mean()
preds_tot.proba.describe()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
preds_tot[preds_tot.proba.ge(.95) & preds_tot.true.eq(0)].acierto.mean()

def get_conf_mat(df, preg=None):
    if preg:
        df = df[df.preguntas.eq(preg)]

    cm = confusion_matrix(df.true, df.pred )
    disp = ConfusionMatrixDisplay(cm, display_labels=['Marca normal', 'Doble marca'])
    disp.plot()
    plt.title(preg)
    plt.show()
    

get_conf_mat(preds_tot)
preds_tot['cuestionario'] = preds_tot.dirs.str.extract('(C[EP])')
preds_tot['pregunta'] = preds_tot.dirs.str.extract('(p\d+)')
preds_tot['pregunta'] = preds_tot.cuestionario + '-' + preds_tot.pregunta
preds_tot.groupby(['preguntas']).agg({'rbd':'count', 'acierto':'mean'}).sort_values('acierto')
preds_tot.preguntas
preds_tot[preds_tot.acierto.eq(0) ].pregunta.value_counts()
preds_tot[preds_tot.acierto.eq(0) & preds_tot.cuestionario.eq('CP')].pregunta.value_counts().div(preds_tot[preds_tot.acierto.eq(0) & preds_tot.cuestionario.eq('CP')].shape[0])
preds_tot[preds_tot.acierto.eq(0) ].pregunta.value_counts().div(preds_tot.pregunta.value_counts()).sort_values(ascending=False)
preds_tot[preds_tot.pregunta.eq('CE-p2')]
preds_tot.dirs.str.extract('(C[EP])').value_counts().div(preds_tot.shape[0])
preds_tot[preds_tot.pregunta.eq('CE-p21') & preds_tot.acierto.eq(0)][['dirs', 'acierto']]
preds_tot.loc[3385].dirs
df_exist[df_exist.preguntas.eq('p22')]

df_exist.preguntas.value_counts()

########## --------------------------
########## --------------------------

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor




config_dict = read_json('config/model.json')
config = ConfigParser(config_dict)

classification_models = list_models(module=torchvision.models)
train = pd.read_csv(dir_train_test / config['data_loader_train']['args']['data_file']) 
train[train.falsa_sospecha.eq(0)].ruta_imagen_output.iloc[4]
train[train.falsa_sospecha.eq(1)].iloc[0]
train.dm_final.value_counts()
origen = pd.read_csv(dir_input / 'CE_Origen_DobleMarca.csv', delimiter=';')
final = pd.read_csv(dir_input / 'CE_Final_DobleMarca.csv', delimiter=';')
final[final.serie.eq(4051426)].p24_1
logger = config.get_logger('train')
num_classes = 2

model  = config.init_obj('arch', module_arch, num_classes=num_classes)
trainloader = config.init_obj('data_loader_train', module_data, validation_split=.1)
valid_data_loader = trainloader.split_validation()
testloader = config.init_obj('data_loader_test', module_data)


# Select device
device, device_ids = prepare_device(config['n_gpu'])
print(f'{device=}')
model = model.to(device)
# Si hay más de una GPU, paraleliza el trabajo:
if len(device_ids) > 1:
    model = torch.nn.DataParallel(model, device_ids=device_ids)


# Define the loss function and optimizer
weight = config['class_weights'] 
weight = torch.tensor(weight).to(device)
criterion = config.init_obj('loss', nn, weight=weight)
metrics = [getattr(module_metric, met) for met in config['metrics']]

#optimizer = optim.Adam(model.parameters())
#optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=.001)
# Mover modelo a dispositivo detectado
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = config.init_obj('optimizer_sgd', torch.optim, trainable_params)
lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=trainloader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

trainer.train()
######## TRAIN LOOP- MLFLOW

experimento_nn = 'exp_general'
run_name = 'cuestionario_aux'
epochs = 100

# armar optimizer y learning rate scheduler. Si se quiere eliminar scheduler hay que borrar todas las líneas
# que mencionen lr_scheduler

# Initialize the minimum validation loss and the patience counter
min_val_loss = float('inf')
patience = 8
counter = 0
val_loss_serie = []

with mlflow.start_run(run_name = run_name) as run:

    params = {
                "epochs": epochs,
                "learning_rate": 0.005,
                "batch_size": batch_size,
                "loss_function": criterion.__class__.__name__,
                "metric_function": metric_fn.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
            }

    mlflow.log_params(params)

    with open("model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(summary(model)))
        
    mlflow.log_artifact("model_summary.txt")
    

# Training loop
for epoch in range(epochs):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] training loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Validation loss
    validation_loss = 0.0
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            

    avg_val_loss = validation_loss / len(testloader)
    val_loss_serie.append(avg_val_loss)
    print('[%d] validation loss: %.3f' %
          (epoch + 1, avg_val_loss))

    # Check if the validation loss has improved
    if avg_val_loss < min_val_loss:
        print('Validation loss decreased from {:.3f} to {:.3f}. Saving model...'.format(min_val_loss, avg_val_loss))
        torch.save(model.state_dict(), dir_modelos / 'best_model_mix.pt')
        min_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        print('Validation loss did not improve. Patience: {}/{}'.format(counter, patience))
        if counter >= patience:
            print('Early stopping')
            break
print('Finished Training')
