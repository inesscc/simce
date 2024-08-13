import torch
from tqdm import tqdm
from torchvision import models
from simce.modelamiento import preparar_capas_modelo
from simce.utils import prepare_device

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def exportar_predicciones(preds, resto_datos, dirs):
    preds_tot = preds.merge(resto_datos, left_on='dirs', right_on='ruta_imagen_output', how='left')
    preds_tot.drop(columns=['index', 'dm_sospecha'])\
            .sort_values('proba', ascending=False)\
            .to_parquet(dirs['dir_predicciones'] / 'predicciones_modelo.parquet')
    
def prepare_model(config, device):
    

    model  = config.init_obj('arch', models)
    model_name = config['arch']['type']
    model = preparar_capas_modelo(model, model_name)

    ruta_modelo = 'saved/models/saved_server/maxvit/model_best_nuevo.pt'
    checkpoint = torch.load(ruta_modelo, map_location= device.type)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    model = model.to(device)

    model.eval()

    return model, model_name
def obtener_predicciones(loader, device, model):
    
    #model_load.eval()

    # Initialize lists to store predictions and true labels
    predictions = []
    probs = []
    lst_directories = []

    with torch.no_grad():
        # Iterate over the test data
        for (images, directories) in tqdm(loader):
            # Move the images and labels to the same device as the model
            images = images.to(device)


            # Make predictions
            
            outputs = model(images)
            print(outputs)
            probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
            max_probabilities = probabilities.max(dim=1)[0]
            
            # Get the predicted class for each image
            _, predicted = torch.max(outputs.data, 1)

            # Store the predictions and true labels
            predictions.extend(predicted.tolist())

            probs.extend(max_probabilities)
            lst_directories.extend(directories)
            break

    print('Predicciones listas!')
    probs_float = [i.item() for i in probs]

    return predictions, probs_float, lst_directories
