import torch
from tqdm import tqdm
from torchvision import models
from simce.modelamiento import preparar_capas_modelo
import pandas as pd
from os import PathLike
import os
from torchvision.models.maxvit import MaxVit
from data_loader.data_loaders import TrainTestDataLoader
from torch.nn import DataParallel
import pathlib
from torch import device
from config.parse_config import ConfigParser
# Ajuste a Path si se está ejecutando script en Windows
if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def exportar_predicciones(preds:pd.DataFrame, resto_datos:pd.DataFrame, dirs:list[PathLike]):
    '''
    Exporta tabla con predicciones a un objeto de tipo [parquet](https://parquet.apache.org). 
    **No retorna nada**.
    
    Args:
        preds: tabla con predicciones y directorios asociados a las subpreguntas predichas.
        
        resto_datos: tabla con otros datos asociados a subpreguntas predichas.

        dirs: lista de directorios del proyecto. 
    
    '''
    preds_tot = preds.merge(resto_datos, left_on='dirs', right_on='ruta_imagen_output', how='left')
    preds_tot.drop(columns=['index', 'dm_sospecha'])\
            .sort_values('proba', ascending=False)\
            .to_parquet(dirs['dir_predicciones'] / 'predicciones_modelo.parquet')
    
def prepare_model(config: ConfigParser, device: device)-> tuple[MaxVit|DataParallel, str]:
    '''
    Carga el modelo y realiza preparaciones en torno a esto. Retorna tanto el modelo como el nombre del modelo.
    
    Args:
        config: tabla con predicciones y directorios asociados a las subpreguntas predichas.
        
        device: tabla con otros datos asociados a subpreguntas predichas.

    Returns:
        model: objeto con el modelo cargado.

        model_name: nombre del modelo en string.
    '''   

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

def obtener_predicciones(loader:TrainTestDataLoader, device: device,
                          model:MaxVit|DataParallel)-> tuple[list[int], list[float], list[str]]:
    '''
    Obtiene predicciones del modelo.

    Args:
        loader: Data loader que carga datos a predecir.

        device: indica si se usa CPU o GPU.

        model: modelo que realizará predicciones.

    Returns:
        predictions: lista de predicciones del modelo (doble marca / falsa sospecha)
        
        probs_float: lista de probabilidades asignadas a cada predicción del modelo.
        
        lst_directories: lista de directorios asociados a cada predicción.
    '''
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
            

    print('Predicciones listas!')
    probs_float = [i.item() for i in probs]

    return predictions, probs_float, lst_directories
