import torch
torch.cuda.is_available()
import pandas as pd
from config.parse_config import ConfigParser
from simce.utils import read_json, prepare_device
import data_loader.data_loaders as module_data
from torchvision import models
import data_loader.data_loaders as module_data
from simce.modelamiento import preparar_capas_modelo
from tqdm import tqdm
from config.proc_img import get_directorios
from config.proc_img import SEED
import numpy as np
import pandas as pd
import argparse
import collections
## Predicciones -----------------------------------------------------
# config_dict = read_json('config/config_pred.json')
# config = ConfigParser(config_dict)
def main(config):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    dirs = get_directorios()
    device, _ = prepare_device(config['n_gpu'])

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


    loader = config.init_obj('data_loader', module_data, model=model_name, 
                                return_directory=True, dir_data=dirs['dir_train_test'])



    #model_load.eval()

    # Initialize lists to store predictions and true labels
    predictions = []
    probs = []
    lst_directories = []

    with torch.no_grad():
        # Iterate over the test data
        for n,(images, directories) in enumerate(tqdm(loader)):
            # Move the images and labels to the same device as the model
            images = images.to(device)


            # Make predictions
            
            outputs = model(images)

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


    resto_datos = pd.read_csv(dirs['dir_train_test'] / 'data_pred.csv')
    preds = pd.DataFrame({'pred': predictions,
                'proba': probs_float,
                'dirs': lst_directories})

    #preds['dirs'] = test.ruta_imagen_output

    preds_tot = preds.merge(resto_datos, left_on='dirs', right_on='ruta_imagen_output', how='left')
    preds_tot.drop(columns=['index', 'dm_sospecha']).sort_values('proba', ascending=False).to_excel(dirs['dir_predicciones'] / 'predicciones_modelo.xlsx')
    print('Predicciones exportadas exitosamente!')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Módulo de predicciones. Aquí se predice y se calculan indicadores de porcentaje e intensidad de tinta')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='directorio del archivo de configuración (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)