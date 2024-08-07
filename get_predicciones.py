import torch
torch.cuda.is_available()
import pandas as pd
from config.parse_config import ConfigParser
from simce.utils import read_json, prepare_device
from simce.indicadores_tinta import get_indices_tinta 
import data_loader.data_loaders as module_data

from simce.predicciones import obtener_predicciones, exportar_predicciones, prepare_model

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
    device, _ = prepare_device(config['n_gpu'])
    dirs = get_directorios()
    
    model, model_name = prepare_model(config, device)



    loader = config.init_obj('data_loader', module_data, model=model_name, 
                                return_directory=True, dir_data=dirs['dir_train_test'])
    predictions, probs_float, lst_directories = obtener_predicciones(loader, device, model)
    resto_datos = pd.read_csv(dirs['dir_train_test'] / 'data_pred.csv')
    preds = pd.DataFrame({'pred': predictions,
                'proba': probs_float,
                'dirs': lst_directories})
    

    

    #preds['dirs'] = test.ruta_imagen_output

    exportar_predicciones(preds, resto_datos, dirs)
    
    print('Predicciones exportadas exitosamente!')

    print('Obteniendo indicadores de tinta')

    get_indices_tinta(dirs)

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