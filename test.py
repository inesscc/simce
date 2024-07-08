import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric

from config.parse_config import ConfigParser
from simce.utils import read_json
import torch.nn as nn
import torchvision.models as models

from simce.modelamiento import preparar_capas_modelo, anotar_metricas_modelo

#config_dict = read_json('config/model_MaxVit_T_Weights.json')
#config = ConfigParser(config_dict)
def main(config):

    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = config.init_obj('data_loader_test', module_data,
                                   model=config['arch']['type'])
    # build model architecture
    model = config.init_obj('arch', models)
    model_name = config['arch']['type']
    model = preparar_capas_modelo(model, model_name)
    logger.info(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # get function handles of loss and metrics
    weight = config['class_weights'] 
    weight = torch.tensor(weight).to(device)
    loss_fn = config.init_obj('loss', nn, weight=weight)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))



    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']

    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size



    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    run_id = config.resume.parts[-2]
    config_name = config.resume.parts[-3]

    anotar_metricas_modelo(config_name, run_id, log)



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)