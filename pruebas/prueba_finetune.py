import argparse
import collections
import torch
import torch.nn as nn
from config.proc_img import SEED
from config.parse_config import ConfigParser
from simce.utils import read_json, prepare_device
import model.model as module_arch
import data_loader.data_loaders as module_data
from trainer import Trainer
import model.metric as module_metric
import numpy as np
import torchvision.models as models

config_dict = read_json('config/model.json')
config = ConfigParser(config_dict)


def main(config):

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)




    logger = config.get_logger('train')
    num_classes = 2
    weights = config.init_obj('weights', models)
    model = config.init_obj('arch', models, weights=weights)

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
    nn.Linear(num_features, 256),  # Additional linear layer with 256 output features
    nn.ReLU(inplace=True),         # Activation function (you can choose other activation functions too)
    nn.Dropout(0.5),               # Dropout layer with 50% probability
    nn.Linear(256, num_classes)    # Final prediction fc layer
    )
    logger.info(model)

    trainloader = config.init_obj('data_loader_train', module_data)
    valid_data_loader = trainloader.split_validation()



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
    criterion = criterion.to(device)
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    #optimizer = optim.Adam(model.parameters())
    #optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=.001)
    # Mover modelo a dispositivo detectado
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    torch.cuda.empty_cache()
    trainer = Trainer(model, criterion, metrics, optimizer,
                        config=config,
                        device=device,
                        data_loader=trainloader,
                        valid_data_loader=valid_data_loader,
                        lr_scheduler=lr_scheduler)

    trainer.train()
    print('Finished Training')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
