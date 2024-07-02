import torch.nn as nn
from pathlib import Path
from openpyxl import load_workbook, Workbook


def preparar_capas_modelo(model, modelo_seleccionado):
    num_classes = 2
    print(f'{modelo_seleccionado=}')
    if modelo_seleccionado == 'vgg16':           
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)
    elif modelo_seleccionado == 'maxvit_t':
        num_features = model.classifier[5].in_features
        model.classifier[5] = nn.Linear(num_features, num_classes)
    elif modelo_seleccionado == 'wide_resnet101_2':
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif 'efficientnet' in modelo_seleccionado:
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

    else:
        raise('Código no está preparado para incorporar modelo. Modifique función para incorporarla')
    
    return model

def get_metricas_modelo(config_name, run_id, val_loss, val_acc, train_loss, train_acc):
    if not Path('metricas_modelos.xlsx').is_file():
        wb = Workbook()
        ws = wb.active
        ws['A1'] = 'config_name'
        ws['B1'] = 'run_id'
        ws['C1'] = 'val_loss'
        ws['D1'] = 'val_acc'
        ws['E1'] = 'train_loss'
        ws['F1'] = 'train_acc'
        wb.save('metricas_modelos.xlsx')

    wb = load_workbook(filename='metricas_modelos.xlsx')

    ws = wb.active

    ws.append([config_name, run_id, val_loss, val_acc, train_loss, train_acc])