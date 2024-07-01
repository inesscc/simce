import torch.nn as nn




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
    else:
        raise('Código no está preparado para incorporar modelo. Modifique función para incorporarla')
    
    return model