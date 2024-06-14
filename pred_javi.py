import torch
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torchvision.models as models
from PIL import Image

import pandas as pd

### def functions -------------------------------------------------------
def preprocess_image(image_paths):
    """
    prepare input model
    """
    transform = v2.Compose([
                    v2.Resize((224, 224)),
                    v2.Grayscale(num_output_channels=3),  # transformacion blanco negro
                    #v2.CenterCrop(224),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), 
                    v2.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
                ])
    image = Image.open(image_paths)
    return transform(image).unsqueeze(0)

    # ESTA FORMA OCUPA MUCHO ESPACIO T-T
    #images = []    
    #for image_path in image_paths:
    #    image = Image.open(image_path)
    #    image = transform(image)
    #    images.append(image)
    
    #return torch.stack(images) 
    


def predict(image_tensor, model, device):
    """retorna las predicciones, ya sean etiquetas como 'proba'"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_label = torch.max(outputs, 1)
    return predicted_label.item(), probabilities.squeeze().tolist()



# estructura modelo tuneado
class MaxViTModel(nn.Module):
    """define model"""
    def __init__(self, num_classes):
        super(MaxViTModel, self).__init__()

        num_classes = 2
        weights = models.MaxVit_T_Weights.DEFAULT
        self.model = models.maxvit_t(weights=weights)
        
        # ultima capa, 2 clases
        num_features = self.model.classifier[5].in_features
        self.model.classifier[5] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

def load_model(model_path, num_classes):

    model = MaxViTModel(num_classes)

    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()  # Poner el modelo en modo de evaluacion
    return model


# Ej --------------------------------------------------------------------------------

## load model
path_model = 'saved/models/maxvit_t/0612_211050/model_best.pt'
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_model(path_model, num_classes).to(device)

### una imgs --------------
image_path = 'data/input_proc/subpreg_recortadas/base/4b/CE/00045/4001770_p27_3.jpg'
image_tensor = preprocess_image(image_path)

# pred
pred_label, pred_proba = predict(image_tensor, model, device)
print(pred_label)
print(pred_proba)

### varias imgs ------------
import time

data_test = pd.read_csv('data/input_modelamiento/test.csv')
predicciones = []

start_time = time.time()
for id in range(data_test.shape[0]):
    print(id)
    img = data_test.iloc[id]
    img_proc = preprocess_image(img['ruta_imagen_output'])

    pred_label, pred_proba = predict(img_proc, model, device)
    predicciones.append({
        'pred_label': pred_label,
        'pred_proba': pred_proba
    })

end_time = time.time()
total_time = end_time - start_time

print(predicciones)
print(total_time)

# save 
pd.concat([data_test, pd.DataFrame(predicciones)], axis = 1).to_csv('test_pred.csv')
