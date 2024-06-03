import torch
torch.cuda.is_available()
torch.cuda.device_count()
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torchmetrics import Accuracy
from torchinfo import summary


print('Finished Training')
from simce.config import dir_tabla_99
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os.path


import mlflow

tipo_cuadernillo = 'padres'
nombre_tabla_casos99 = f'casos_99_entrenamiento_compilados_{tipo_cuadernillo}.csv'
df99 = pd.read_csv(dir_tabla_99 / nombre_tabla_casos99)
df99.dm_final.value_counts()
df99.head()
df_exist = df99[df99.ruta_imagen_output.apply(lambda x: Path(x).is_file())].reset_index()
nombre_tabla_casos99 = 'prueba_torch.csv'
df_exist.to_csv(dir_tabla_99 / 'prueba_torch.csv')


#### 

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.labels_frame.loc[idx, 'ruta_imagen_output'])
        image = Image.open(img_path)
        label = self.labels_frame.loc[idx,  'dm_final']
        directory = self.labels_frame.loc[idx, 'ruta_imagen_output']

        if self.transform:
            image = self.transform(image)

        return image, label, directory
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomImageDataset(csv_file=dir_tabla_99 / nombre_tabla_casos99, root_dir='', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


dataiter = iter(dataloader)
images, labels, dirs = next(dataiter)
dirs[0]

####



from torch.utils.data import random_split

# Let's say the total size of your dataset is `dataset_size`
dataset_size = len(dataset)
test_size = int(dataset_size * 0.2)  # Let's reserve 20% of the data for the test set
train_size = dataset_size - test_size
batch_size = 32

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True
                                          #, num_workers=2
                                          )
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True
                                          #, num_workers=2
                                          )

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


num_classes = 2
model = SimpleCNN(num_classes=num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move the model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


######## TRAIN LOOP- MLFLOW

experimento_nn = 'exp_general'
run_name = 'cuestionario_aux'
epoch = 2
metric_fn = Accuracy(task="multiclass", num_classes=2).to(device)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(experimento_nn)

with mlflow.start_run(run_name = run_name) as run:

    params = {
                "epochs": epoch,
                "learning_rate": 1e-3,
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
    ## idealmente ponerlo despues en una funcion fuera jiji
    for epoch in range(10):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            print(i)
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accuracy = metric_fn(outputs, labels)
            
            #print(accuracy)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 20 == 19:  # Print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
                step = i // 20 * (epoch + 1)
                mlflow.log_metric("loss", f"{loss:2f}", step=step)
                mlflow.log_metric("accuracy", f"{accuracy:2f}", step=step)

        mlflow.end_run()

