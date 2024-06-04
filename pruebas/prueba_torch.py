import torch
torch.cuda.is_available()
import torchvision
import torchvision.transforms.v2 as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from simce.config import dir_modelos


print('Finished Training')
from simce.config import dir_tabla_99
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os.path
from torchvision.models import list_models

classification_models = list_models(module=torchvision.models)
padres99 = f'casos_99_entrenamiento_compilados_padres.csv'
est99 = f'casos_99_entrenamiento_compilados_estudiantes.csv'
df99p = pd.read_csv(dir_tabla_99 / padres99)

df99e = pd.read_csv(dir_tabla_99 / est99).sample(frac=.1, random_state=42)
df99 = pd.concat([df99e, df99p]).reset_index(drop=True)

df_exist = df99[df99.ruta_imagen_output.apply(lambda x: Path(x).is_file())].reset_index()
df_exist.dm_sospecha = (df_exist.dm_sospecha == 99).astype(int)
nombre_tabla_casos99 = 'prueba_torch.csv'
df_exist.to_csv(dir_tabla_99 / 'prueba_torch.csv')

class CustomImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, filter_sospecha=False):
        self.labels_frame = pd.read_csv(csv_file)

        if filter_sospecha:
            self.labels_frame = self.labels_frame[self.labels_frame['dm_sospecha'] == 1].reset_index(drop=True)

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
    
transformations_random = [
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Randomly adjust color
    transforms.RandomHorizontalFlip()        # Randomly flip the image horizontally
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomApply(transformations_random, p=.1),    
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


    
dataset = CustomImageDataset(csv_file=dir_tabla_99 / nombre_tabla_casos99, root_dir='', transform=transform,
                             filter_sospecha=True)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


dataiter = iter(dataloader)
images, labels, dirs = next(dataiter)
dirs[0]


from torch.utils.data import random_split

# Let's say the total size of your dataset is `dataset_size`
dataset_size = len(dataset)
test_size = int(dataset_size * 0.2)  # Let's reserve 20% of the data for the test set
train_size = dataset_size - test_size
batch_size = 32

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

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

# Select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'{device=}')
# Define the loss function and optimizer
weights = [50.0, 1.0] 
weights = torch.tensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
#optimizer = optim.Adam(model.parameters(), lr=0.0005)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=.001)
# Mover modelo a dispositivo detectado
model = model.to(device)


# Initialize the minimum validation loss and the patience counter
min_val_loss = float('inf')
patience = 8
counter = 0
val_loss_serie = []

# Training loop
for epoch in range(100):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # Print every 200 mini-batches
            print('[%d, %5d] training loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # Validation loss
    validation_loss = 0.0
    for i, data in enumerate(testloader, 0):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            

    avg_val_loss = validation_loss / len(testloader)
    val_loss_serie.append(avg_val_loss)
    print('[%d] validation loss: %.3f' %
          (epoch + 1, avg_val_loss))

    # Check if the validation loss has improved
    if avg_val_loss < min_val_loss:
        print('Validation loss decreased from {:.3f} to {:.3f}. Saving model...'.format(min_val_loss, avg_val_loss))
        torch.save(model.state_dict(), dir_modelos / 'best_model_mix.pt')
        min_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        print('Validation loss did not improve. Patience: {}/{}'.format(counter, patience))
        if counter >= patience:
            print('Early stopping')
            break
print('Finished Training')




## Predicciones -----------------------------------------------------


# Set the model to evaluation mode
model_load = SimpleCNN(num_classes=2)
# Load the model parameters from a saved state
model_load.load_state_dict(torch.load('best_model_mix.pt'))
model_load = model_load.to(device)
#model_load.eval()

# Initialize lists to store predictions and true labels
predictions = []
probs = []
true_labels = []

# Iterate over the test data
for images, labels, dirs in testloader:
    # Move the images and labels to the same device as the model
    images = images.to(device)
    labels = labels.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model_load(images)

    probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
    max_probabilities = probabilities.max(dim=1)[0]
    
    # Get the predicted class for each image
    _, predicted = torch.max(outputs.data, 1)

    # Store the predictions and true labels
    predictions.extend(predicted.tolist())
    true_labels.extend(labels.tolist())
    probs.extend(max_probabilities)

print('Predicciones listas!')
probs_float = [i.item() for i in probs]
dirs = [i[2] for i in test_dataset]

preds = pd.DataFrame({'pred': predictions,
              'true': true_labels,
              'dirs': dirs,
              'proba': probs_float}).set_index('dirs')

preds_tot = preds.merge(df_exist, left_on='dirs', right_on='ruta_imagen_output', how='left')
preds_tot['acierto'] = preds_tot.pred == preds_tot.true
casi_dm = preds[(preds.dm_sospecha.eq(99)) & (preds.true.eq(0))]
preds[preds.true.eq(1) & preds.pred.eq(0)].reset_index().dirs.apply(lambda x: Path(x).name).value_counts()

preds_tot['deciles'] = pd.qcut(preds_tot.proba, q=10)
preds_tot.groupby('deciles').acierto.mean().plot()
preds_tot.acierto.mean()
preds_tot.proba.describe()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
preds_tot[preds_tot.proba.ge(.95) & preds_tot.true.eq(0)].acierto.mean()

def get_conf_mat(df, preg):

    df_filter = df[df.preguntas.eq(preg)]

    cm = confusion_matrix(df_filter.pred, df_filter.true)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Marca normal', 'Doble marca'])
    disp.plot()
    plt.title(preg)
    plt.show()
    

get_conf_mat(preds_tot, 'p22')
preds_tot.groupby(['preguntas']).agg({'rbd':'count', 'acierto':'mean'}).sort_values('acierto')

casi_dm.pred.value_counts()
df_exist[df_exist.preguntas.eq('p22')]

df_exist.preguntas.value_counts()

########## --------------------------
########## --------------------------

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
""" dirs[1]
images_denom = image_denormalized = denormalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
imshow(torchvision.utils.make_grid(images))
import cv2



transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])




trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
 """

""" # get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) """