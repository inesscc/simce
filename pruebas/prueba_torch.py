import torch
torch.cuda.is_available()
import torchvision
import torchvision.transforms.v2 as v2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchinfo import summary
from config.proc_img import dir_modelos
import mlflow
print('Finished Training')
from config.proc_img import dir_train_test
import pandas as pd
from pathlib import Path
from torchvision.models import list_models
from config.parse_config import ConfigParser
from simce.utils import read_json, prepare_device
import model.model as module_arch
import data_loader.data_loaders as module_data
from trainer import Trainer
import model.metric as module_metric

config_dict = read_json('config/model.json')
config = ConfigParser(config_dict)

classification_models = list_models(module=torchvision.models)
train = pd.read_csv(dir_train_test / config['data_loader_train']['args']['data_file']) 
train.dm_final.value_counts()


logger = config.get_logger('train')
num_classes = 2

model  = config.init_obj('arch', module_arch, num_classes=num_classes)
trainloader = config.init_obj('data_loader_train', module_data, validation_split=.1)
valid_data_loader = trainloader.split_validation()
testloader = config.init_obj('data_loader_test', module_data)


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
metrics = [getattr(module_metric, met) for met in config['metrics']]

#optimizer = optim.Adam(model.parameters())
#optimizer = optim.SGD(model.parameters(), lr=0.004, momentum=0.9, weight_decay=.001)
# Mover modelo a dispositivo detectado
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = config.init_obj('optimizer_sgd', torch.optim, trainable_params)
lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=trainloader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

trainer.train()
######## TRAIN LOOP- MLFLOW

experimento_nn = 'exp_general'
run_name = 'cuestionario_aux'
epochs = 100

# armar optimizer y learning rate scheduler. Si se quiere eliminar scheduler hay que borrar todas las líneas
# que mencionen lr_scheduler

# Initialize the minimum validation loss and the patience counter
min_val_loss = float('inf')
patience = 8
counter = 0
val_loss_serie = []

with mlflow.start_run(run_name = run_name) as run:

    params = {
                "epochs": epochs,
                "learning_rate": 0.005,
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
for epoch in range(epochs):  # Loop over the dataset multiple times
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
config_dict = read_json('config/model.json')
config = ConfigParser(config_dict)
ruta_modelo = 'saved/models/AlexNet_modelo_base/mejor_modelo_actual/model_best.pt'
model_load  = config.init_obj('arch', module_arch, num_classes=num_classes)
checkpoint = torch.load(ruta_modelo)
state_dict = checkpoint['state_dict']
model_load.load_state_dict(state_dict)
model_load.to(device)


#model_load.eval()

# Initialize lists to store predictions and true labels
predictions = []
probs = []
true_labels = []

# Iterate over the test data
for images, labels in testloader:
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

test = pd.read_csv(dir_train_test / 'test.csv')


preds = pd.DataFrame({'pred': predictions,
              'true': true_labels,
              'proba': probs_float})
preds['dirs'] = test.ruta_imagen_output

preds_tot = preds.merge(test, left_on='dirs', right_on='ruta_imagen_output', how='left')
preds_tot['acierto'] = preds_tot.pred == preds_tot.true
preds[preds.true.eq(1) & preds.pred.eq(0)].reset_index().dirs.apply(lambda x: Path(x).name).value_counts()

preds_tot['deciles'] = pd.qcut(preds_tot.proba, q=20)
preds_tot.groupby('deciles').acierto.mean().plot()
plt.show()
preds_tot.acierto.mean()
preds_tot.proba.describe()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
preds_tot[preds_tot.proba.ge(.95) & preds_tot.true.eq(0)].acierto.mean()

def get_conf_mat(df, preg=None):
    if preg:
        df = df[df.preguntas.eq(preg)]

    cm = confusion_matrix(df.pred, df.true)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Marca normal', 'Doble marca'])
    disp.plot()
    plt.title(preg)
    plt.show()
    

get_conf_mat(preds_tot)
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
