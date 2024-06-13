
import torchvision.transforms.v2 as v2
from base import BaseDataLoader
from dataset import CustomImageDataset
from config.proc_img import dir_train_test
import torch

class TrainTestDataLoader(BaseDataLoader):

    def __init__(self, data_file, batch_size, shuffle=True, validation_split=0.0, num_workers=2):
        transform = v2.Compose([
                    v2.Resize((224, 224)),
                    v2.Grayscale(num_output_channels=3),  # transformacion blanco negro
                    #v2.CenterCrop(224),
                    v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), 
                    v2.Normalize(mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229])
                ])
        self.data_file = data_file
        self.dataset = CustomImageDataset(csv_file=dir_train_test / self.data_file, transform=transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
