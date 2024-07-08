
import torchvision.transforms.v2 as v2
from base import BaseDataLoader
from dataset import CustomImageDataset
from config.proc_img import dir_train_test
import torch

class TrainTestDataLoader(BaseDataLoader):

    def __init__(self, data_file, batch_size, model, cortar_bordes, shuffle=True, validation_split=0.0, num_workers=2,
                 return_directory=False, ):
        self.model = model
        self.cortar_bordes = cortar_bordes

        if 'eficientnet' in model:
            transform = v2.Compose([
                lambda img: v2.Resize((560, 560))(img) if cortar_bordes else v2.Resize((480, 480))(img),
                v2.CenterCrop(480),
                v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # Normaliza valores a 0-1
                v2.Normalize(mean=[0.485,0.456, 0.406], std=[0.229,0.224, 0.225]) 

            ])

        else:
            transform = v2.Compose([
                        lambda img: v2.Resize((248, 248))(img) if cortar_bordes else v2.Resize((224, 224))(img),

                       # v2.Grayscale(num_output_channels=1),  # transformacion blanco negro
                        v2.CenterCrop(224),
                        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]), # Normaliza valores a 0-1
                        v2.Normalize(mean=[0.485,0.456, 0.406], std=[0.229,0.224, 0.225]) # Color
                        #v2.Normalize(mean=[0.5], std=[0.5]) #ByN
                    ])
        self.data_file = data_file
        self.return_directory = return_directory
        self.dataset = CustomImageDataset(csv_file=dir_train_test / self.data_file, transform=transform, return_directory=self.return_directory)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
