from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, filter_sospecha=False, return_directory=False):
        self.labels_frame = pd.read_csv(csv_file)

        if filter_sospecha:
            self.labels_frame = self.labels_frame[self.labels_frame['dm_sospecha'] == 1].reset_index(drop=True)


        self.transform = transform
        self.return_directory = return_directory

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_path = self.labels_frame.loc[idx, 'ruta_imagen_output']
        image = Image.open(img_path)
        label = self.labels_frame.loc[idx,  'dm_final']
        directory = self.labels_frame.loc[idx, 'ruta_imagen_output']



        if self.transform:
            image = self.transform(image)

        if self.return_directory:
            return image, label, directory
        else:
            return image, label