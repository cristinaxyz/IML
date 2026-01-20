import torch
import torchvision
from torchvision.transforms import ToPILImage
import torchvision.transforms as T
from torch.utils.data import Dataset
import pathlib 

BATCH_SIZE = 16
IMAGE_SIZE = (256, 256)
DATA_DIR = pathlib.Path('dataset')


class PreprocessingData(Dataset):
    def __init__(self, seed, transform=True):
        self.data_dir = DATA_DIR
        self.transform = transform
        self.batch_size = BATCH_SIZE
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset = torchvision.datasets.ImageFolder(
            self.data_dir,
            transform=self._transform(train=True))
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def _transform(self, train=True):
        if train:
            return T.Compose([
                T.Resize(IMAGE_SIZE),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize(IMAGE_SIZE),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ])
    
    def split_data(self):
        dataset = self.dataset
        n = len(dataset)
        train_len = int(0.8 * n)
        val_len   = int(0.1 * n)
        test_len  = n - train_len - val_len

        train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [train_len, val_len, test_len], self.generator)
        
        # validation and test data shouldnt get tranfsomerd
        train_ds.dataset.transform = self._transform(train=True)
        val_ds.dataset.transform = self._transform(train=False)
        test_ds.dataset.transform = self._transform(train=False)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, generator=self.generator)
        val_loader   = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

datamodel = PreprocessingData(3404, transform=True)
train_loader, val_loader, test_loader = datamodel.split_data()

" if you want to see the images"
#for images, labels in train_loader:
#    image = ToPILImage()(images[0])
#    image.save("sample_image.png")
