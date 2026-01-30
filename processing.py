import pathlib
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset

BATCH_SIZE = 4
IMAGE_SIZE = (256, 256)
DATA_DIR = pathlib.Path("dataset")


class TransformSubset(Dataset):
    """Apply transforms to a Subset without creating separate ImageFolder objects"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

def compute_mean_std():
    loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            DATA_DIR, transform=T.Compose([T.Resize(IMAGE_SIZE), T.ToTensor()])
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total = 0

    for images, _ in loader:
        batch_size = images.size(0)
        mean += images.mean(dim=[0, 2, 3]) * batch_size
        std += images.std(dim=[0, 2, 3]) * batch_size
        total += batch_size

    mean = mean / total
    std = std / total

    return mean.tolist(), std.tolist()

# Precomputed mean and std for the dataset
# gotten by running compute_mean_std()
mean = [0.5169, 0.5253, 0.5061]
std = [0.2330, 0.2237, 0.2404]

class PreprocessingData:
    def __init__(self, seed):
        self.data_dir = DATA_DIR
        self.batch_size = BATCH_SIZE
        self.generator = torch.Generator().manual_seed(seed)
        self.dataset = torchvision.datasets.ImageFolder(self.data_dir)
       
    def _transform(self, train=True):
        if train:
            return T.Compose(
                [
                    T.Resize(IMAGE_SIZE),
                    T.RandomHorizontalFlip(),
                    T.RandomRotation(10),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            )
        else:
            return T.Compose(
                [
                    T.Resize(IMAGE_SIZE),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ]
            )

    def split_data(self):
        dataset = self.dataset
        n = len(dataset)
        train_len = int(0.8 * n)
        val_len = int(0.1 * n)
        test_len = n - train_len - val_len

        indices = torch.randperm(n, generator=self.generator).tolist()
        train_idx = indices[:train_len]
        val_idx = indices[train_len : train_len + val_len]
        test_idx = indices[train_len + val_len :]
        
        train_ds = TransformSubset(
            torch.utils.data.Subset(dataset, train_idx),
            self._transform(train=True)
        )
        val_ds = TransformSubset(
            torch.utils.data.Subset(dataset, val_idx),
            self._transform(train=False)
        )
        test_ds = TransformSubset(
            torch.utils.data.Subset(dataset, test_idx),
            self._transform(train=False)
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=self.generator,
            num_workers=4,
        )

        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        return train_loader, val_loader, test_loader



