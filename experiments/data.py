import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import pytorch_lightning as pl

from typing import Optional

CIFAR10_MEAN = torch.tensor([0.49156195, 0.48253155, 0.44702223], dtype=float) #mean of each RGB channel
CIFAR10_STD  = torch.tensor([0.24701314, 0.24346219, 0.26157323], dtype=float)


class FastCIFAR10(CIFAR10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).
    
    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """
    
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        
        super().__init__(*args, **kwargs)
        
        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255) #Rescale to [0, 1]
        
        #self.data = self.data.sub_(CIFAR10_MEAN).div_(CIFAR10_STD) #(NOT) Normalize to 0 centered with 1 std 
        
        self.data = torch.moveaxis(self.data, -1, 1) #-> set dim to: (batch, channels, height, width)
    
    def __getitem__(self, index : int):
        """ 
        Parameters
        ----------
        index : int
            Index of the element to be returned
        
        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """

        img = self.data[index]
        target = self.targets[index]

        return img, target


class MapDataset(torch.utils.data.Dataset):

    def __init__(self,
                 dataset : "torch.utils.data.Dataset",
                 transform = None):
        """
        Given a dataset of tuples (features, labels),
        returns a new dataset with a transform applied to the features (lazily, only when an item is called).

        Note that data is not cloned/copied from the initial dataset.
        
        Parameters
        ----------
        dataset : "torch.utils.data.Dataset"
            Dataset of tuples (features, labels)
        transform : function
            Transformation applied to the features of the original dataset
        """
    
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:     
            x = self.transform(self.dataset[index][0]) 
        else:     
            x = self.dataset[index][0]  # image
            
        y = self.dataset[index][1]   # label      
        return x, y

    def __len__(self):
        return len(self.dataset) 
    
class CIFAR10DataModule(pl.LightningDataModule):

    def __init__(self, batch_size : int = 64, train_data_transform : "torchvision.transforms" = None):
        super().__init__()

        self.batch_size = batch_size
        self.train_data_transform = train_data_transform
    
    def setup(self, stage: Optional[str] = None):
        self.test_dataset = FastCIFAR10('dataset', train=False, download=True)

        self.full_train_dataset = FastCIFAR10('dataset', train=True, download=True)
        
        #Train-Validation split
        n_samples = len(self.full_train_dataset)
        n_training = 40000 # + 10000 for validation

        self.train_dataset, self.val_dataset = random_split(self.full_train_dataset, [n_training, n_samples-n_training])

        if self.train_data_transform is not None:
            self.train_dataset = MapDataset(self.train_dataset, self.train_data_transform) #Apply transform just to the training part of the data


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)