import torch
import pytorch_lightning as pl
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
class iNaturalistDataModule(pl.LightningDataModule):
    def __init__(self, train_dir: str,test_dir: str, batch_size: int=128, num_workers: int = 2,train_transforms=transforms.ToTensor(), test_transforms=transforms.ToTensor(), train_val_split: float = 0.8,seed=3407):
      super().__init__()

      self.save_hyperparameters()
      self.train_dir = train_dir
      self.test_dir = test_dir
      self.batch_size = batch_size
      self.num_workers = num_workers
      self.train_val_split = train_val_split

      self.train_transforms = train_transforms
      self.test_transforms=test_transforms
      self.seed=seed

    def setup(self, stage=None):
      torch.manual_seed(self.seed)
      torch.cuda.manual_seed(self.seed)
      train_dataset = datasets.ImageFolder(root=self.train_dir, transform=self.train_transforms)
      total_size = len(train_dataset)
      train_size = int(total_size * self.train_val_split)
      val_size = total_size - train_size
      self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

      self.test_dataset = datasets.ImageFolder(root=self.test_dir, transform=self.test_transforms)

    def train_dataloader(self):
      return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):

      return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
      return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)

class iNaturalistDataModule_new(pl.LightningDataModule):
    def __init__(self, train_dir: str,test_dir: str, batch_size: int=128, num_workers: int = 2,train_transforms=transforms.ToTensor(), test_transforms=transforms.ToTensor(), train_val_split: float = 0.8,seed=3407):
      super().__init__()

      self.save_hyperparameters()
      self.train_dir = train_dir
      self.test_dir = test_dir
      self.batch_size = batch_size
      self.num_workers = num_workers
      self.train_val_split = train_val_split

      self.train_transforms = train_transforms
      self.test_transforms=test_transforms
      self.seed=seed

    def setup(self, stage=None):
      torch.manual_seed(self.seed)
      torch.cuda.manual_seed(self.seed)

      train_dataset = ImageFolderWithPaths(root=self.train_dir, transform=self.train_transforms)
      total_size = len(train_dataset)
      train_size = int(total_size * self.train_val_split)
      val_size = total_size - train_size
      self.train_dataset, self.val_dataset = random_split(train_dataset, [train_size, val_size])

      self.test_dataset = ImageFolderWithPaths(root=self.test_dir, transform=self.test_transforms)

    def train_dataloader(self):
      return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):

      return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
      return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)