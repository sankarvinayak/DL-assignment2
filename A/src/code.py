import os, random,torch,torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchvision import transforms, datasets
from PIL import Image
from torch.utils.data import DataLoader, random_split
from torchvision import models
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torch.nn import init




############################Question1
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

class iNaturalistModel(pl.LightningModule):
    def __init__(self, in_channels=3,input_size=(224, 224),dense_size=512,  num_filters_layer: list = [16, 32, 64, 128, 256],filter_size: list = [3, 3, 3, 3, 3], stride=1,  padding=1, num_classes=10,activation=nn.ReLU,dropout_rate=0,optimizer=torch.optim.Adam,lr=0.001,batch_norm=False ):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer=optimizer
        self.lr=lr
        layers = []
        size = input_size
        for num_channel, k in zip(num_filters_layer, filter_size):
            layers.append(nn.Conv2d(in_channels=in_channels,out_channels=num_channel,kernel_size=k,stride=stride,padding=padding))
            if batch_norm:
              layers.append(nn.BatchNorm2d(num_features=num_channel))
            layers.append(activation())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_channel
            conv_h = (size[0] + 2 * padding - k) // stride + 1
            conv_w = (size[1] + 2 * padding - k) // stride + 1
            pool_h = conv_h // 2
            pool_w = conv_w // 2
            size = (pool_h, pool_w)
        layers.append(nn.Flatten())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        flattened_size = size[0] * size[1] * num_filters_layer[-1]
        layers.append(nn.Linear(in_features=flattened_size, out_features=dense_size))
        layers.append(activation())
        layers.append(nn.Linear(in_features=dense_size, out_features=num_classes))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)
    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("validation_acc", acc, prog_bar=True)
        self.log("validation_loss", loss, prog_bar=True)
        return {"validation_loss":loss,"validation_acc":acc}
    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self.forward(images)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss":loss,"test_acc":acc}
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr)
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()])
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',train_transforms=transform,test_transforms=transform)
naturalist_DM.setup()
train_loader=naturalist_DM.train_dataloader()
val_loader=naturalist_DM.val_dataloader()
test_loader=naturalist_DM.test_dataloader()

img = Image.open("inaturalist_12K/train/Amphibia/02f95591e712f05cae136a91a4d73ea5.jpg")
width, height = img.size
print(f"Width: {width}, Height: {height}")
import torch
mean = torch.zeros(3)
std = torch.zeros(3)
total_samples = 0
for images, _ in train_loader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, 3, -1)

    mean += images.mean(dim=2).sum(dim=0)
    std += images.std(dim=2).sum(dim=0)
    total_samples += batch_samples

mean /= total_samples
std /= total_samples

print(f"Computed Mean: {mean.tolist()}")
print(f"Computed Std: {std.tolist()}")

train_loader = naturalist_DM.train_dataloader()
shape=None
for images, labels in train_loader:
    print("Batch image shape:", images.shape)
    shape=images.shape
    break
print(shape)



#######################################Question 2
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
naturalist_DM=iNaturalistDataModule(train_dir='/content/inaturalist_12K/train',test_dir='/content/inaturalist_12K/val',train_transforms=transform,test_transforms=transform)
naturalist_DM.setup()

model=iNaturalistModel(dense_size=4096,num_filters_layer = [32,64,128,256,512],filter_size = [3, 3, 3, 3, 3], stride=1,  padding=1, num_classes=10,activation=torch.nn.ReLU,dropout_rate=0.5,optimizer=torch.optim.Adam,lr=1e-4,batch_norm=True )
trainer=pl.Trainer(max_epochs=30)
trainer.fit(model,datamodule=naturalist_DM)


import wandb
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "num_filter": {
            "values":[16,32,64]
        },
        "activation_fun": {
            "values":["ReLU", "GELU", "SiLU", "Mish"]
        },
        "filter_org": {
            "values":["same","double"]
        },
        "data_augmentation": {
            "values":["No","Yes"]
        },
        "batch_norm": {
            "values":["No","Yes"]
        },
        "dropout": {
            "values":[0,0.2,0.3,0.5,0.7]
        },

        "filter_size": {
            "values":[3,5,7]
        },
        "epoch": {
            "values":[5,10,20]
        },
        "batch_size": {
            "values":[32,64,128,256]
        },"dene_size": {
            "values":[512,1024,2046]
        },"lr": {
            "values":[1e-3,5e-4,1e-4]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="DL-Addignemt2_A")



from pytorch_lightning.loggers import WandbLogger
def train_model():
    torch.manual_seed(3407)
    wandb.init()
    config = wandb.config
    run_name = f"Augment{config.data_augmentation}activation_fun{config.activation_fun}_dropout_{config.dropout}"
    wandb.run.name = run_name

    if config.data_augmentation == "Yes":
      train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
      transforms.RandomRotation(degrees=15),
      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
      transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
    else:
      train_transforms = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    num_filters=config.num_filter
    num_filters_layer=[num_filters,num_filters*2,num_filters*4,num_filters*8,num_filters*16]if config.filter_org=="double" else [num_filters]*5
    if config.activation_fun=="ReLU":
      activation=torch.nn.ReLU
    elif config.activation_fun=="GELU":
      activation=torch.nn.GELU
    elif config.activation_fun=="SiLU":
      activation=torch.nn.SiLU
    elif config.activation_fun=="Mish":
      activation=nn.Mish
    filter_size=config.filter_size
    filter_sizes=[filter_size]*5
    if config.batch_norm=="Yes":
      model = iNaturalistModel(num_filters_layer=num_filters_layer,activation=activation,dropout_rate=config.dropout,batch_norm=True,filter_size=filter_sizes,dense_size=config.dense_size,lr=config.lr)
    else:
      model = iNaturalistModel(num_filters_layer=num_filters_layer,activation=activation,dropout_rate=config.dropout,filter_size=filter_sizes,dense_size=config.dense_size,lr=config.lr)
    print(model)



    wandb_logger = WandbLogger(project="DL-Addignemt2_A",name=run_name)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=config.epoch)

    test_tranform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
    naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=config.batch_size,train_transforms=train_transforms,test_transforms=test_tranform)
    trainer.fit(model, naturalist_DM)
import wandb
wandb.login()

import wandb
wandb.agent(sweep_id, function=train_model)

import wandb

sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "num_filter": {
            "values": [16, 32, 64, 128]
        },
        "activation_fun": {
            "values": ["ReLU", "GELU", "SiLU", "Mish"]
        },
        "filter_org": {
            "values": ["same", "double", "half"]
        },
        "data_augmentation": {
            "values": ["No", "Yes"]
        },
        "batch_norm": {
            "values": ["No", "Yes"]
        },
        "dropout": {
            "values": [0, 0.2, 0.3, 0.5, 0.7]
        },
        "filter_size": {
            "values": [
                [3, 3, 3, 3, 3],
                [3, 3, 3, 5, 5],
                [5, 5, 7, 7, 7],
                [3, 5, 5, 7, 7],
                [7, 7, 5, 5, 3],
                [7, 5, 5, 3, 3],
                [5, 5, 7, 7, 7],
                [3, 5, 5, 7, 7],
                [3, 3, 5, 5, 7]
            ]
        },
        "epoch": {
            "values": [5, 10, 20]
        },
        "batch_size": {
            "values": [32, 64, 128]
        },
        "dense_size": {
            "values": [512, 1024, 2046]
        },
        "lr": {
            "values": [1e-3, 5e-4, 1e-4]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="DL-Addignemt2_A")


from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
def train_model():
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    wandb.init()
    config = wandb.config
    run_name = f"Augment{config.data_augmentation}activation_fun{config.activation_fun}_dropout_{config.dropout}"
    wandb.run.name = run_name

    if config.data_augmentation == "Yes":
      train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
      transforms.RandomRotation(degrees=15),
      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
      transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
    else:
      train_transforms = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
    num_filters=config.num_filter
    if config.filter_org=="double":
      num_filters_layer=[num_filters,num_filters*2,num_filters*4,num_filters*8,num_filters*16]
    elif config.filter_org=="half":
      num_filters_layer=[num_filters,num_filters//2,num_filters//4,num_filters//8,num_filters//16]
    else:
     num_filters_layer=[num_filters]*5
    if config.activation_fun=="ReLU":
      activation=torch.nn.ReLU
    elif config.activation_fun=="GELU":
      activation=torch.nn.GELU
    elif config.activation_fun=="SiLU":
      activation=torch.nn.SiLU
    elif config.activation_fun=="Mish":
      activation=nn.Mish
    filter_sizes=config.filter_size
    # filter_sizes=[filter_size]*5
    if config.batch_norm=="Yes":
      model = iNaturalistModel(num_filters_layer=num_filters_layer,activation=activation,dropout_rate=config.dropout,batch_norm=True,filter_size=filter_sizes,dense_size=config.dense_size,lr=config.lr)
    else:
      model = iNaturalistModel(num_filters_layer=num_filters_layer,activation=activation,dropout_rate=config.dropout,filter_size=filter_sizes,dense_size=config.dense_size,lr=config.lr)
    print(model)

    early_stop_cb = EarlyStopping(
    monitor="validation_loss",
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode="min"
)

    wandb_logger = WandbLogger(project="DL-Addignemt2_A",name=run_name)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=config.epoch,callbacks=[early_stop_cb])
    test_tranform=transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
    naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=config.batch_size,train_transforms=train_transforms,test_transforms=test_tranform)
    trainer.fit(model, naturalist_DM)





#################################### Question 4

from torchvision.datasets import ImageFolder

from torchvision.datasets import ImageFolder

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
naturalist_DM_new = iNaturalistDataModule_new(
    train_dir='inaturalist_12K/train',
    test_dir='inaturalist_12K/val',
    batch_size=batch_size,
    train_transforms=train_transforms,
    test_transforms=test_transforms
)
naturalist_DM_new.setup()

class_names = naturalist_DM_new.test_dataset.classes
test_loader = naturalist_DM_new.test_dataloader()

show_prediction_grid(model, test_loader, class_names, device='cuda')

import torch
import random
import matplotlib.pyplot as plt
from PIL import Image

def show_random_predictions(model, dataset, class_names, device='cuda', num_samples=30, rows=3, cols=10):
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]

    fig, axes = plt.subplots(rows, cols, figsize=(20, 6))
    axes = axes.flatten()

    with torch.no_grad():
        for i, (img, label, path) in enumerate(samples):
            img_tensor = img.unsqueeze(0).to(device)  
            output = model(img_tensor)
            pred = output.argmax(dim=1).item()
            img_display = Image.open(path).convert("RGB")

            ax = axes[i]
            ax.imshow(img_display)
            ax.axis("off")
            ax.set_title(f"Pred: {class_names[pred]}\nActual: {class_names[label]}", fontsize=8)

    plt.tight_layout()
    plt.show()



show_random_predictions(model, naturalist_DM_new.test_dataset, class_names, device='cuda')


import random
import torch
from PIL import Image
import wandb

def log_random_predictions_separate(
    model,
    dataset,
    class_names,
    device='cuda',
    num_samples=30,
    key="random_preds"
):

    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    samples = [dataset[i] for i in indices]

    images_to_log = []
    with torch.no_grad():
        for img_tensor, label, path in samples:
     
            inp = img_tensor.unsqueeze(0).to(device)
            output = model(inp)
            pred = output.argmax(dim=1).item()

            img = Image.open(path).convert("RGB")
            caption = f"Pred: {class_names[pred]} / Actual: {class_names[label]}"
            images_to_log.append(wandb.Image(img, caption=caption))

    wandb.log({ key: images_to_log })
wandb.init(project="DL-Addignemt2_A")
log_random_predictions_separate(
    model,
    naturalist_DM_new.test_dataset,      
    class_names,     
    device='cuda',
    num_samples=30,
    key="Model prediction"
)
wandb.finish()


