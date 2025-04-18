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

######################Question 1
weights=torchvision.models.ViT_B_16_Weights.DEFAULT
auto_transforms=weights.transforms()
model=torchvision.models.vit_b_16(weights=weights)

import torch.nn as nn
model.heads=nn.Sequential(nn.Linear(in_features=768, out_features=10))
print(model)
from torchvision import models
class VIT_iNaturalist(pl.LightningModule):
    def __init__(self,num_classes=10,optimizer=torch.optim.Adam,lr=0.001,weights=models.ViT_B_16_Weights.DEFAULT):
      super().__init__()
      self.optimizer=optimizer
      self.lr=lr
      self.model =models.vit_b_16(weights=weights)
      self.model.heads=nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))
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
        self.log("test_acc", acc, prog_bar=True,)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss":loss,"test_acc":acc}
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-5)

class Transfer_iNaturalistModel(pl.LightningModule):
    def __init__(self,num_classes=10,optimizer=torch.optim.Adam,lr=0.001,base=models.resnet50(weights="DEFAULT")):
      super().__init__()
      self.optimizer=optimizer
      self.lr=lr
      num_filters = base.fc.in_features
      layers = list(base.children())[:-1]
      self.feature_extractor = nn.Sequential(*layers)
      self.model = nn.Linear(num_filters, num_classes)
    def forward(self, x):
        x=self.feature_extractor(x)
        x=torch.flatten(x,start_dim=1)
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
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-5)







#####################Question 2


from torchvision import models
class VIT_iNaturalist_whole(pl.LightningModule):
    def __init__(self,num_classes=10,optimizer=torch.optim.Adam,lr=0.001,weights=models.ViT_B_16_Weights.DEFAULT):
      super().__init__()
      self.save_hyperparameters()
      self.optimizer=optimizer
      self.lr=lr
      self.model =models.vit_b_16(weights=weights)
      self.model.heads=nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))
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
        self.log("test_acc", acc, prog_bar=True,)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss":loss,"test_acc":acc}
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
    






import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models

wandb_logger = WandbLogger(project="DL_assignment2_B")
wandb.init(project="DL_assignment2_B", reinit=True)
weights = models.ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()
batch_size = 32

model=VIT_iNaturalist_whole(optimizer=torch.optim.Adam,lr=3e-4,weights=weights)
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=auto_transforms,test_transforms=auto_transforms)

wandb_logger.experiment.config.update({
    "model_class": model.__class__.__name__
})
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

trainer.fit(model, naturalist_DM)

wandb.watch(model, log="all", log_freq=100)

wandb.finish()

from torchvision import models
class VIT_iNaturalist_first_k(pl.LightningModule):
    def __init__(self,num_classes=10, k:int=5,optimizer=torch.optim.Adam,lr=0.001,weights=models.ViT_B_16_Weights.DEFAULT):
      super().__init__()

      self.save_hyperparameters()
      self.optimizer=optimizer
      self.lr=lr
      self.model =models.vit_b_16(weights=weights)
      for param in self.model.parameters():
        param.requires_grad = False
      for param in self.model.encoder.layers[k:].parameters():
        param.requires_grad = True
      self.model.heads=nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))
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
        self.log("test_acc", acc, prog_bar=True,)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss":loss,"test_acc":acc}
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
    
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models


wandb_logger = WandbLogger(project="DL_assignment2_B")
wandb.init(project="DL_assignment2_B", reinit=True)

weights = models.ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()
batch_size = 64

model=VIT_iNaturalist_first_k(optimizer=torch.optim.Adam,lr=0.001,weights=weights,k=7)
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=auto_transforms,test_transforms=auto_transforms)

wandb_logger.experiment.config.update({
    "model_class": model.__class__.__name__
})
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

trainer.fit(model, naturalist_DM)

wandb.watch(model, log="all", log_freq=100)

wandb.finish()


import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models


wandb_logger = WandbLogger(project="DL_assignment2_B")
wandb.init(project="DL_assignment2_B", reinit=True)

weights = models.ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()
batch_size = 128

model=VIT_iNaturalist_first_k(optimizer=torch.optim.Adam,lr=0.001,weights=weights,k=3)
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=auto_transforms,test_transforms=auto_transforms)

wandb_logger.experiment.config.update({
    "model_class": model.__class__.__name__
})
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

trainer.fit(model, naturalist_DM)

wandb.watch(model, log="all", log_freq=100)

wandb.finish()


from torchvision import models
class VIT_iNaturalist_dense_only(pl.LightningModule):
    def __init__(self,num_classes=10,optimizer=torch.optim.Adam,lr=0.001,weights=models.ViT_B_16_Weights.DEFAULT):
      super().__init__()

      self.save_hyperparameters()
      self.optimizer=optimizer
      self.lr=lr
      self.model =models.vit_b_16(weights=weights)
      for param in self.model.parameters():
        param.requires_grad = False
      self.model.heads=nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))
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
        self.log("test_acc", acc, prog_bar=True,)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss":loss,"test_acc":acc}
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
    

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models

wandb_logger = WandbLogger(project="DL_assignment2_B")
weights = models.ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()
batch_size = 128
wandb.init(project="DL_assignment2_B", reinit=True)
model=VIT_iNaturalist_dense_only(optimizer=torch.optim.Adam,lr=0.001,weights=weights)
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=auto_transforms,test_transforms=auto_transforms)

wandb_logger.experiment.config.update({
    "model_class": model.__class__.__name__
})
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

trainer.fit(model, naturalist_DM)

wandb.watch(model, log="all", log_freq=100)

wandb.finish()


import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torchvision.models as models

wandb_logger = WandbLogger(project="DL_assignment2_B")
weights = models.ViT_B_16_Weights.DEFAULT
auto_transforms = weights.transforms()
batch_size = 256
wandb.init(project="DL_assignment2_B", reinit=True)
model=VIT_iNaturalist_dense_only(optimizer=torch.optim.Adam,lr=0.01,weights=weights)
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=auto_transforms,test_transforms=auto_transforms)

wandb_logger.experiment.config.update({
    "model_class": model.__class__.__name__
})
trainer = pl.Trainer(max_epochs=10, logger=wandb_logger)

trainer.fit(model, naturalist_DM)

wandb.watch(model, log="all", log_freq=100)

wandb.finish()


###########################Question 3



weights=torchvision.models.ViT_B_16_Weights.DEFAULT
auto_transforms=weights.transforms()
auto_transforms

from torchvision import transforms
from torchvision.transforms import InterpolationMode
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
train_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

class iNaturalistDataModule_finetune(pl.LightningDataModule):
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
      full_dataset = ImageFolder(root=self.train_dir)
      total_size = len(full_dataset)
      train_size = int(total_size * self.train_val_split)
      val_size = total_size - train_size
      train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
      self.train_dataset = TransformedSubset(train_subset, self.train_transforms)
      self.val_dataset = TransformedSubset(val_subset, self.test_transforms)
      self.test_dataset = ImageFolder(root=self.test_dir, transform=self.test_transforms)

    def train_dataloader(self):
      return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)

    def val_dataloader(self):

      return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
      return DataLoader(self.test_dataset,batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)


from torchvision import models
class VIT_iNaturalist_dense_only(pl.LightningModule):
    def __init__(self,num_classes=10,optimizer=torch.optim.Adam,lr=0.001,weights=models.ViT_B_16_Weights.DEFAULT,dropout=0.5,dene_size=0):
      super().__init__()
      self.save_hyperparameters()
      self.optimizer=optimizer
      self.lr=lr
      self.model =models.vit_b_16(weights=weights)
      for param in self.model.parameters():
        param.requires_grad = False
      if dene_size!=0:
        self.model.heads=nn.Sequential(nn.Linear(in_features=768,out_features=dene_size),nn.SELU(), nn.Dropout(p=dropout), nn.Linear(in_features=dene_size, out_features=num_classes))
      else:
        self.model.heads=nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features=768, out_features=num_classes))
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
        self.log("test_acc", acc, prog_bar=True,)
        self.log("test_loss", loss, prog_bar=True)
        return {"test_loss":loss,"test_acc":acc}
    def configure_optimizers(self):
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=1e-4)
    

import wandb
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "validation_loss",
        "goal": "minimize"
    },
    "parameters": {
        "dropout": {
            "values":[0,0.2,0.3,0.5,0.7]
        },
        "batch_size": {
            "values":[32,64,128,256]
        },"dene_size": {
            "values":[0,512,1024,2046]
        },"lr": {
            "values":[1e-3,5e-4,1e-4]
        }
    }
}
sweep_id = wandb.sweep(sweep_config, project="DL-Addignemt2_B_finetune")



import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
early_stop_cb = EarlyStopping(
    monitor="validation_loss",
    min_delta=0.00,
    patience=10,
    verbose=True,
    mode="min"
)

def fine_tune(config=None):
    with wandb.init(config=config):
        config = wandb.config
        wandb_logger = WandbLogger(project="DL-Addignemt2_B_finetune")
        dropout=config.dropout
        batch_size=config.batch_size
        dene_size=config.dene_size
        lr=config.lr
        model=VIT_iNaturalist_dense_only(lr=lr,dropout=dropout,dene_size=dene_size)
        naturalist_DM=iNaturalistDataModule_finetune(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=train_transform,test_transforms=auto_transforms)
        trainer = pl.Trainer(logger=wandb_logger, max_epochs=100,callbacks=[early_stop_cb])
        trainer.fit(model, naturalist_DM)


wandb.agent(sweep_id, function=fine_tune)



from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
def fine_tune(dropout=0.0,batch_size=32,dene_size=32,lr=1e-3):

    early_stop_cb = EarlyStopping(
        monitor="validation_loss",
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode="min"
    )
    checkpoint_cb = ModelCheckpoint(
    monitor="validation_loss",
    mode="min",
    save_top_k=1,
    verbose=True,
    dirpath="checkpoints/",
    filename="best-model"
    )

    wandb_logger = WandbLogger(project="DL-Addignemt2_B_finetune")
 
    model=VIT_iNaturalist_dense_only(lr=lr,dropout=dropout,dene_size=dene_size)
    naturalist_DM=iNaturalistDataModule_finetune(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=train_transform,test_transforms=auto_transforms)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=100,callbacks=[early_stop_cb,checkpoint_cb])
    trainer.fit(model, naturalist_DM)
    best_model = iNaturalistModel.load_from_checkpoint(checkpoint_cb.best_model_path)
    trainer.test(best_model, datamodule=naturalist_DM)
    wandb.finish()

fine_tune(dropout=0.7,batch_size=32,dene_size=512,lr=0.0001)
