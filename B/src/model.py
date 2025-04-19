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

from torch.optim.lr_scheduler import ReduceLROnPlateau


from torchvision import models
class VIT_iNaturalist_dense_only(pl.LightningModule):
    
    """Use the Vision transformer as the base model and modify the fully connected layer to suit the needs,
    if the dense size is mentioned as 0 the output of the transformer block will the directly connected to the final output neuron else there will be a fully connected set of neurons in between
    """
    def __init__(self,num_classes=10,optimizer=torch.optim.Adam,lr=0.001,weights=models.ViT_B_16_Weights.DEFAULT,dropout=0.5,dene_size=0,weight_decay=0):
      super().__init__()
      self.save_hyperparameters()
      self.optimizer=optimizer
      self.lr=lr
      self.model =models.vit_b_16(weights=weights)
      self.weight_decay =weight_decay
    
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
        return self.optimizer(self.model.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5) #do seems to work good but not used for majority of the runs so commented out
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "validation_loss", 
        #         "interval": "epoch",
        #         "frequency": 1
        #     }
        # }
