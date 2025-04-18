from .model import VIT_iNaturalist_dense_only
from .data import iNaturalistDataModule_finetune
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
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
from torchvision.transforms import InterpolationMode
import wandb

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
def fine_tune(config=None):
    with wandb.init(config=config):
        torch.manual_seed(3407)
        torch.cuda.manual_seed(3407)
        early_stop_cb = EarlyStopping(monitor="validation_loss",min_delta=0.00,patience=10,verbose=True,mode="min")
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
def fine_tune_manual(project="DL-Addignemt2_B_finetune",dropout=0,batch_size=64,dense_size=0,lr=1e-3):
    early_stop_cb = EarlyStopping(monitor="validation_loss",min_delta=0.00,patience=10,verbose=True,mode="min")
    checkpoint_cb = ModelCheckpoint(monitor="validation_loss",mode="min",save_top_k=1,verbose=True,dirpath="checkpoints/",filename="best-model")
    wandb_logger = WandbLogger(project=project)
    model=VIT_iNaturalist_dense_only(lr=lr,dropout=dropout,dene_size=dense_size)
    naturalist_DM=iNaturalistDataModule_finetune(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=train_transform,test_transforms=auto_transforms)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=100,callbacks=[early_stop_cb,checkpoint_cb])
    trainer.fit(model, naturalist_DM)
    best_model = VIT_iNaturalist_dense_only.load_from_checkpoint(checkpoint_cb.best_model_path)
    trainer.test(best_model, datamodule=naturalist_DM)
    wandb.finish()