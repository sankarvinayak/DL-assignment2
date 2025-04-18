from .model import VIT_iNaturalist_dense_only
from .data import iNaturalistDataModule_finetune

import random
import torch,torchvision
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import wandb

def log_random_predictions_separate(
    model,
    dataset,
    class_names,
    device,
    num_samples=30,
    key="random_preds"
):

    
    model.eval().to(device)
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

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
def fine_tune(config=None):
    with wandb.init(config=config):
        torch.manual_seed(3407)
        torch.cuda.manual_seed(3407)
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
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT
        auto_transforms=weights.transforms()
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
def fine_tune_manual(project="DL-Addignemt2_B_finetune",dropout=0,batch_size=64,dense_size=0,lr=1e-3,epochs=10):
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
    weights=torchvision.models.ViT_B_16_Weights.DEFAULT
    auto_transforms=weights.transforms()
    wandb.login()
    early_stop_cb = EarlyStopping(monitor="validation_loss",min_delta=0.00,patience=10,verbose=True,mode="min")
    checkpoint_cb = ModelCheckpoint(monitor="validation_loss",mode="min",save_top_k=1,verbose=True,dirpath="checkpoints/",filename="best-model")
    wandb_logger = WandbLogger(project=project)
    model=VIT_iNaturalist_dense_only(lr=lr,dropout=dropout,dene_size=dense_size)
    naturalist_DM=iNaturalistDataModule_finetune(train_dir='src/inaturalist_12K/train',test_dir='src/inaturalist_12K/val',batch_size=batch_size,train_transforms=train_transform,test_transforms=auto_transforms)
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=epochs,callbacks=[early_stop_cb,checkpoint_cb])
    trainer.fit(model, naturalist_DM)
    best_model = VIT_iNaturalist_dense_only.load_from_checkpoint(checkpoint_cb.best_model_path)
    trainer.test(best_model, datamodule=naturalist_DM)
    class_names = naturalist_DM.test_dataset.classes
    device='cuda' if torch.cuda.is_available() else 'cpu'
    log_random_predictions_separate(best_model,naturalist_DM.test_dataset,class_names,device=device,num_samples=30,key="Model prediction")
    wandb.finish()