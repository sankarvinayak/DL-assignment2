train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
      transforms.RandomRotation(degrees=15),
      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
      transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
test_transforms = transforms.Compose([
    transforms.Resize(256),            # shorter side → 256px
    transforms.CenterCrop(224),        # then 224×224 center patch
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
activation_fun="SiLU"
activation=torch.nn.SiLU
batch_norm="Yes"
batch_size=64
data_augmentation="Yes"
dense_size=1024
dropout=0.5
epoch=50
filter_org="double"
filter_size=3
filter_sizes=[filter_size]*5
lr=0.0001
num_filters=32


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


num_filters_layer=[num_filters,num_filters*2,num_filters*4,num_filters*8,num_filters*16]
# num_filters_layer=[num_filters]*5

run_name = f"Augment{data_augmentation}activation_fun{activation_fun}_dropout_{dropout}"
torch.manual_seed(3407)
wandb_logger = WandbLogger(project="DL-Addignemt2_A",name=run_name)

model = iNaturalistModel(num_filters_layer=num_filters_layer,activation=activation,dropout_rate=dropout,batch_norm=True,filter_size=filter_sizes,dense_size=dense_size,lr=lr)
naturalist_DM=iNaturalistDataModule(train_dir='inaturalist_12K/train',test_dir='inaturalist_12K/val',batch_size=batch_size,train_transforms=train_transforms,test_transforms=test_transforms)

trainer = pl.Trainer(logger=wandb_logger, max_epochs=epoch,callbacks=[early_stop_cb,checkpoint_cb])
trainer.fit(model, naturalist_DM)