
from .data import iNaturalistDataModule, iNaturalistDataModule_with_cls_name
from .model import iNaturalistModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from torchvision import transforms
import random
import torch
from PIL import Image
import wandb

import torch
import random
import matplotlib.pyplot as plt
from PIL import Image

def show_random_predictions(model, dataset, class_names, device='cuda', num_samples=30, rows=3, cols=10):
    """Function which shows the model prediciton """
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

def log_random_predictions_separate( model, dataset,class_names,device='cuda', num_samples=30,key="random_preds"):
    """Function which log grid of 30 images to wandb"""
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




def wandb_train(project="DL-Addignemt2_A",augment=True,activation_fun="SiLU",dense_size=1024,dropout=0.5,epoch=50,lr=0.0001,num_filters=32,filter_size=3,filter_org="double",batch_size=64,batch_norm=True):
   
    if augment: #did try with single augmentations techniques but from experiments it is seen that combining different augmentations techniques helps in improving performance hence a set of augmentations which are commonly used for different model for imagenet dataset(like ViT,VGG,EffNet etc) is used here in wandb initial few runs does have single augmentation technique logged
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),#make all the images into same shape 224x224
            transforms.RandomHorizontalFlip(p=0.5), #with probab
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.3),
            transforms.RandomRotation(degrees=15),  #randomly rotate the image by a small degree 
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]) #imagenet mean and standard deviation used in many placess
        test_transforms = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], #similar to default transforms of ViT which is used in the part B and seems to give better perfomance than using .5 directly 
                            std=[0.229, 0.224, 0.225]), # for the train set Computed Mean: [0.47089460492134094, 0.45923930406570435, 0.3884953558444977]  Std: [0.19317267835140228, 0.18763333559036255, 0.1841067522764206] which is close enough hence using these
    ])
    else:
      train_transforms = transforms.Compose([
      transforms.Resize((224,224)), #convert to fixed size 224x224
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])
      test_transforms = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])  #use same set of train and test transforms if data augmentation is not enabled
    if activation_fun=="ReLU":
      activation=torch.nn.ReLU
    elif activation_fun=="GELU":
      activation=torch.nn.GELU
    elif activation_fun=="SiLU":
      activation=torch.nn.SiLU
    elif activation_fun=="Mish":
      activation=torch.nn.Mish
   
    wandb.login()
    if filter_org=="double":
      num_filters_layer=[num_filters,num_filters*2,num_filters*4,num_filters*8,num_filters*16]
    elif filter_org=="half":
      num_filters_layer=[num_filters,num_filters//2,num_filters//4,num_filters//8,num_filters//16]
    else:
       num_filters_layer=[num_filters]*5
    early_stop_cb = EarlyStopping(monitor="validation_loss",min_delta=0.00,patience=10,verbose=True,mode="min") #early stopping callback which will terminate the training if the validation does not decrease for 10 consecutive steps
    checkpoint_cb = ModelCheckpoint(monitor="validation_loss", mode="min",save_top_k=1,verbose=True,dirpath="checkpoints/",filename="best-model") #save the model when the loss it at lowest and finally return the result of this best model
    # num_filters_layer=[num_filters,num_filters*2,num_filters*4,num_filters*8,num_filters*16]
    # num_filters_layer=[num_filters]*5
    # if len(filter_size)==1:
    filter_sizes=[filter_size]*5
    # else:
    #    filter_sizes=filter_size
    # filter_sizes=filter_size
    run_name = f"Augment{augment}activation_fun{activation_fun}_dropout_{dropout}"
    torch.manual_seed(3407)
    torch.cuda.manual_seed(3407)
    wandb_logger = WandbLogger(project=project,name=run_name)

    model = iNaturalistModel(num_filters_layer=num_filters_layer,activation=activation,dropout_rate=dropout,batch_norm=batch_norm,filter_size=filter_sizes,dense_size=dense_size,lr=lr)
    naturalist_DM=iNaturalistDataModule(train_dir='src/inaturalist_12K/train',test_dir='src/inaturalist_12K/val',batch_size=batch_size,train_transforms=train_transforms,test_transforms=test_transforms)

    trainer = pl.Trainer(logger=wandb_logger, max_epochs=epoch,callbacks=[early_stop_cb,checkpoint_cb])
    trainer.fit(model, naturalist_DM)
    best_model = iNaturalistModel.load_from_checkpoint(checkpoint_cb.best_model_path)
    trainer.test(best_model, datamodule=naturalist_DM)
    # trainer.test(best_model, datamodule=naturalist_DM)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    naturalist_DM_new = iNaturalistDataModule_with_cls_name(
    train_dir='src/inaturalist_12K/train',
    test_dir='src/inaturalist_12K/val',
    batch_size=batch_size,
    train_transforms=train_transforms,
    test_transforms=test_transforms
    )
    naturalist_DM_new.setup()
    class_names = naturalist_DM_new.test_dataset.classes
    log_random_predictions_separate(best_model,naturalist_DM_new.test_dataset,class_names,device=device,num_samples=30,key="Model prediction")
    wandb.finish()
