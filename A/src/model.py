import torch
import torch.nn as nn
import pytorch_lightning as pl
class iNaturalistModel(pl.LightningModule):
    """Pytorch lightming module which inherits nn.Module which is used for network in pytrch
        construct the network of specification and store as part of the class object
    """
    def __init__(self, in_channels=3,input_size=(224, 224),dense_size=512,  num_filters_layer: list = [16, 32, 64, 128, 256],filter_size: list = [3, 3, 3, 3, 3], stride=1,  padding='same', num_classes=10,activation=nn.ReLU,dropout_rate=0,optimizer=torch.optim.Adam,lr=0.001,batch_norm=False ):
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
        """Will be called while training model.fit"""
        images, labels = batch
        logits = self.forward(images)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    def validation_step(self, batch, batch_idx):
        """Will be clalled while validation model.fit"""
        images, labels = batch
        logits = self.forward(images)
        loss = nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("validation_acc", acc, prog_bar=True)
        self.log("validation_loss", loss, prog_bar=True)
        return {"validation_loss":loss,"validation_acc":acc}
    def test_step(self, batch, batch_idx):
        """Will be called while testing model.test"""
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
