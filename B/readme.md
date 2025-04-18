

# DL assignment 2 B

## 🌿 iNaturalist Image Classification with ViT  

---

## 📚 Overview

This project fine-tunes a **Vision Transformer (ViT)** model on the [iNaturalist 12K dataset](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) for multi-class image classification.

It uses:

- **PyTorch Lightning** for clean and scalable training
- **WandB (Weights & Biases)** for experiment tracking

---

## 🧾 Project Structure

```
.
├── main.py                  # Entry point to download data and run training
├── readme.md                # Project description and usage
└── src/
    ├── data.py              # DataModule definition (train/val/test loaders)
    ├── model.py             # ViT-based classifier wrapped in LightningModule
    └── wandb_fns.py         # wandb logging utilities (e.g. predictions)
```

---

## 🖼 Dataset

- 📦 Downloaded automatically from:  
  `https://storage.googleapis.com/wandb_datasets/nature_12K.zip`

- 📁 Extracted to:  
  `src/inaturalist_12K/`

---

## 🚀 How to Run



###  Train the model

```bash
python main.py --project ViT-iNaturalist --dropout 0.2 --batch_size 512  --dense_size 512 --lr 0.0001  --epochs 10
```

---

## CLI Arguments

| Argument        | Description                                | Default              |
|------------------|--------------------------------------------|----------------------|
| `--project`      | WandB project name                         | `"Default_project"`  |
| `--dropout`      | Dropout rate for dense layer              | `0`                  |
| `--batch_size`   | Training batch size                       | `64`                 |
| `--dense_size`   | Hidden units in dense layer (if used)     | `0`                  |
| `--lr`           | Learning rate                             | `0.001`              |
| `--epochs`       | Number of training epochs                 | `10`                 |

---

## 🧠 Pretrained model

- Make use of `vit_b_16` from torchvision models
- Optional dense layer with dropout
- Final classifier layer matching number of classes in iNaturalist dataset

---

## 📊 WandB Logging

The training script logs the following to Weights & Biases:

- Training & validation loss
- Accuracy metrics
- Random prediction samples with ground truth
- Model graph and hyperparameters

---

