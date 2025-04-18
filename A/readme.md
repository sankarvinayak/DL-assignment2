
# DL assignment Part A
## 🌿 iNaturalist ConvNet Classifier from Scratch  

## 🧾 Project Structure

```
.
├── main.py            # Entry point to train the model from scratch
├── readme.md          # Project documentation
└── src
    ├── data.py        # DataModule for iNaturalist dataset
    ├── model.py       # PyTorch Lightning model with CNN architecture
    └── wandb_fn.py    # wandb_train() function with training logic
```

---

## 📦 Dataset

- Downloads the **iNaturalist_12K** dataset automatically if not already in `src/inaturalist_12K/`


---

## 🚀 How to Run Training

Run the model with any configuration of your choice:

```bash
python main.py --project ConvNet-Nature12K  --dropout 0.3  --batch_size 128  --lr 0.0001  --dense_size 512  --epoch 30  --activation_fun ReLU  --num_filters 64  --filter_size 3  --filter_org double  --batch_norm   --no-augment
```

---

## ⚙️ Available Arguments

| Argument            | Description                                                    | Default         |
|---------------------|----------------------------------------------------------------|-----------------|
| `--project`         | WandB project name                                             | `"default_project"` |
| `--batch_norm`      | Enable batch normalization                                     | `False`         |
| `--augment` / `--no-augment` | Enable/disable data augmentation                     | `True`          |
| `--activation_fun`  | Activation function to use (`ReLU`, `SiLU`, `LeakyReLU`, etc.) | `"SiLU"`        |
| `--dense_size`      | Size of the fully connected dense layer                        | `1024`          |
| `--dropout`         | Dropout rate before the final layer                            | `0.5`           |
| `--epoch`           | Number of training epochs                                      | `50`            |
| `--lr`              | Learning rate                                                  | `0.0001`        |
| `--num_filters`     | Number of base filters in conv layers                          | `32`            |
| `--filter_size`     | Size of convolution kernel                                     | `3`             |
| `--filter_org`      | Filter organization strategy (`double`, `same`, etc.)          | `"double"`      |
| `--batch_size`      | Batch size                                                     | `64`            |

---

## 🧠 Model Highlights

- Fully configurable ConvNet
- Custom activation and dropout support
- BatchNorm toggle
- Trained using **PyTorch Lightning**
- Flexible conv filter organization (`--filter_org`)
- Logs to [Weights & Biases](https://wandb.ai)

---

## 🧪 Weights & Biases Integration

Automatically logs:
- Training and validation accuracy/loss
- Hyperparameters
- Epoch-wise metrics

---
