
# DL assignment Part A
##  iNaturalist ConvNet Classifier from Scratch  
[Wandb report](https://wandb.ai/cs24m041-iit-madras/DA6401-Assignment2/reports/DA6401-Assignment-2--VmlldzoxMjAzNTUzNA?accessToken=pvuaifa3kvtlvgihqg4d87n1l6ddmj2w3dri9xvoix5rpxbhui31wd1pkdskjf64) associated can be found here
## Instructions
clone the repository
```
git clone https://github.com/sankarvinayak/DL-assignment2.git
cd DL-assignment2
```
install requirements
```
pip install requirements.txt
```
navigate to the directory
```
cd A
```



## Project Structure

```
.
├── main.py            
├── readme.md          
└── src
    ├── data.py        
    ├── model.py      
    └── wandb_fn.py   
```

---

##  Dataset

- Downloads the **iNaturalist_12K** dataset automatically if not already in `src/inaturalist_12K/`


---

##  How to Run Training

Run the model with any configuration of your choice:

```bash
python main.py --project project_name --dropout 0.3  --batch_size 128  --lr 0.0001  --dense_size 512  --epoch 30  --activation_fun ReLU  --num_filters 64  --filter_size 3  --filter_org double  --batch_norm   --no-augment
```

---

## Arguments

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


## Weights & Biases Integration

Automatically logs:
- Training and validation accuracy/loss
- Hyperparameters
- Epoch-wise metrics

---
