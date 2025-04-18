import argparse

import os
from src.wandb_fn import wandb_train


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a model with wandb logging.")
    parser.add_argument("--project", type=str, default="default_project", help="WandB project name")

    parser.add_argument("--batch_norm", action="store_true",help="Enable batch_norm")
    parser.set_defaults(augment=True)
    parser.add_argument("--augment", action="store_true",help="Enable data augmentation")
    parser.add_argument("--no-augment", dest="augment", action="store_false",help="Disable data augmentation")
    parser.set_defaults(augment=True)

    parser.add_argument("--activation_fun", type=str, default="SiLU",help="Activation function to use (default: SiLU)")
    parser.add_argument("--dense_size", type=int, default=1024,help="Size of the dense layer (default: 1024)")
    parser.add_argument("--dropout", type=float, default=0.5,help="Dropout rate (default: 0.5)")
    parser.add_argument("--epoch", type=int, default=50,help="Number of training epochs (default: 50)")
    parser.add_argument("--lr", type=float, default=0.0001,help="Learning rate (default: 0.0001)")
    parser.add_argument("--num_filters", type=int, default=32,help="Number of filters in conv layers (default: 32)")
    parser.add_argument("--filter_size", type=int, default=3,help="Size of the convolutional filter (default: 3)")
    parser.add_argument("--filter_org", type=str, default="double",help="Filter organization strategy (default: double)")
    parser.add_argument("--batch_size", type=int, default=64,help="Batch size for training (default: 64)")
    args = parser.parse_args()

    
    dataset_dir = "src/inaturalist_12K"
    dataset_zip = "src/nature_12K.zip"

    if not os.path.exists(dataset_dir):
        os.makedirs("src", exist_ok=True)
        print("Downloading and extracting nature_12K dataset into src/...")
        os.system(f"wget -O {dataset_zip} https://storage.googleapis.com/wandb_datasets/nature_12K.zip")
        os.system(f"unzip -q {dataset_zip} -d src/")
    else:
        print("Dataset already exists in src/. Skipping download.")

    wandb_train(
        project=args.project,
        augment=args.augment,
        activation_fun=args.activation_fun,
        dense_size=args.dense_size,
        dropout=args.dropout,
        epoch=args.epoch,
        lr=args.lr,
        num_filters=args.num_filters,
        filter_size=args.filter_size,
        filter_org=args.filter_org,
        batch_size=args.batch_size,
        batch_norm=args.batch_norm
    )