import argparse

from src.wandb_fns import fine_tune_manual

import os

if __name__ == "__main__":
    dataset_dir = "src/inaturalist_12K"
    dataset_zip = "src/nature_12K.zip"

    if not os.path.exists(dataset_dir):
        os.makedirs("src", exist_ok=True)
        print("Downloading and extracting nature_12K dataset into src/...")
        os.system(f"wget -O {dataset_zip} https://storage.googleapis.com/wandb_datasets/nature_12K.zip")
        os.system(f"unzip -q {dataset_zip} -d src/")
    else:
        print("Dataset already exists in src/. Skipping download.")

    parser = argparse.ArgumentParser(description="Fine-tune a deep learning model manually.")
    parser.add_argument("--project", type=str, default="Default_project",help="Name of the project (default: DL-Addignemt2_B_finetune)")
    parser.add_argument("--dropout", type=float, default=0,help="Dropout rate (default: 0)")
    parser.add_argument("--batch_size", type=int, default=64,help="Batch size for training (default: 64)")
    parser.add_argument("--dense_size", type=int, default=0,help="Size of the dense layer (default: 0)")
    parser.add_argument("--lr", type=float, default=1e-3,help="Learning rate (default: 0.001)")

    parser.add_argument("--epochs", type=int, default=10,help="Learning rate (default: 0.001)")
    args = parser.parse_args()
    fine_tune_manual(
        project=args.project,
        dropout=args.dropout,
        batch_size=args.batch_size,
        dense_size=args.dense_size,
        lr=args.lr,
        epochs=args.epochs
    )