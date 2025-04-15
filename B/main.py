import argparse

from src.wandb_fns import fine_tune_manual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a deep learning model manually.")
    parser.add_argument("--project", type=str, default="DL-Addignemt2_B_finetune",help="Name of the project (default: DL-Addignemt2_B_finetune)")
    parser.add_argument("--dropout", type=float, default=0,help="Dropout rate (default: 0)")
    parser.add_argument("--batch_size", type=int, default=64,help="Batch size for training (default: 64)")
    parser.add_argument("--dense_size", type=int, default=0,help="Size of the dense layer (default: 0)")
    parser.add_argument("--lr", type=float, default=1e-3,help="Learning rate (default: 0.001)")
    args = parser.parse_args()
    fine_tune_manual(
        project=args.project,
        dropout=args.dropout,
        batch_size=args.batch_size,
        dense_size=args.dense_size,
        lr=args.lr
    )