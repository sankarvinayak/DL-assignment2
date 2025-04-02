from src.cli.options import parse_args
from src.wandb.wandb_functions import wandb_run_experiment
def main():
    args = parse_args()
    wandb_run_experiment(args)
if __name__ == "__main__":
    main()
    
