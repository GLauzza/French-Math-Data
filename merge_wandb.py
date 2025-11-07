import argparse

import wandb
import pandas as pd

ENTITY = "G-lauzzanaa"
PROJECT = "french-cot"
API = wandb.Api()

def main(args):
    runs = []
    for run_id in args.run_ids:
        runs.append(API.run(f"{ENTITY}/{PROJECT}/{run_id}").history())

    df = pd.concat(runs).reset_index(drop=True)

    wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=args.output_id,
    )

    for index, row in df.iterrows():
        metrics_dict = row.to_dict()

        step = int(metrics_dict['train/global_step'])

        wandb.log(metrics_dict, step=step)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train model with TRL')
    parser.add_argument('--run_ids', type=str, nargs='+', required=True, help='run ids to merge')
    parser.add_argument('--output_id', type=str, help='id of the merged run', required=True)
    args = parser.parse_args()

    main(args)