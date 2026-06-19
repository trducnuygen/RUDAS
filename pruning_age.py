import argparse
import os
import torch
from src.data import get_dataloader
from src.feat_pruner import AgePruner
import numpy as np
import random
import numpy as np
import pandas as pd

def get_path_from_concat(full_set, idx):
    for dataset in full_set.datasets:
        if idx < len(dataset):
            return dataset.samples[idx][0]
        idx -= len(dataset)
    raise IndexError("Index out of range")


parser = argparse.ArgumentParser(description="ImageNet pruning")

parser.add_argument("-d", "--data_path", type=str, required=True, help='Path to the dataset')
# parser.add_argument("--batch_size", type=int, default=32, help='Batch size for feature extraction')
parser.add_argument("-re", "--rate_easy", type=float, default=0.1, help='compression rate for easy samples')
parser.add_argument("-rm", "--rate_moderate", type=float, default=0.1, help='compression rate for moderate samples')
parser.add_argument("-rh", "--rate_hard", type=float, default=0.1, help='compression rate for hard samples')
parser.add_argument("-q", "--quotient", type=float, default=0.1, help='quotient for defining hard/easy samples')
parser.add_argument("--n_bins", type=int, default=10, help='Number of bins for middle-age samples')
parser.add_argument("--age", type=str, default="age_table.csv", help='Path to the age table csv file')
# parser.add_argument("--device", type=str, default="cuda", help='Device to use for computation')
parser.add_argument("--seed", type=int, default=42, help='Random seed for reproducibility')
args = parser.parse_args()


def main(args):
    # device = torch.device(args.device)

    random.seed(args.seed)  
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output_folder = 'reduced_result_age'
    output_dir = os.path.join(output_folder, f"re-{args.rate_easy}_rm-{args.rate_moderate}_rh-{args.rate_hard}_q-{args.quotient}_100")
    output_file = os.path.join(output_dir, "selected_paths.txt")
    os.makedirs(output_dir, exist_ok=True)

    # if os.path.exists(output_file):
    #     print(f"[INFO] Found existing {output_file}, skipping pruning.")
    #     return
    args.batch_size = 32 # not used in this script, but needed for get_dataloader

    train_dir = f"{args.data_path}/train"
    _, train_dataset = get_dataloader(train_dir, args.batch_size, return_dataset=True)

    test_dir = f"{args.data_path}/val"
    _, val_dataset = get_dataloader(test_dir, args.batch_size, return_dataset=True)

    # concatenate train and test datasets
    full_set = torch.utils.data.ConcatDataset([train_dataset, val_dataset])

    # we have age table
    age = pd.read_csv(args.age)
    age_train = age[age['split'] == 'train'].drop(columns=['split'])['age'].values
    age_val = age[age['split'] == 'val'].drop(columns=['split'])['age'].values
    
    labels_train = []
    for i in range(len(train_dataset)):
        labels_train.append(train_dataset.samples[i][1])
    labels_val = []
    for i in range(len(val_dataset)):
        labels_val.append(val_dataset.samples[i][1])
    
    # pruning must be done separately for train and val
    pruner_train = AgePruner(age_train, labels=np.array(labels_train))
    pruner_val = AgePruner(age_val, labels=np.array(labels_val))
    print(f"Pruning train set with {len(train_dataset)} samples...")
    selected_indices_train = pruner_train.prune(re=args.rate_easy, rm=args.rate_moderate, rh=args.rate_hard, n_bins=args.n_bins, q=args.quotient)
    
    print(f"Pruning val set with {len(val_dataset)} samples...")
    selected_indices_val = pruner_val.prune(re=args.rate_easy, rm=args.rate_moderate, rh=args.rate_hard, n_bins=args.n_bins, q=args.quotient)
    
    print(f"Selected {len(selected_indices_train)} samples from train set and {len(selected_indices_val)} samples from val set.")
    selected_indices = selected_indices_train + [idx + len(train_dataset) for idx in selected_indices_val]
    selected_paths = [get_path_from_concat(full_set, i) for i in selected_indices]

    # save selected paths to a text file
    with open(output_file, "w") as f:
        for path in selected_paths:
            f.write(path + "\n")
        # f.write(f'Total samples: {len(selected_paths)}')
    print(f"Saved {len(selected_paths)} selected paths to {output_file}")
    

if __name__ == "__main__":
    main(args)

