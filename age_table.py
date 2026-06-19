import os
import argparse
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.data import get_dataloader

def build_age_table(data_root: str, scores_path: str, output_path: str) -> None:

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    _, train_dataset = get_dataloader(train_dir, batch_size=1, shuffle=False, return_dataset=True)
    _, val_dataset = get_dataloader(val_dir, batch_size=1, shuffle=False, return_dataset=True)
    train_samples = train_dataset.samples # list[(path, cls)]
    val_samples = val_dataset.samples

    n_train = len(train_samples)
    n_val = len(val_samples)
    N = n_train + n_val

    print(f"Train samples: {n_train}")
    print(f"Val samples: {n_val}")
    print(f"Total: {N}")

    scores = np.load(scores_path)
    print(f"Scores shape: {scores.shape}")

    if scores.shape[0] != N:
        raise ValueError(f"Number of scores ({scores.shape[0]}) does not match number of samples ({N}).")

    paths = [p for p, _ in train_samples] + [p for p, _ in val_samples]
    splits = ["train"] * n_train + ["val"] * n_val

    df = pd.DataFrame({
        "path" : paths,
        "split": splits,
        "age"  : scores,
    })

    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows → {output_path}")
    print(df.head(10).to_string(index=False))

    print("\nstatistics by split:")
    for split in ("train", "val"):
        sub = df[df["split"] == split]["age"]
        print(f"  {split:5s}  mean={sub.mean():.3f}  "
              f"std={sub.std():.3f}  "
              f"min={sub.min()}  max={sub.max()}")
    print(df.describe())
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a CSV mapping each sample to its AGE score."
    )
    parser.add_argument(
        "-d", "--data", required=True, default='dataset',
        help="Root dataset directory containing train/ and val/ subfolders."
    )
    parser.add_argument(
        "-s", "--scores", required=True,
        help="Path to the age_scores_NNN.npy file."
    )
    parser.add_argument(
        "-o", "--output", default="age_table.csv",
        help="Output CSV path (default: age_table.csv)."
    )
    args = parser.parse_args()
    build_age_table(args.data, args.scores, args.output)