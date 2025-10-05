#!/usr/bin/env python3
"""
Plot per-epoch average train & val losses from an SVD-Pi3 style log.

Usage:
  python plot_epoch_avgs.py --log /path/to/50_epochs_train.log --out epoch_avgs.png
"""

import re
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_epoch_avgs(log_path):
    # regex tailored to your lines
    pat_train = re.compile(
        r"Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*(\d+)\s*\].*?loss:\s*([0-9]*\.[0-9]+)\s*\(([0-9]*\.[0-9]+)\)"
    )
    pat_val = re.compile(
        r"Validation Epoch:\s*\[(\d+)\]\s*\[\s*(\d+)\s*/\s*(\d+)\s*\].*?loss:\s*([0-9]*\.[0-9]+)\s*\(([0-9]*\.[0-9]+)\)"
    )

    train_losses_by_epoch = defaultdict(list)
    val_losses_by_epoch   = defaultdict(list)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "Failed to load data" in line:
                continue

            m_tr = pat_train.search(line) if "Validation Epoch" not in line else None
            if m_tr:
                e = int(m_tr.group(1))
                inst_loss = float(m_tr.group(4))  # instantaneous loss (before parentheses)
                train_losses_by_epoch[e].append(inst_loss)
                continue

            m_val = pat_val.search(line)
            if m_val:
                e = int(m_val.group(1))
                inst_loss = float(m_val.group(4))
                val_losses_by_epoch[e].append(inst_loss)
                continue

    # compute per-epoch means
    epochs_train = sorted(train_losses_by_epoch.keys())
    epochs_val   = sorted(val_losses_by_epoch.keys())
    epochs = sorted(set(epochs_train) | set(epochs_val))

    train_means = [float(np.mean(train_losses_by_epoch[e])) if e in train_losses_by_epoch else np.nan for e in epochs]
    val_means   = [float(np.mean(val_losses_by_epoch[e]))   if e in val_losses_by_epoch   else np.nan for e in epochs]

    df = pd.DataFrame({
        "epoch": epochs,
        "train_loss_mean": train_means,
        "val_loss_mean": val_means,
    })
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to training log")
    ap.add_argument("--out", default="epoch_avgs.png", help="Output PNG")
    args = ap.parse_args()

    df = parse_epoch_avgs(args.log)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss_mean"], marker="o", linewidth=1.5, label="train (mean per epoch)")
    plt.plot(df["epoch"], df["val_loss_mean"],   marker="o", linewidth=1.5, label="val (mean per epoch)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Per-epoch Average Loss (Train vs Val)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved figure: {args.out}")

    csv_path = args.out.replace(".png", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

if __name__ == "__main__":
    main()