#!/usr/bin/env python3
"""
Train and evaluate single‐embedding classifiers over multiple seeds.

Example:
  python run_ablation_train_eval.py \
    --seeds 12 21 36 42 80 \
    --train_template /data/.../Llama-3.2-1B_{} /dataset_tensor/ \
    --val_test_dir   /data/.../Llama-3.2-1B_12/dataset_tensor/ \
    --results_base   /data/.../results/summary_text/ \
    --embed_size     2048 \
    --common_dim     2048 \
    --class_num      10 \
    --batch_size     8 \
    --epochs         15 \
    --lr             2e-4 \
    --device         cuda:0
"""


#!/usr/bin/env python3

import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# ──────────────────────────────────────────────────────────────────────────────
# Inlined MySingleDataset
# ──────────────────────────────────────────────────────────────────────────────
class MySingleDataset(torch.utils.data.Dataset):
    def __init__(self, mode, b_path, layer_to_select=-1):
        self.sents_reps = torch.load(b_path + f'{mode}_sents.pt')
        self.labels = torch.load(b_path + f'{mode}_labels.pt')
        self.sample_num = self.labels.shape[0]
        self.layer_to_select = layer_to_select

    def __getitem__(self, index):
        if self.sents_reps.dim() == 3:
            layer_embedding = self.sents_reps[index, self.layer_to_select, :]
        else:
            layer_embedding = self.sents_reps[index, :]
        return layer_embedding, self.labels[index]

    def __len__(self):
        return self.sample_num


# ──────────────────────────────────────────────────────────────────────────────
# Inlined DownstreamModelSingle
# ──────────────────────────────────────────────────────────────────────────────
class DownstreamModelSingle(nn.Module):
    def __init__(self, embed_size: int, class_num: int, common_dim: int = None):
        super().__init__()
        if common_dim and (common_dim != embed_size):
            self.compress = nn.Sequential(
                nn.Linear(embed_size, common_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            final_input_dim = common_dim
        else:
            self.compress = nn.Identity()
            final_input_dim = embed_size

        self.fc1 = nn.Linear(final_input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(256, class_num)

    def forward(self, x):
        out = self.compress(x)
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        return self.fc3(out)


# ──────────────────────────────────────────────────────────────────────────────
# Inlined Train & Test from model_ops_single
# ──────────────────────────────────────────────────────────────────────────────
from tqdm import tqdm

def Train(dataloader, device, model, loss_fn, optimizer, class_num, wandb_logger=None):
    model.train()
    all_preds, all_labels = [], []
    total_loss, num_batches = 0.0, 0

    for batch_emb, batch_labels in tqdm(dataloader, desc="Training"):
        batch_emb = batch_emb.to(device).float()
        batch_labels = batch_labels.to(device)

        logits = model(batch_emb)
        loss = loss_fn(logits, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pred_y = torch.argmax(logits, dim=1)
        all_preds.extend(pred_y.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / num_batches
    acc = accuracy_score(all_labels, all_preds)
    metrics = {
        'train_loss': avg_loss,
        'train_accuracy': acc,
        'train_macro_f1': f1_score(all_labels, all_preds, average='macro'),
        'train_micro_f1': f1_score(all_labels, all_preds, average='micro'),
        'train_weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        'train_weighted_precision': precision_score(all_labels, all_preds, average='weighted'),
        'train_weighted_recall': recall_score(all_labels, all_preds, average='weighted'),
        'train_weighted_accuracy': acc
    }
    if wandb_logger:
        wandb_logger.log(metrics)
    return metrics

def Test(dataloader, device, model, loss_fn, class_num, wandb_logger=None, mode="val"):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, num_batches = 0.0, 0

    with torch.no_grad():
        for batch_emb, batch_labels in tqdm(dataloader, desc=f"Evaluating({mode})"):
            batch_emb = batch_emb.to(device).float()
            batch_labels = batch_labels.to(device)
            logits = model(batch_emb)
            loss = loss_fn(logits, batch_labels)
            total_loss += loss.item()
            num_batches += 1
            pred_y = torch.argmax(logits, dim=1)
            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / num_batches
    acc = accuracy_score(all_labels, all_preds)
    metrics = {
        f'{mode}_loss': avg_loss,
        f'{mode}_accuracy': acc,
        f'{mode}_macro_f1': f1_score(all_labels, all_preds, average='macro'),
        f'{mode}_micro_f1': f1_score(all_labels, all_preds, average='micro'),
        f'{mode}_weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
        f'{mode}_weighted_precision': precision_score(all_labels, all_preds, average='weighted'),
        f'{mode}_weighted_recall': recall_score(all_labels, all_preds, average='weighted'),
        f'{mode}_weighted_accuracy': acc
    }
    if wandb_logger:
        wandb_logger.log(metrics)
    return metrics


######################
# MySingleDataset.py #
######################
import torch
from torch.utils.data import Dataset

    
class MySingleDataset(Dataset):
    """
    Custom Dataset for loading a single pre-processed embedding and labels,
    with the ability to select a specific layer from multi-layer embeddings.
    """
    def __init__(self, mode, b_path, layer_to_select=-1):
        """
        Args:
            mode (str): Dataset mode ('train', 'validation', etc.).
            b_path (str): Path to the saved embeddings and labels.
            layer_to_select (int): Index of the layer to use (-1 for last layer by default).
        """
        self.sents_reps = torch.load(b_path + f'{mode}_sents.pt')  # Load multi-layer embeddings
        self.labels = torch.load(b_path + f'{mode}_labels.pt')  # Load labels
        self.sample_num = self.labels.shape[0]
        self.layer_to_select = layer_to_select  # Specify the layer index

    def __getitem__(self, index):
        """
        Returns:
            embedding (tensor), label (int)
        """
        # Select the specified layer from the multi-layer embeddings
        if self.sents_reps.dim() == 3:
            layer_embedding = self.sents_reps[index, self.layer_to_select, :]
            return layer_embedding, self.labels[index]
        else:
            return layer_embedding, self.labels[index]

    def __len__(self):
        return self.sample_num



##############################
# DownstreamModelSingle.py  #
##############################
import torch
import torch.nn as nn
import torch.nn as nn

class DownstreamModelSingle(nn.Module):
    """
    A flexible single-embedding classification model:
      - Optionally compresses the embedding to an internal dimension (common_dim)
      - Then applies a few FC layers with improvements (normalization, adjusted dropout, 
        and alternative activations) to help training without adding extra FC layers.
    """
    def __init__(self, embed_size: int, class_num: int, common_dim: int = None):
        """
        Args:
            embed_size: The dimensionality of the input embeddings.
            class_num:  The number of output classes for classification.
            common_dim: If not None and different from embed_size, 
                        compress to this dimension first (optional).
        """
        super(DownstreamModelSingle, self).__init__()
        
        # 1) Optional compression (flexible design)
        if common_dim and (common_dim != embed_size):
            self.compress = nn.Sequential(
                nn.Linear(embed_size, common_dim),
                nn.ReLU(),          # you could also try LeakyReLU here
                nn.Dropout(0.3)     # reduced dropout rate
            )
            final_input_dim = common_dim
        else:
            self.compress = nn.Identity()  # pass-through
            final_input_dim = embed_size

        # 2) Classification MLP with improvements
        # FC Layer 1
        self.fc1 = nn.Linear(final_input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)      # Batch normalization
        self.act1 = nn.LeakyReLU()          # LeakyReLU instead of ReLU
        self.dropout1 = nn.Dropout(0.3)       # Adjusted dropout
        
        # FC Layer 2
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Final classification layer (logits)
        self.fc3 = nn.Linear(256, class_num)

    def forward(self, x):
        """
        Forward pass for a batch of embeddings x of shape (batch_size, embed_size).

        Returns:
            logits of shape (batch_size, class_num).
        """
        # Optional compression
        out = self.compress(x)
        
        # FC Layer 1
        out = self.fc1(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # FC Layer 2
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        # Final classification layer (logits)
        logits = self.fc3(out)
        return logits


import os
import sys
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

def set_seed(seed: int):
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_metrics(metrics: dict, path: str):
    """Write a simple key: value metrics file."""
    with open(path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

def evaluate_and_save(dataloader: DataLoader, model: nn.Module,
                      device: str, results_dir: str):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device).float(), y.to(device)
            logits = model(X)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    # Compute metrics
    metrics = {
        "accuracy":           accuracy_score(all_labels, all_preds),
        "balanced_accuracy":  balanced_accuracy_score(all_labels, all_preds),
        "precision":          precision_score(all_labels, all_preds, average="weighted"),
        "recall":             recall_score(all_labels, all_preds, average="weighted"),
        "f1":                 f1_score(all_labels, all_preds, average="weighted"),
    }
    save_metrics(metrics, os.path.join(results_dir, "metrics.txt"))

    # Classification report
    report = classification_report(all_labels, all_preds, digits=4)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    return metrics

def train_and_evaluate_variant(seed: int,
                               train_template: str,
                               val_test_dir: str,
                               results_base: str,
                               embed_size: int,
                               common_dim: int,
                               class_num: int,
                               batch_size: int,
                               epochs: int,
                               lr: float,
                               device: str):
    """Train & evaluate one data variant for a given seed."""
    set_seed(seed)
    # Prepare paths
    train_dir = train_template.format(seed)
    results_dir = os.path.join(results_base, f"Seed_{seed}")
    os.makedirs(results_dir, exist_ok=True)


    # DataLoaders
    train_ds = MySingleDataset(mode="train", b_path=train_dir)
    val_ds   = MySingleDataset(mode="validation", b_path=val_test_dir)
    test_ds  = MySingleDataset(mode="test", b_path=val_test_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model     = DownstreamModelSingle(embed_size, class_num, common_dim).to(device)
    loss_fn   = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        train_metrics = Train(train_loader, device, model, loss_fn, optimizer, class_num)
        curr_loss     = train_metrics.get("train_loss", None)
        if curr_loss is not None and curr_loss < best_loss:
            best_loss = curr_loss
            torch.save(model.state_dict(), os.path.join(results_dir, "best_model.pt"))

    # Load best
    best_fp = os.path.join(results_dir, "best_model.pt")
    if os.path.exists(best_fp):
        model.load_state_dict(torch.load(best_fp, map_location=device))

    # Evaluate
    _ = evaluate_and_save(val_loader, model, device, results_dir)
    test_metrics = evaluate_and_save(test_loader, model, device, results_dir)

    return test_metrics

def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed train & eval")
    p.add_argument("--seeds",            type=int, nargs="+", required=True)
    p.add_argument("--train_template",   required=True,
                   help="Template, e.g. /.../Llama-3.2-1B_{}/dataset_tensor/")
    p.add_argument("--val_test_dir",     required=True,
                   help="Path to validation+test dataset_tensor/")
    p.add_argument("--results_base",     required=True,
                   help="Where to write Seed_{n}/ subfolders")
    p.add_argument("--embed_size",       type=int, required=True)
    p.add_argument("--common_dim",       type=int, required=True)
    p.add_argument("--class_num",        type=int, default=10)
    p.add_argument("--batch_size",       type=int, default=8)
    p.add_argument("--epochs",           type=int, default=15)
    p.add_argument("--lr",               type=float, default=2e-4)
    p.add_argument("--device",           default="cuda:0")
    return p.parse_args()

def main():
    args = parse_args()
    all_results = []
    for seed in args.seeds:
        print(f"\n--- Running seed {seed} ---")
        res = train_and_evaluate_variant(
            seed=seed,
            train_template=args.train_template,
            val_test_dir=args.val_test_dir,
            results_base=args.results_base,
            embed_size=args.embed_size,
            common_dim=args.common_dim,
            class_num=args.class_num,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device=args.device,
        )
        all_results.append(res)

    # Summarize across seeds
    keys = all_results[0].keys()
    means = {k: np.mean([r[k] for r in all_results]) for k in keys}
    vars_ = {k: np.var ([r[k] for r in all_results]) for k in keys}

    summary_fp = os.path.join(args.results_base, "summary_results.txt")
    with open(summary_fp, "w") as f:
        f.write("Mean across seeds:\n")
        for k,v in means.items(): f.write(f"{k}: {v:.4f}\n")
        f.write("\nVariance across seeds:\n")
        for k,v in vars_.items(): f.write(f"{k}: {v:.4f}\n")

    print(f"\nAll done! Summary saved to {summary_fp}")

if __name__ == "__main__":
    main()
