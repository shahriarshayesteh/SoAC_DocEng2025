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


##############################
# model_ops_single.py       #
##############################
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
import numpy as np

def Train(dataloader, device, model, loss_fn, optimizer, class_num, wandb_logger=None):
    """
    Trains for one epoch, returning a dictionary of averaged metrics:
      - train_loss
      - train_accuracy
      - train_macro_f1
      - train_micro_f1
      - train_weighted_f1
      - train_weighted_precision
      - train_weighted_recall
      - train_weighted_accuracy
    """
    model.train()
    
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    for batch_emb, batch_labels in tqdm(dataloader, desc="Training"):
        batch_emb = batch_emb.to(device).float()
        batch_labels = batch_labels.to(device)

        # Forward pass => logits
        logits = model(batch_emb)
        
        # Compute loss
        loss = loss_fn(logits, batch_labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

        # Predictions
        pred_y = torch.argmax(logits, dim=1)
        all_preds.extend(pred_y.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    # Calculate aggregated metrics
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    weighted_precision = precision_score(all_labels, all_preds, average='weighted')
    weighted_recall = recall_score(all_labels, all_preds, average='weighted')
    weighted_accuracy = acc  # Weighted accuracy typically the same as overall acc

    metrics_dict = {
        'train_loss': avg_loss,
        'train_accuracy': acc,
        'train_macro_f1': macro_f1,
        'train_micro_f1': micro_f1,
        'train_weighted_f1': weighted_f1,
        'train_weighted_precision': weighted_precision,
        'train_weighted_recall': weighted_recall,
        'train_weighted_accuracy': weighted_accuracy,
    }

    # Log to wandb
    if wandb_logger is not None:
        wandb_logger.log(metrics_dict)

    return metrics_dict


def Test(dataloader, device, model, loss_fn, class_num, wandb_logger=None, mode="val"):
    """
    Evaluates a single-embedding classifier, returning a dictionary of metrics:
      - {mode}_loss
      - {mode}_accuracy
      - {mode}_macro_f1
      - {mode}_micro_f1
      - {mode}_weighted_f1
      - {mode}_weighted_precision
      - {mode}_weighted_recall
      - {mode}_weighted_accuracy
    """
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_emb, batch_labels in tqdm(dataloader, desc=f"Evaluating({mode})"):
            batch_emb = batch_emb.to(device).float()
            batch_labels = batch_labels.to(device)

            # Forward
            logits = model(batch_emb)
            loss = loss_fn(logits, batch_labels)

            total_loss += loss.item()
            num_batches += 1

            # Predictions
            pred_y = torch.argmax(logits, dim=1)
            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    avg_loss = total_loss / num_batches

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    weighted_precision = precision_score(all_labels, all_preds, average='weighted')
    weighted_recall = recall_score(all_labels, all_preds, average='weighted')
    weighted_accuracy = acc

    metrics_dict = {
        f'{mode}_loss': avg_loss,
        f'{mode}_accuracy': acc,
        f'{mode}_macro_f1': macro_f1,
        f'{mode}_micro_f1': micro_f1,
        f'{mode}_weighted_f1': weighted_f1,
        f'{mode}_weighted_precision': weighted_precision,
        f'{mode}_weighted_recall': weighted_recall,
        f'{mode}_weighted_accuracy': weighted_accuracy,
    }

    if wandb_logger is not None:
        wandb_logger.log(metrics_dict)

    return metrics_dict


################
# main_single.py
################
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
# 1. LABEL MAPPING
# ──────────────────────────────────────────────────────────────────────────────
LABEL_MAPPING = {
    "finance, marketing & human resources": 0,
    "information technology & electronics": 1,
    "consumer & supply chain": 2,
    "civil, mechanical & electrical": 3,
    "medical": 4,
    "sports, media & entertainment": 5,
    "education": 6,
    "government, defense & legal": 7,
    "travel, food & hospitality": 8,
    "non-profit": 9,
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

# ──────────────────────────────────────────────────────────────────────────────
# 2. EVALUATION & SAVE
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_and_save_single(dataloader, model, device, results_dir):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    df = pd.DataFrame({
        "True Label":      [REVERSE_LABEL_MAPPING[l] for l in all_labels],
        "Predicted Label": [REVERSE_LABEL_MAPPING[p] for p in all_preds],
        "Probabilities":   all_probs
    })
    df.to_csv(os.path.join(results_dir, "predictions.csv"), index=False)

    acc = accuracy_score(all_labels, all_preds)
    w_acc = balanced_accuracy_score(all_labels, all_preds)
    w_prec = precision_score(all_labels, all_preds, average="weighted")
    w_rec = recall_score(all_labels, all_preds, average="weighted")
    w_f1 = f1_score(all_labels, all_preds, average="weighted")

    with open(os.path.join(results_dir, "test_results.txt"), "w") as f:
        f.write(f"Overall Accuracy:  {acc:.4f}\n")
        f.write(f"Weighted Accuracy: {w_acc:.4f}\n")
        f.write(f"Weighted Precision:{w_prec:.4f}\n")
        f.write(f"Weighted Recall:   {w_rec:.4f}\n")
        f.write(f"Weighted F1:       {w_f1:.4f}\n")

    report = classification_report(all_labels, all_preds, digits=4)
    with open(os.path.join(results_dir, "class_based_results.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    class_names = [None] * len(LABEL_MAPPING)
    for name, idx in LABEL_MAPPING.items():
        class_names[idx] = name

    plt.figure(figsize=(16, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    print(f"[✔] Evaluation artifacts saved to {results_dir}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. TRAINING LOGIC
# ──────────────────────────────────────────────────────────────────────────────
def run_single(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = os.path.join(args.embeddings_root, args.dataset_subdir, "dataset_tensor")
    results_dir = os.path.join(args.results_root, args.model_variant + "_single")
    os.makedirs(results_dir, exist_ok=True)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        dir=args.log_dir,
        config={
            "embed_size": args.embed_size,
            "common_dim": args.common_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    )


    train_ds = MySingleDataset("train", b_path=dataset_root)
    val_ds = MySingleDataset("validation", b_path=dataset_root)
    test_ds = MySingleDataset("test", b_path=dataset_root)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DownstreamModelSingle(args.embed_size, class_num=10, common_dim=args.common_dim).to(device)
    wandb.watch(model, log="all")
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_loss = float("inf")
    best_model_fp = os.path.join(results_dir, "best_model.pt")

    for ep in range(1, args.epochs + 1):
        print(f"\n=== Epoch {ep}/{args.epochs} ===")
        train_metrics = Train(train_loader, device, model, loss_fn, optimizer, 10, wandb)
        print("Train:", train_metrics)

        if train_metrics["train_loss"] < best_loss:
            best_loss = train_metrics["train_loss"]
            torch.save(model.state_dict(), best_model_fp)

        if ep % args.val_check_interval == 0:
            val_metrics = Test(val_loader, device, model, loss_fn, 10, wandb, mode="val")
            print("Val:", val_metrics)

    # Final Test
    if os.path.exists(best_model_fp):
        model.load_state_dict(torch.load(best_model_fp, map_location=device))
    Test(test_loader, device, model, loss_fn, 10, wandb, mode="test")
    evaluate_and_save_single(test_loader, model, device, results_dir)

    wandb.finish()

# ──────────────────────────────────────────────────────────────────────────────
# 4. ARGPARSE
# ──────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train single-embedding classifier.")

    parser.add_argument("--model_variant", type=str, required=True)
    parser.add_argument("--embed_size", type=int, required=True)
    parser.add_argument("--common_dim", type=int, default=512)
    parser.add_argument("--dataset_subdir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--val_check_interval", type=int, default=1)

    parser.add_argument("--embeddings_root", type=str, required=True)
    parser.add_argument("--results_root", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="SoAC")
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./wandb_logs")
    parser.add_argument("--wandb_entity", type=str, default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    run_single(args)

if __name__ == "__main__":
    main()
