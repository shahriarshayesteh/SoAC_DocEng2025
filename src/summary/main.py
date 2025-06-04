import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define summary lengths
# SUMMARY_LENGTHS = [2,10,15,20,25,30]
SUMMARY_LENGTHS = [4]


# Base directories for experiment
# BASE_DIR = "/data/sxs7285/Porjects_code/thesis/SoAC/len_exp/"
BASE_DIR ="/data/sxs7285/Porjects_code/thesis/DocEng/summary/len_exp/"
RESULTS_BASE_DIR = "/data/sxs7285/Porjects_code/thesis/DocEng/summary/len_exp/results"

# Function to save results
def save_results(results, file_path):
    with open(file_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

def evaluate_and_save(dataloader, model, device, results_dir):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            softmax_output = torch.softmax(outputs, dim=1)
            preds = torch.argmax(softmax_output, dim=1)
            all_probs.extend(softmax_output.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    
    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average='weighted'),
        "recall": recall_score(all_labels, all_preds, average='weighted'),
        "f1": f1_score(all_labels, all_preds, average='weighted')
    }
    save_results(metrics, os.path.join(results_dir, "metrics.txt"))
    
    # Save classification report
    report = classification_report(all_labels, all_preds, digits=4)
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()
    return metrics

import sys
sys.path.append("/data/sxs7285/Porjects_code/thesis/SoAC/soac/classification/")
from single_llm import MySingleDataset, DownstreamModelSingle, Train, Test

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Iterate over summary lengths
for summary_length in SUMMARY_LENGTHS:
    print(f"Running experiment with summary length {summary_length}...")
    
    # Define paths
    # DATASET_DIR = os.path.join(BASE_DIR, f"sc{summary_length}/Meta-Llama-3-8B/dataset_tensor/")
    # RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f"sc{summary_length}/Meta-Llama-3-8B/dataset_tensor/")
    DATASET_DIR = os.path.join(BASE_DIR, f"sc{summary_length}/Meta-Llama-3-8B/dataset_tensor/")
    RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f"sc{summary_length}/Meta-Llama-3-8B/dataset_tensor/")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load dataset
    train_data = MySingleDataset(mode='train', b_path=DATASET_DIR)
    val_data = MySingleDataset(mode='validation', b_path=DATASET_DIR)
    # test_data = MySingleDataset(mode='test', b_path=DATASET_DIR)
    test_data = MySingleDataset(mode='validation', b_path=DATASET_DIR)


    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # Define model
    model = DownstreamModelSingle(embed_size=4096, class_num=10, common_dim=4096).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    
    # Training loop
    for epoch in range(1, 15):
        Train(dataloader=train_loader, device=device, model=model, loss_fn=loss_fn, optimizer=optimizer, class_num=10)

    # Validation and test evaluation
    val_metrics = evaluate_and_save(val_loader, model, device, RESULTS_DIR)
    test_metrics = evaluate_and_save(test_loader, model, device, RESULTS_DIR)

print("Experiment completed. Results saved.")
