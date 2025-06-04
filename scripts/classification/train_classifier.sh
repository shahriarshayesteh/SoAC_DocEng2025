#!/usr/bin/env bash
# Train a single-layer embedding classifier with specified hyperparameters.

set -e  # Exit immediately on error
set -u  # Treat unset variables as an error

# === CONFIGURABLE VARIABLES ===

# Model identifier used to name outputs and track the experiment
MODEL_VARIANT="llama3-8b"

# Dimensionality of the input embeddings (e.g., 4096 for LLaMA3-8B)
EMBED_SIZE=4096

# Optional projection layer size; if different from EMBED_SIZE, the model compresses before classification
COMMON_DIM=512

# Subdirectory (inside embeddings_root) where the precomputed sentence embeddings are stored
DATASET_SUBDIR="model_embeddings/Meta-Llama-3-8B"

# Number of epochs to train
EPOCHS=10

# Batch size used during training and evaluation
BATCH_SIZE=32

# Learning rate for the Adam optimizer
LEARNING_RATE=1e-4

# How often to run validation (in epochs)
VAL_INTERVAL=1

# Root folder containing all embedding subdirectories
EMBEDDINGS_ROOT="/data/sxs7285/Porjects_code/thesis/DocEng/classification/embeddings"

# Root folder where training results, metrics, and confusion matrices will be saved
RESULTS_ROOT="/data/sxs7285/Porjects_code/thesis/DocEng/classification/results"

# W&B project name (for organizing multiple runs)
WANDB_PROJECT="SoAC"

# Name of this run as it appears in W&B
WANDB_NAME="LLaMA3_Classifier_Run1"

# Optional: your W&B team or user name (can be left empty if you're using the default account)
WANDB_ENTITY="your_wandb_entity"

# Where W&B stores local logs and offline runs
LOG_DIR="./wandb_logs"

# === EXECUTION ===
python3 /data/sxs7285/Porjects_code/thesis/DocEng/SoAC-DocEng/src/SoACer_training/train_single_model.py \
  --model_variant "$MODEL_VARIANT" \
  --embed_size "$EMBED_SIZE" \
  --common_dim "$COMMON_DIM" \
  --dataset_subdir "$DATASET_SUBDIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --learning_rate "$LEARNING_RATE" \
  --val_check_interval "$VAL_INTERVAL" \
  --embeddings_root "$EMBEDDINGS_ROOT" \
  --results_root "$RESULTS_ROOT" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_NAME" \
  --wandb_entity "$WANDB_ENTITY" \
  --log_dir "$LOG_DIR"

echo "[DONE] Classifier training completed for $MODEL_VARIANT"
