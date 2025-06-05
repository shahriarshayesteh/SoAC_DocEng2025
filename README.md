
# SoAC and SoACer: A Sector-Based Corpus and LLM-Based Framework for Sectoral Website Classification

## Introduction

One approach to understanding the vastness and complexity of the web is to categorize websites into sectors that reflect the specific industries or domains in which they operate. Existing classification approaches often struggle with noisy, unstructured, and lengthy web content, and current datasets lack a universal, web-tailored sector labeling system. To address these issues, we introduce **SoAC** (Sector of Activity Corpus), a large-scale corpus of 195,495 websites categorized into 10 broad, web-specific sectors, and **SoACer** (Sector of Activity Classifier), an end-to-end LLM-based framework that:

1. Fetches website information
2. Uses extractive summarization (LexRank) to condense noisy content
3. Generates lightweight LLM embeddings (Llama3-8B) followed by a classification head to predict each website’s sector.

Through extensive experiments (including ablation studies and error analysis), SoACer achieves **72.6% overall accuracy** on SoAC.

---

## SoAC Corpus

To load the SoAC dataset directly from Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("Shahriar/SoAC_Corpus")
```

This automatically returns three splits:

* **train** (109,476 websites, 56 %)
* **validation** (27,370 websites, 14 %)
* **test** (58,649 websites, 30 %)&#x20;

You can also browse or download it from:
[https://huggingface.co/datasets/Shahriar/SoAC\_Corpus](https://huggingface.co/datasets/Shahriar/SoAC_Corpus)

---

## SoACer

SoACer is a three-stage pipeline (Pre-processing → Inference → Post-processing) that uses extractive summarization + LLM embeddings + a lightweight classification head. Figure 2 below illustrates the high-level architecture:

<p align="center">
  <img width="1644" alt="Figure 2: SoACer Framework" width="80%" src="https://github.com/user-attachments/assets/5787ad8b-4604-4214-a053-155a5adbbe71" />

</p>


Here's the **updated Inference section** for your `README.md`, incorporating both the **single URL sync pipeline** and the **batch async pipeline**:

---

### Inference

The SoACer framework supports both **single URL prediction** and **batch asynchronous prediction** for sector classification.

---

#### Single URL Prediction

Run inference on a single website URL using the synchronous pipeline:

```bash
python src/inference/SoACer_pipeline.py \
  --input "https://example.com" \
  --output_dir results/
```

##### What This Does:

1. Crawls the provided URL (depth-limited).
2. Extracts main textual content using Boilerpipe.
3. Applies LexRank to summarize the content.
4. Computes a mean-pooled embedding using LLaMA3-8B.
5. Passes the embedding through the fine-tuned classification head.
6. Saves a `.json` file to `results/` containing:

   * **Top-1 predicted sector** + confidence
   * **Confidence scores** for all 10 sectors
   * **Extracted summary** used for classification

---

#### Batch Async Prediction

To process multiple URLs asynchronously, use the async script:

```bash
bash scripts/SoACer/SoACer_pipeline_async.sh input_urls.txt output_dir/
```

* `input_urls.txt` should contain one URL per line.
* `output_dir/` is where all prediction results will be saved.

Each URL will generate a JSON file (similar to the single-mode output), stored in the output directory with the format:

```
output_dir/<url_hostname>.json
```

This script uses `src/inference/SoACer_pipeline_async.py` and supports concurrent crawling and classification for faster batch processing.

---

## Experiments

Below are three main experiment workflows. Each section assumes you have a `scripts/` folder with corresponding helper scripts; adjust paths as needed.

### Preparation

Before running any experiments, clone the repository and install the required packages:

```bash
# Clone the repository
git clone https://github.com/your-username/SoAC-SoACer.git
cd SoAC-SoACer

# Create a Python virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If you prefer Conda, you can create and activate an environment as follows:
>
> ```bash
> conda create -n soacer_env python=3.8 -y
> conda activate soacer_env
> pip install --upgrade pip
> pip install -r requirements.txt
> ```

### Summary Generation

This step produces extractive summaries for every website in SoAC (used later for training/inference). By default, we extract 20 sentences per document.

```bash
bash scripts/summary/generate_summary.sh 20 12 data/summaries/
```

This script downloads SoAC from Hugging Face if needed, then generates 20-sentence summaries using 12 workers. The summaries are saved under `data/summaries/`. 

Here's an updated **Training Section** for your README to clearly reflect the **two-step SoACer training process**, based on the scripts `generate_embeddings.sh` and `train_classifier.sh`, and aligned with the methodology outlined in your DocEng paper:

---

### SoACer Training

To train the **SoACer** classifier, follow the **two-step pipeline**: first generate sentence embeddings using an LLM, then train the classification head on these embeddings.

#### Step 1: Generate Embeddings

We use a frozen LLM (default: LLaMA3-8B) to embed LexRank summaries for each website. You can run the embedding generation with default or overridden parameters.

```bash
bash scripts/classification/generate_embeddings.sh
```

To customize settings (e.g., change model or output directory), override variables like this:

```bash
MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
OUTPUT_DIR="embeddings/deepseek_run" \
MAX_LEN=1024 \
BATCH_SIZE=8 \
bash scripts/classification/generate_embeddings.sh
```

This will generate embeddings for the train/validation/test splits and save them under your specified output directory (default: `embeddings/model_embeddings`).

---

#### Step 2: Train Classifier

After embeddings are generated, train a **classification head** (a 2-layer MLP) on top of them.

```bash
bash scripts/classification/train_classifier.sh
```

You can adjust the classifier hyperparameters and training settings by editing the variables inside the script, such as:

* `MODEL_VARIANT="llama3-8b"`
* `EMBED_SIZE=4096`
* `COMMON_DIM=512`
* `EPOCHS=10`
* `BATCH_SIZE=32`
* `LEARNING_RATE=1e-4`

By default, the training uses:

* Embeddings from: `classification/embeddings/model_embeddings/Meta-Llama-3-8B`
* Output results to: `classification/results/`

The best model (lowest validation loss) is saved automatically. Training logs and evaluation metrics are recorded under `wandb_logs/`.


Here's the **updated Ablation Study section** for your `README.md`, reflecting the use of the new `run_embedding.sh` and `run_ablation.sh` scripts for full-text vs. summary evaluation:

---

### Ablation Study: Full-Text vs. Summary-Based Classification

To assess the impact of extractive summarization, we compare the SoACer framework's performance using full-text content vs. summary-based input. This experiment evaluates classification accuracy, efficiency, and robustness across input types using different LLM embeddings.

#### Step 1: Generate Embeddings

Use the following script to generate embeddings for both full-text and summary-based datasets using any LLM (e.g., LLaMA3, DeepSeek, etc.).

```bash
bash scripts/ablation/run_embedding.sh <MODEL_ID> [BATCH_SIZE] [MAX_LEN] [TASK_NAME] [OUTPUT_DIR]
```

##### Example:

```bash
bash scripts/ablation/run_embedding.sh meta-llama/Meta-Llama-3-8B 8 1024 fulltext_embeddings ablation/embeddings
```

This will create embeddings under:

```
ablation/embeddings/fulltext_embeddings/Meta-Llama-3-8B/dataset_tensor/
```

Repeat the process for both summary and full-text variants if desired (e.g., different `TASK_NAME`s like `summary_embeddings`, `fulltext_embeddings`, etc.).

---

#### Step 2: Run Ablation Training and Evaluation

Use the `run_ablation.sh` script to train and evaluate classifiers on the generated embeddings:

```bash
bash scripts/ablation/run_ablation.sh <MODEL_NAME> [EMBED_SIZE] [COMMON_DIM] [SEEDS]
```

* `MODEL_NAME`: Folder name used under `data/websector/` (e.g., `Llama-3.2-1B`)
* `EMBED_SIZE`: Embedding vector size (e.g., 2048)
* `COMMON_DIM`: Projection size before classification (e.g., 512 or 2048)
* `SEEDS`: (Optional) Comma-separated list or a single random seed (default: 12)

##### Example:

```bash
bash scripts/ablation/run_ablation.sh Llama-3.2-1B 2048 512 12
```

This command will:

* Load embeddings from `data/websector/Llama-3.2-1B/12/dataset_tensor/`
* Train a classifier and log results to `ablation_results/Llama-3.2-1B/`

---

## Citation

If you use SoAC or SoACer in your work, please cite:

> Shayesteh, S., Srinath, M., Matheson, L., Xian, L., Saha, S., Giles, C. L., & Wilson, S. (2025).
> SoAC and SoACer: A Sector-Based Corpus and LLM-Based Framework for Sectoral Website Classification. In *Proceedings of \[Conference Name]*.

Full reference details and DOI will be updated once available.


---

## License

Place Holder. 
