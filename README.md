
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


### Inference

```bash
python src/inference/SoACer_pipeline.py \
  --input "https://example.com" \
  --output_dir ./results
```

* This script will:

  1. Crawl the URL, extract text, and run LexRank.
  2. Compute a mean-pooled embedding from Llama3-8B.
  3. Pass the embedding through a fine-tuned MLP head to predict one of 10 sectors.
  4. Save a JSON containing:

     * Top-1 sector + confidence
     * All sector confidence scores
     * Generated summary

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

### 1. Summary Generation

This step produces extractive summaries for every website in SoAC (used later for training/inference). By default, we extract 20 sentences per document.

```bash
bash scripts/summary/generate_summary.sh \
  --input_dir data/raw_html/ \
  --output_dir data/summaries/ \
  --sentences_count 20
```

* `--input_dir`: directory of raw-HTML files (or plain-text versions) for each website
* `--output_dir`: directory where summaries (one JSON per site) are saved
* `--sentences_count`: number of sentences to extract (e.g., 2, 4, 10, 15, 20, etc.)

> To test different lengths (e.g., sc2, sc4, sc10, sc15, sc20, sc25, sc30), modify `--sentences_count`. 

### 2. SoACer Training

Train SoACer on the summarized corpus. By default, the hyperparameters are:

* **Epochs**: 15
* **Batch size**: 8
* **Learning rate**: 2 × 10⁻⁴
* **Dropout**: 0.3&#x20;

```bash
bash scripts/classification/train_classifier.sh \
  --train_data data/summaries/train.jsonl \
  --valid_data data/summaries/validation.jsonl \
  --model_output_dir models/soacer_20sent/ \
  --epochs 15 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --dropout 0.3
```

Upon completion, the best checkpoint (lowest validation loss) is saved under `models/soacer_20sent/`.

### 3. Ablation Study (Full-text vs. Summary)

To compare full-text classification (subsampled to ≤ 7,000 tokens) versus summary-based (20 sentences), run:

```bash
# Full-text variant (using Llama-3.2-1B, subsampled dataset)
bash scripts/ablation/run_ablation.sh \
  --mode full_text \
  --train_data data/full_text/train_subsampled.jsonl \
  --valid_data data/full_text/valid_subsampled.jsonl \
  --test_data data/full_text/test_subsampled.jsonl \
  --model_output_dir models/ablation_fulltext/

# Summary-based variant (same Llama-3.2-1B, using 20-sentence summaries)
bash scripts/ablation/run_ablation.sh \
  --mode summary \
  --train_data data/summaries/train.jsonl \
  --valid_data data/summaries/validation.jsonl \
  --test_data data/summaries/test.jsonl \
  --model_output_dir models/ablation_summary/
```
---

## Citation

If you use SoAC or SoACer in your work, please cite:

> Shayesteh, S., Srinath, M., Matheson, L., Xian, L., Saha, S., Giles, C. L., & Wilson, S. (2025).
> SoAC and SoACer: A Sector-Based Corpus and LLM-Based Framework for Sectoral Website Classification. In *Proceedings of \[Conference Name]*.

Full reference details and DOI will be updated once available.


---

## License

Place Holder. 
