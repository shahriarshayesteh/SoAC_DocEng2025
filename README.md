
# SoAC and SoACer: A Sector-Based Corpus and LLM-Based Framework for Sectoral Website Classification

## Introduction

One approach to understanding the vastness and complexity of the web is to categorize websites into sectors that reflect the specific industries or domains in which they operate. Existing classification approaches often struggle with noisy, unstructured, and lengthy web content, and current datasets lack a universal, web-tailored sector labeling system. To address these issues, we introduce **SoAC** (Sector of Activity Corpus), a large-scale corpus of 195,495 websites categorized into 10 broad, web-specific sectors, and **SoACer** (Sector of Activity Classifier), an end-to-end LLM-based framework that:

1. Fetches website information
2. Uses extractive summarization (LexRank) to condense noisy content
3. Generates lightweight LLM embeddings (Llama3-8B) followed by a classification head to predict each website’s sector .

Through extensive experiments (including ablation studies and error analysis), SoACer achieves **72.6% overall accuracy** on SoAC, demonstrating that extractive summarization not only reduces computational overhead but also improves classification performance .

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


### Pre-processing

1. **Raw text or URL**

   * If you supply raw text, it directly goes to summarization.
   * If you supply a website URL, SoACer crawls the landing page (plus depth-1 internal links) and extracts boilerplate-free content using Boilerpipe.

2. **Summarization (LexRank)**

   * By default, we extract the top 20 sentences per website (≈ 765 tokens), which empirically yields the best trade-off between accuracy and efficiency .

### Inference

```bash
python src/soacer_inference.py \
  --input_type url \
  --input_value "https://example.com" \
  --output_file results.json
```

* This script will:

  1. Crawl the URL, extract text, and run LexRank.
  2. Compute a mean-pooled embedding from Llama3-8B.
  3. Pass the embedding through a fine-tuned MLP head to predict one of 10 sectors.
  4. Save a JSON containing:

     * Top-1 sector + confidence
     * All sector confidence scores
     * Generated summary
     * Raw scraped text

> **Note:** Adjust `--input_type {url,text}` and point `--input_value` accordingly.

### Post-processing

* The inference script outputs a JSON (or CSV) where each entry includes:

  * `url` (or identifier)
  * `predicted_sector`
  * `confidence_scores` (dict of all 10 sectors)
  * `summary_text`
  * `raw_text`

---

## Experiments

Below are three main experiment workflows. Each section assumes you have a `scripts/` folder with corresponding helper scripts; adjust paths as needed.

### 1. Summary Generation

This step produces extractive summaries for every website in SoAC (used later for training/inference). By default, we extract 20 sentences per document.

```bash
bash scripts/generate_summaries.sh \
  --input_dir data/raw_html/ \
  --output_dir data/summaries/ \
  --sentences_count 20
```

* `--input_dir`: directory of raw-HTML files (or plain-text versions) for each website
* `--output_dir`: directory where summaries (one JSON per site) are saved
* `--sentences_count`: number of sentences to extract (e.g., 2, 4, 10, 15, 20, etc.)

> To test different lengths (e.g., sc2, sc4, sc10, sc15, sc20, sc25, sc30), modify `--sentences_count`. The best validation performance was at 20 sentences (≈ 72.3 % accuracy) .

### 2. SoACer Training

Train SoACer on the summarized corpus. By default, the hyperparameters are:

* **Epochs**: 15
* **Batch size**: 8
* **Learning rate**: 2 × 10⁻⁴
* **Dropout**: 0.3&#x20;

```bash
bash scripts/train_soacer.sh \
  --train_data data/summaries/train.jsonl \
  --valid_data data/summaries/validation.jsonl \
  --model_output_dir models/soacer_20sent/ \
  --epochs 15 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --dropout 0.3
```

Upon completion, the best-checkpoint (lowest validation loss) is saved under `models/soacer_20sent/`. The test accuracy will be close to **72.6 %** when using Llama3-8B embeddings .

### 3. Ablation Study (Full-text vs. Summary)

To compare full-text classification (subsampled to ≤ 7,000 tokens) versus summary-based (20 sentences), run:

```bash
# Full-text variant (using Llama-3.2-1B, subsampled dataset)
bash scripts/run_ablation.sh \
  --mode full_text \
  --train_data data/full_text/train_subsampled.jsonl \
  --valid_data data/full_text/valid_subsampled.jsonl \
  --test_data data/full_text/test_subsampled.jsonl \
  --model_output_dir models/ablation_fulltext/

# Summary-based variant (same Llama-3.2-1B, using 20-sentence summaries)
bash scripts/run_ablation.sh \
  --mode summary \
  --train_data data/summaries/train.jsonl \
  --valid_data data/summaries/validation.jsonl \
  --test_data data/summaries/test.jsonl \
  --model_output_dir models/ablation_summary/
```

After training, compare metrics (accuracy, balanced accuracy, weighted F1, etc.). In our experiments, summary-based outperformed full-text by:

* * 3.5 % overall accuracy
* * 3.2 % balanced accuracy
* * 3.8 % weighted F1 .

---

## Hyperparameters & Implementation Details

All code is implemented in Python 3 (≥ 3.8) with PyTorch 2.x and Hugging Face Transformers. We recommend using a machine with ≥ 1 GPU (≥ 12 GB VRAM) for training:

* **LexRank summarization**: uses a thresholded cosine-similarity graph and PageRank (via the `lexrank` package).
* **Embedding layer**: Meta-Llama-3-8B (frozen). We extract mean-pooled embeddings from the last hidden layer.
* **Classification head** (MLP):

  1. (optional) Linear compression → BatchNorm → LeakyReLU → Dropout 0.3
  2. FC layer (512 units) → BatchNorm → LeakyReLU → Dropout 0.3
  3. FC layer (256 units) → BatchNorm → LeakyReLU → Dropout 0.3
  4. Final linear → 10-way softmax .

Detailed hyperparameters are in `scripts/train_soacer.sh` (and appendix of the paper).

---

## Citation

If you use SoAC or SoACer in your work, please cite:

> Shayesteh, S., Srinath, M., Matheson, L., Xian, L., Saha, S., Giles, C. L., & Wilson, S. (2025).
> SoAC and SoACer: A Sector-Based Corpus and LLM-Based Framework for Sectoral Website Classification. In *Proceedings of \[Conference Name]*.

Full reference details and DOI will be updated once available.


---

## License

Place Holder. 
