#!/usr/bin/env python3
"""
LexRank summarization for SoAC website content from Hugging Face Hub.

Example usage:
  python run_lexrank.py \
    --output_dir /data/.../soac/summary/lexrank/ \
    --splits train validation test \
    --sentences_count 15 \
    --num_workers 12
"""

import os
import json
import time
import gc
import argparse
import psutil
import nltk


from typing import List, Dict, Tuple, Optional, Any, Union

import pandas as pd
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

def summarize_text(text: str, sentences_count: int) -> str:
    if not text or not text.strip():
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summarizer.threshold = 0.1
    summarizer.epsilon = 0.05
    summary_sentences = summarizer(parser.document, sentences_count)
    return " ".join(str(sentence) for sentence in summary_sentences)

def process_batch(texts: List[str], sentences_count: int) -> List[str]:
    return [summarize_text(t, sentences_count) for t in texts]

def log_system_usage():
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 ** 2)
    cpu_pct = psutil.cpu_percent(interval=None)
    print(f"[LOG] Memory usage: {mem_mb:.2f} MB | CPU usage: {cpu_pct:.1f}%")

def process_split(
    split: str,
    output_dir: str,
    sentences_count: int,
    num_workers: int,
):
    start_ts = time.time()
    print(f"\n=== Processing split: '{split}' ===")

    dataset = load_dataset("Shahriar/SoAC_Corpus", split=split)
    df = pd.DataFrame(dataset)
    num_rows = len(df)
    print(f"[INFO] Loaded {num_rows} rows from split '{split}'")

    texts: List[str] = df["Website_Content"].astype(str).tolist()

    if num_workers <= 0:
        num_workers = 1
    batch_size = num_rows // num_workers if num_workers < num_rows else 1
    batches: List[List[str]] = []
    idx = 0
    for w in range(num_workers):
        end = idx + batch_size
        if w == num_workers - 1:
            end = num_rows
        batches.append(texts[idx:end])
        idx = end

    print(f"[INFO] Running parallel summarization: {num_workers} workers.")
    summaries: List[str] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_batch, batch, sentences_count)
                   for batch in batches]
        for f in futures:
            summaries.extend(f.result())

    if len(summaries) != num_rows:
        print(f"[ERROR] Summary count mismatch: expected {num_rows}, got {len(summaries)}")
        return

    new_col = f"sum_lexrank_sc{sentences_count}"
    df[new_col] = summaries

    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"lexRank_summary_sc{sentences_count}_{split}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote summarized DataFrame to '{out_csv}'")

    elapsed = time.time() - start_ts
    timing = {
        "split": split,
        "sentences_count": sentences_count,
        "duration_seconds": elapsed,
    }
    timing_json = os.path.join(output_dir, f"lexRank_summary_sc{sentences_count}_{split}_time.json")
    with open(timing_json, "w") as fp:
        json.dump(timing, fp, indent=2)

    print(f"[INFO] Timing written to '{timing_json}'")
    log_system_usage()
    gc.collect()
    print(f"--- Completed split '{split}' in {elapsed:.2f} seconds ---\n")

def parse_args():
    p = argparse.ArgumentParser(
        description="LexRank summarization for SoAC website content (HuggingFace version)."
    )
    p.add_argument(
        "--output_dir", "-o",
        required=True,
        help="Folder where LexRank summary CSVs and timing JSONs will be saved"
    )
    p.add_argument(
        "--splits", "-s",
        nargs="+",
        default=["train", "validation", "test"],
        help="List of splits to process"
    )
    p.add_argument(
        "--sentences_count", "-n",
        type=int,
        default=15,
        help="Number of sentences to extract per document"
    )
    p.add_argument(
        "--num_workers", "-w",
        type=int,
        default=12,
        help="Number of parallel processes to use"
    )
    return p.parse_args()

def main():
    args = parse_args()

    print("\n=== LexRank Summarization Script (HuggingFace) ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Splits:           {args.splits}")
    print(f"Sentences count:  {args.sentences_count}")
    print(f"Num workers:      {args.num_workers}\n")

    for split in args.splits:
        process_split(
            split=split,
            output_dir=args.output_dir,
            sentences_count=args.sentences_count,
            num_workers=args.num_workers,
        )

    print("All splits completed.")

if __name__ == "__main__":
    main()




# ./generate_summary.sh 20 8 /data/sxs7285/Porjects_code/thesis/DocEng/SoAC-DocEng/src/summary/summary_results/sc20