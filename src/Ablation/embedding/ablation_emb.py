#!/usr/bin/env python3
"""
Ablation-style embedding generator with filtering & multi-seed support.

Example:
  python run_ablation.py \
    --task exp2 \
    --model_id meta-llama/Llama-3.2-1B \
    --dataset_name Shahriar/SoAC_Corpus \
    --splits train validation test \
    --seeds 12 21 36 42 80 \
    --max_len 1024 \
    --step 1 \
    --max_tokens 7000 \
    --samples_per_class 2000 \
    --output_root /data/.../Ablation/embedding/embeding_saved \
    --filter_on content
"""

import os
import sys
import argparse
import random
import torch
from tqdm import trange
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Coarse-grained label map
COARSE_GRAIN = {
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

def estimate_token_count(text: str) -> float:
    return len(text.split()) * 1.3  # approx

def filter_samples(
    contents: list[str],
    summaries: list[str],
    labels: list[str],
    max_tokens: int,
    samples_per_class: int | None,
    seed: int
) -> tuple[list[str], list[str]]:
    """Filter by content length & optionally subsample per class."""
    filtered = [
        (summ, lab)
        for cont, summ, lab in zip(contents, summaries, labels)
        if estimate_token_count(cont) <= max_tokens
    ]
    if samples_per_class is None:
        return zip(*filtered)
    # group by class
    class_groups: dict[str, list[tuple[str,str]]] = {}
    for summ, lab in filtered:
        class_groups.setdefault(lab, []).append((summ, lab))
    final = []
    random.seed(seed)
    for lab, items in class_groups.items():
        choose = random.sample(items, min(len(items), samples_per_class))
        final.extend(choose)
    return zip(*final)

def multi_rep_extract(
    task: str,
    split: str,
    summaries: list[str],
    labels: list[str],
    model_id: str,
    max_len: int,
    step: int,
    seed: int,
    output_root: str
):
    """Generate & save 5-layer mean-pooled embeddings."""
    # tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(
        model_id, trust_remote_code=True, output_hidden_states=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.gradient_checkpointing_enable()
    model.eval()

    reps = []
    for i in trange(0, len(summaries), step, desc=f"{model_id} {split}"):
        batch = summaries[i : i + step]
        enc = tokenizer(batch, return_tensors="pt",
                        max_length=max_len, padding="max_length", truncation=True)
        # ensure on GPU shards
        for k, v in enc.items():
            enc[k] = v.to("cuda", non_blocking=True)
        torch.cuda.empty_cache()

        with torch.no_grad():
            out = model(**enc)
            layers = [out.hidden_states[l].mean(dim=1) for l in range(-1, -6, -1)]
            reps.append(torch.stack(layers, dim=1).cpu())

        del enc, out, layers
        torch.cuda.empty_cache()

    reps = torch.cat(reps, dim=0)
    lbl_tensors = torch.stack([torch.tensor(COARSE_GRAIN[l]) for l in labels])

    save_dir = os.path.join(
        output_root,
        task,
        model_id.split("/")[-1] + f"_{seed}",
        "dataset_tensor"
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(reps, os.path.join(save_dir, f"{split}_sents.pt"))
    torch.save(lbl_tensors, os.path.join(save_dir, f"{split}_labels.pt"))

    del model, tokenizer, reps, lbl_tensors
    torch.cuda.empty_cache()

def parse_args():
    p = argparse.ArgumentParser(description="Ablation embedding pipeline")
    p.add_argument("--task",        required=True)
    p.add_argument("--model_id",    required=True)
    p.add_argument("--dataset_name",required=True)
    p.add_argument("--splits",      nargs="+",
                     default=["train"], help="e.g., train validation test")
    p.add_argument("--seeds",       nargs="+", type=int,
                     default=[12,21,36,42,80])
    p.add_argument("--max_len",     type=int, default=512)
    p.add_argument("--step",        type=int, default=2)
    p.add_argument("--max_tokens",  type=int, default=7000)
    p.add_argument("--samples_per_class", type=int, default=2000,
                     help="None => no subsampling",)
    p.add_argument("--output_root", required=True)
    p.add_argument("--filter_on",   choices=["content","summary"],
                     default="content", help="Field to measure length")
    return p.parse_args()

def main():
    args = parse_args()
    ds = load_dataset(args.dataset_name)

    for seed in args.seeds:
        random.seed(seed)
        for split in args.splits:
            data = ds[split]
            summaries = data["Website_Summary"]
            labels    = data["Coarse_Grained_Sector_Label"]
            contents  = data["Website_Content"]

            if args.filter_on == "content":
                filtered_summaries, filtered_labels = filter_samples(
                    contents, summaries, labels,
                    max_tokens=args.max_tokens,
                    samples_per_class=(None if args.samples_per_class<0 else args.samples_per_class),
                    seed=seed
                )
            else:
                # length filter on summaries
                filtered_summaries, filtered_labels = filter_samples(
                    summaries, summaries, labels,
                    max_tokens=args.max_tokens,
                    samples_per_class=(None if args.samples_per_class<0 else args.samples_per_class),
                    seed=seed
                )

            print(f"[Seed {seed} | {split}] {len(filtered_summaries)} samples")
            multi_rep_extract(
                task=args.task,
                split=split,
                summaries=list(filtered_summaries),
                labels=list(filtered_labels),
                model_id=args.model_id,
                max_len=args.max_len,
                step=args.step,
                seed=seed,
                output_root=args.output_root,
            )
            print(f"[Seed {seed} | {split}] done.\n")

if __name__ == "__main__":
    main()
