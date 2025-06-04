#!/usr/bin/env python3
"""
Multi-GPU embedding extraction from SoAC_Corpus using causal LLMs.
"""

import os
import torch
import json
from tqdm import trange
from datasets import load_dataset
import argparse
from typing import List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig
)

# Label encoding
coarse_grain = {
    "finance, marketing & human resources": 0,
    "information technology & electronics": 1,
    "consumer & supply chain": 2,
    "civil, mechanical & electrical": 3,
    "medical": 4,
    "sports, media & entertainment": 5,
    "education": 6,
    "government, defense & legal": 7,
    "travel, food & hospitality": 8,
    "non-profit": 9
}


def multi_rep_extract(task: str, mode: str, sents: List[str], labels1: List[str],
                      max_len: int, step: int, model_id: str, output_base: str):
    """
    Extract sentence-level embeddings from a causal LLM with device_map='auto'.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision='main',
        use_auth_token=None,
        output_hidden_states=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    model.eval()

    sents_reps = []
    for idx in trange(0, len(sents), step):
        batch_sents = sents[idx: idx + step]
        inputs = tokenizer(
            batch_sents,
            return_tensors="pt",
            max_length=max_len,
            padding="max_length",
            truncation=True
        ).to("cuda")

        with torch.no_grad():
            out = model(**inputs)
            reps = []
            for layer in range(-1, -6, -1):
                reps.append(torch.mean(out.hidden_states[layer], dim=1))
            reps = torch.stack(reps, dim=1)
        sents_reps.append(reps.cpu())

    sents_reps = torch.cat(sents_reps, dim=0)

    label_tensors = [torch.tensor(coarse_grain[l]) for l in labels1]
    labels = torch.stack(label_tensors)

    save_path = os.path.join(output_base, task, model_id.split("/")[-1], "dataset_tensor")
    os.makedirs(save_path, exist_ok=True)

    torch.save(sents_reps, os.path.join(save_path, f"{mode}_sents.pt"))
    torch.save(labels, os.path.join(save_path, f"{mode}_labels.pt"))

    print(f"[SAVED] {mode}_sents.pt | shape: {sents_reps.shape}")
    print(f"[SAVED] {mode}_labels.pt | shape: {labels.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Embedding extraction using LLaMA with device_map=auto")
    parser.add_argument("--model_id", type=str, required=True, help="Model checkpoint from Hugging Face (e.g., meta-llama/Meta-Llama-3-8B)")
    parser.add_argument("--task", type=str, default="model_embeddings", help="Name of task for saving files")
    parser.add_argument("--output_base", type=str, default="/data/soac/embeddings", help="Base path to save .pt outputs")
    parser.add_argument("--max_len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (step)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset("Shahriar/SoAC_Corpus")

    for split in ['train', 'validation', 'test']:
        print(f"\n=== Processing {split.upper()} ===")
        data = dataset[split]
        multi_rep_extract(
            task=args.task,
            mode=split,
            sents=data["Website_Summary"],
            labels1=data["Coarse_Grained_Sector_Label"],
            max_len=args.max_len,
            step=args.batch_size,
            model_id=args.model_id,
            output_base=args.output_base
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
