"""Download HotpotQA (distractor/validation), sample 1000 queries, and write
original.jsonl split into two 500-query subsets (val_1, val_2).

Example:
    python -m get_data.get_hf_data.get_hotpotqa_data --out_dir data
"""
import argparse
import json
import os
import random

from datasets import load_dataset
from tqdm import tqdm

NUM_SAMPLES = 1000
SPLIT_AT = 500


def main():
    parser = argparse.ArgumentParser(description="Download and reformat HotpotQA.")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading data...")
    dataset = load_dataset("hotpotqa/hotpot_qa", name="distractor", split="validation")
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    sampled = dataset.select(indices)

    out_file_1 = os.path.join(args.out_dir, "hotpotqa", "val_1", "original.jsonl")
    out_file_2 = os.path.join(args.out_dir, "hotpotqa", "val_2", "original.jsonl")
    os.makedirs(os.path.dirname(out_file_1), exist_ok=True)
    os.makedirs(os.path.dirname(out_file_2), exist_ok=True)

    print(f"Writing to {out_file_1} and {out_file_2}...")
    with open(out_file_1, "w", encoding="utf-8") as f1, open(out_file_2, "w", encoding="utf-8") as f2:
        for i, (qid, question, gold_answer, docs) in enumerate(
            tqdm(zip(sampled["id"], sampled["question"], sampled["answer"], sampled["context"]))
        ):
            documents = [
                {"id": str(j), "text": title + "\n" + " ".join(sentences)}
                for j, (title, sentences) in enumerate(zip(docs["title"], docs["sentences"]))
            ]
            to_write = {"id": qid, "query": question, "gold-answer": gold_answer, "documents": documents}
            f = f1 if i < SPLIT_AT else f2
            f.write(json.dumps(to_write) + "\n")


if __name__ == "__main__":
    main()
