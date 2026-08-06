"""Download Natural Questions (dev/validation), sample 1000 queries, strip HTML
from the source document, and write original.jsonl split into two 500-query
subsets (val_1, val_2).

Example:
    python -m get_data.get_hf_data.get_naturalquestions_data --out_dir data
"""
import argparse
import json
import os
import random

from bs4 import BeautifulSoup
from datasets import load_dataset
from tqdm import tqdm

NUM_SAMPLES = 1000
SPLIT_AT = 500


def clean_html(html: str) -> str:
    """Remove HTML tags, scripts, and styles, returning only visible text."""
    soup = BeautifulSoup(html, "html.parser")
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    return " ".join(soup.get_text(separator=" ").split())


def main():
    parser = argparse.ArgumentParser(description="Download and reformat Natural Questions.")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading data...")
    dataset = load_dataset("natural_questions", "dev", split="validation")
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    sampled = dataset.select(indices)

    out_file_1 = os.path.join(args.out_dir, "naturalquestions", "val_1", "original.jsonl")
    out_file_2 = os.path.join(args.out_dir, "naturalquestions", "val_2", "original.jsonl")
    os.makedirs(os.path.dirname(out_file_1), exist_ok=True)
    os.makedirs(os.path.dirname(out_file_2), exist_ok=True)

    print(f"Writing to {out_file_1} and {out_file_2}...")
    with open(out_file_1, "w", encoding="utf-8") as f1, open(out_file_2, "w", encoding="utf-8") as f2:
        for i, (qid, question, docs) in enumerate(tqdm(zip(sampled["id"], sampled["question"], sampled["document"]))):
            documents = [{"id": "0", "text": clean_html(docs["html"])}]
            to_write = {"id": qid, "query": question["text"], "documents": documents}
            f = f1 if i < SPLIT_AT else f2
            f.write(json.dumps(to_write) + "\n")


if __name__ == "__main__":
    main()
