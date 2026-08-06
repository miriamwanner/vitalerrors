"""Download the FActScore bios dataset (dskar/FActScore) and write original.jsonl.

Example:
    python -m get_data.get_hf_data.get_fs_data --out_dir data
"""
import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download and reformat the FActScore bios dataset.")
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    out_file = os.path.join(args.out_dir, "factscore", "bios", "original.jsonl")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    print("Loading data...")
    dataset = load_dataset("dskar/FActScore", split="test")

    print(f"Writing to {out_file}...")
    with open(out_file, "w", encoding="utf-8") as f:
        for i, (topic, wiki_doc) in enumerate(tqdm(zip(dataset["entity"], dataset["wikipedia_text"]))):
            to_write = {
                "query": topic,
                "response": {"id": str(i), "text": None},
                "documents": [{"id": "0", "text": wiki_doc}],
            }
            f.write(json.dumps(to_write) + "\n")


if __name__ == "__main__":
    main()
