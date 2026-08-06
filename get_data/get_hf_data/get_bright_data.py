"""Download BRIGHT (xlangai/BRIGHT) and write original.jsonl for each subset.

Example:
    python -m get_data.get_hf_data.get_bright_data --out_dir data
"""
import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

SUBSETS = ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"]


def main():
    parser = argparse.ArgumentParser(description="Download and reformat BRIGHT.")
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    print("Loading data...")
    examples = load_dataset("xlangai/BRIGHT", "examples")
    documents = load_dataset("xlangai/BRIGHT", "documents")

    for subset in SUBSETS:
        out_file = os.path.join(args.out_dir, "bright", subset, "original.jsonl")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        print(f"Getting {subset} subset...")
        subset_examples = examples[subset]
        subset_documents = documents[subset]

        print(f"Writing to {out_file}...")
        with open(out_file, "w", encoding="utf-8") as f:
            for example in tqdm(subset_examples):
                query = example["query"]
                question_label = example["gold_ids"][0].split("/")[0]

                docs_sufficient = []
                for doc in subset_documents:
                    if doc["id"].split("/")[0] == question_label and doc["id"] in example["gold_ids"]:
                        docs_sufficient.append({"id": doc["id"], "text": doc["content"]})

                to_write = {
                    "query": query,
                    "gold_response": {"id": example["id"], "text": example["gold_answer"]},
                    "response": {"id": example["id"], "text": None},
                    "documents": docs_sufficient,
                }
                f.write(json.dumps(to_write) + "\n")


if __name__ == "__main__":
    main()
