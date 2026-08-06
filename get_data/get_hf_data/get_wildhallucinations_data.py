"""Download WildHallucinations (wentingzhao/WildHallucinations), filter to the
culture & entertainment and geographic categories, and write original.jsonl.
The culture & entertainment category is split into 4 subsets of 500 entities
each (cult_ent_1..4); geographic is written as a single subset.

Example:
    python -m get_data.get_hf_data.get_wildhallucinations_data --out_dir data
"""
import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

CULT_ENT_SPLIT_SIZE = 500


def main():
    parser = argparse.ArgumentParser(description="Download and reformat WildHallucinations.")
    parser.add_argument("--out_dir", type=str, default="data")
    args = parser.parse_args()

    print("Loading data...")
    dataset = load_dataset("wentingzhao/WildHallucinations", split="train")

    print("Filtering data...")
    dataset_cult_ent = dataset.filter(lambda x: x["category"] == "culture & entertainment")
    dataset_geographic = dataset.filter(lambda x: x["category"] == "geographic")

    cult_ent_out_files = [
        os.path.join(args.out_dir, "wildhallucinations", f"cult_ent_{i + 1}", "original.jsonl") for i in range(4)
    ]
    for f in cult_ent_out_files:
        os.makedirs(os.path.dirname(f), exist_ok=True)

    print(f"Writing {len(cult_ent_out_files)} culture & entertainment subsets...")
    with open(cult_ent_out_files[0], "w", encoding="utf-8") as f1, \
         open(cult_ent_out_files[1], "w", encoding="utf-8") as f2, \
         open(cult_ent_out_files[2], "w", encoding="utf-8") as f3, \
         open(cult_ent_out_files[3], "w", encoding="utf-8") as f4:
        writers = [f1, f2, f3, f4]
        for i, (topic, info, wiki) in enumerate(
            tqdm(zip(dataset_cult_ent["entity"], dataset_cult_ent["info"], dataset_cult_ent["wiki"]))
        ):
            to_write = {"query": topic, "documents": info, "wiki": wiki}
            writers[min(i // CULT_ENT_SPLIT_SIZE, 3)].write(json.dumps(to_write) + "\n")

    geographic_out_file = os.path.join(args.out_dir, "wildhallucinations", "geographic", "original.jsonl")
    os.makedirs(os.path.dirname(geographic_out_file), exist_ok=True)
    print(f"Writing to {geographic_out_file}...")
    with open(geographic_out_file, "w", encoding="utf-8") as f:
        for topic, info, wiki in tqdm(zip(dataset_geographic["entity"], dataset_geographic["info"], dataset_geographic["wiki"])):
            to_write = {"query": topic, "documents": info, "wiki": wiki}
            f.write(json.dumps(to_write) + "\n")


if __name__ == "__main__":
    main()
