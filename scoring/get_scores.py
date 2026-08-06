"""Aggregate scores across all datasets/subsets/prompt conditions into one table
(FactScore + SAFE-style recall/F1, NuggetRecall, and VITAL precision/recall/F1).

Example:
    python -m scoring.get_scores --root_dir data
"""
import argparse
import json
import os
import statistics

import pandas as pd

from get_data.data_setup.data_info import datasets, subsets

PROMPTS = ["normal", "missing", "wrong"]


def get_factscore_scores(in_file):
    """FactScore precision plus a SAFE-style recall/F1 computed from the same decomposition."""
    factscores = []
    num_facts = []
    num_supported = []
    with open(in_file, "r") as file:
        for line in file:
            line = json.loads(line)
            if line["score"] is None:
                continue
            factscores.append(line["score"])
            passage_num_fact = 0
            passage_num_supported = 0
            for decomp_dict in line["decomposition"]:
                passage_num_fact += len(decomp_dict["decomp"])
                for atom in decomp_dict["decomp"]:
                    if atom["judgment"] is True:
                        passage_num_supported += 1
            num_facts.append(passage_num_fact)
            num_supported.append(passage_num_supported)

    avg_fs = sum(factscores) / len(factscores)
    avg_n_atom = sum(num_facts) / len(num_facts)
    median_num_atoms = statistics.median(num_facts)
    recall_scores = [min(n / median_num_atoms, 1) for n in num_supported]
    avg_safe_recall = sum(recall_scores) / len(recall_scores)
    safe_f1_scores = [(2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0 for prec, rec in zip(factscores, recall_scores)]
    avg_safe_f1 = sum(safe_f1_scores) / len(safe_f1_scores)
    return avg_fs, avg_n_atom, avg_safe_recall, median_num_atoms, avg_safe_f1


def get_nugget_scores(nuggets_file):
    """NUGGETRECALL ("All Strict"): fraction of nuggets fully supported by the response."""
    strict_all_scores = []
    num_nuggets = []
    with open(nuggets_file, "r") as file:
        for line in file:
            line = json.loads(line)
            strict_all_scores.append(line["response"]["strict-all-score"])
            num_nuggets.append(len(line["nuggets"]))
    return sum(strict_all_scores) / len(strict_all_scores), sum(num_nuggets) / len(num_nuggets)


def get_vital_metric_scores(new_metric_file):
    """VITAL precision/recall/F1 and the vital/okay/less-important precision breakdown."""
    fields = [
        "weighted-precision", "weighted-recall", "weighted-f1",
        "vital-precision", "vital-subclaims",
        "okay-precision", "okay-subclaims",
        "less-important-precision", "less-important-subclaims",
        "linear-decay-precision", "linear-decay-recall", "linear-decay-f1",
    ]
    totals = {field: [] for field in fields}
    with open(new_metric_file, "r") as file:
        for line in file:
            line = json.loads(line)
            scores = line["scores"]
            for field in fields:
                totals[field].append(scores[field])
    return {field: sum(values) / len(values) for field, values in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Aggregate scores across all datasets/subsets/prompts.")
    parser.add_argument("--root_dir", type=str, default="data")
    args = parser.parse_args()

    results = {
        "dataset": [], "subset": [], "prompt": [],
        "factscore": [], "num-subclaims": [], "safe-recall": [], "safe-k": [], "safe-f1": [],
        "nuggets-strict-all": [], "num-nuggets": [],
        "weighted-precision": [], "weighted-recall": [], "weighted-f1": [],
        "vital-precision": [], "num-vital-subclaims": [],
        "okay-precision": [], "num-okay-subclaims": [],
        "less-important-precision": [], "num-less-important-subclaims": [],
        "linear-decay-precision": [], "linear-decay-recall": [], "linear-decay-f1": [],
    }

    for d in datasets:
        for s in subsets[d]:
            for p in PROMPTS:
                factscore_file = os.path.join(args.root_dir, d, s, p, "factscore-out.jsonl")
                nuggets_file = os.path.join(args.root_dir, d, s, p, "nuggets-out.jsonl")
                new_metric_file = os.path.join(args.root_dir, d, s, p, "new-metric-out.jsonl")

                avg_fs, avg_n_atom, safe_recall, safe_k, safe_f1 = get_factscore_scores(factscore_file)
                nuggets_strict_all, avg_n_nuggets = get_nugget_scores(nuggets_file)
                vital_scores = get_vital_metric_scores(new_metric_file)

                results["dataset"].append(d)
                results["subset"].append(s)
                results["prompt"].append(p)
                results["factscore"].append(avg_fs)
                results["num-subclaims"].append(avg_n_atom)
                results["safe-recall"].append(safe_recall)
                results["safe-k"].append(safe_k)
                results["safe-f1"].append(safe_f1)
                results["nuggets-strict-all"].append(nuggets_strict_all)
                results["num-nuggets"].append(avg_n_nuggets)
                results["weighted-precision"].append(vital_scores["weighted-precision"])
                results["weighted-recall"].append(vital_scores["weighted-recall"])
                results["weighted-f1"].append(vital_scores["weighted-f1"])
                results["vital-precision"].append(vital_scores["vital-precision"])
                results["num-vital-subclaims"].append(vital_scores["vital-subclaims"])
                results["okay-precision"].append(vital_scores["okay-precision"])
                results["num-okay-subclaims"].append(vital_scores["okay-subclaims"])
                results["less-important-precision"].append(vital_scores["less-important-precision"])
                results["num-less-important-subclaims"].append(vital_scores["less-important-subclaims"])
                results["linear-decay-precision"].append(vital_scores["linear-decay-precision"])
                results["linear-decay-recall"].append(vital_scores["linear-decay-recall"])
                results["linear-decay-f1"].append(vital_scores["linear-decay-f1"])

    results_df = pd.DataFrame(results)
    print(results_df.to_string())


if __name__ == "__main__":
    main()
