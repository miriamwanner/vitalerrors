import argparse
import json
import os
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def get_safe_scores(in_file):
    factscores = []
    num_facts = []
    num_supported = []
    with open(in_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            if line["score"]:
                factscores.append(line["score"])
                passage_num_fact = 0
                passage_num_supported = 0
                for decomp_dict in line["decomposition"]:
                    passage_num_fact += len(decomp_dict["decomp"])
                    for atom in decomp_dict["decomp"]:
                        if atom["judgment"] == True:
                            passage_num_supported += 1
                num_facts.append(passage_num_fact)
                num_supported.append(passage_num_supported)

    avg_fs = sum(factscores) / len(factscores)
    median_num_atoms = statistics.median(num_facts)
    recall_scores = [min(a / median_num_atoms, 1) for a in num_supported]
    avg_safe_recall = sum(recall_scores) / len(recall_scores)
    safe_f1_scores = [(2 * prec * rec) / (prec + rec) for prec, rec in zip(factscores, recall_scores)]
    avg_safe_f1 = sum(safe_f1_scores) / len(safe_f1_scores)
    return avg_fs, avg_safe_recall, median_num_atoms, avg_safe_f1


def get_nugget_scores(nuggets_file):
    nuggets_vital = []
    any_vital_nuggets_unsupported = []
    with open(nuggets_file, 'r') as f:
        for line in f:
            line = json.loads(line)
            vital_total = 0
            vital_support = 0
            for support, n in zip(line["response"]["nugget-assignment"], line["nuggets"]):
                if n["importance"] == "vital":
                    vital_total += 1
                    if support != "not_support":
                        vital_support += 1
            if vital_total == 0:
                nuggets_vital_recall = 0
            else:
                nuggets_vital_recall = vital_support / vital_total
            nuggets_vital.append(nuggets_vital_recall)
            any_vital_nuggets_unsupported.append(0 if nuggets_vital_recall >= 1 else 1)

    avg_vital_recall = sum(nuggets_vital) / len(nuggets_vital)
    avg_any_vital_nuggets_unsupported = sum(any_vital_nuggets_unsupported) / len(any_vital_nuggets_unsupported)
    return avg_vital_recall, avg_any_vital_nuggets_unsupported


def get_new_metric_scores(new_metric_file):
    vital_prec = []
    any_vital_wrong = []
    with open(new_metric_file, 'r') as f:
        for line in f:
            scores = json.loads(line)["scores"]
            vital_prec.append(scores["vital-precision"])
            any_vital_wrong.append(0 if scores["vital-precision"] >= 1 else 1)

    avg_vital_prec = sum(vital_prec) / len(vital_prec)
    avg_any_vital_wrong = sum(any_vital_wrong) / len(any_vital_wrong)
    return avg_vital_prec, avg_any_vital_wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    args = parser.parse_args()

    datasets = ["hotpotqa", "naturalquestions", "triviaqa", "bright", "wildhallucinations", "factscore"]
    subsets = {
        "bright": ["biology", "earth_science", "economics", "psychology", "robotics", "stackoverflow", "sustainable_living"],
        "wildhallucinations": ["cult_ent_1", "cult_ent_2", "cult_ent_3", "cult_ent_4", "geographic"],
        "factscore": ["bios"],
        "hotpotqa": ["val_1", "val_2"],
        "naturalquestions": ["val_1", "val_2"],
        "triviaqa": ["rc_1", "rc_2"],
    }
    prompts = ["normal", "missing", "wrong"]

    rows = []
    for d in datasets:
        for s in subsets[d]:
            for p in prompts:
                fs_file = os.path.join(args.root_dir, d, s, p, "factscore-out.jsonl")
                nug_file = os.path.join(args.root_dir, d, s, p, "nuggets-out.jsonl")
                new_file = os.path.join(args.root_dir, d, s, p, "importance-rank-new-metric-out.jsonl")

                missing = [f for f in [fs_file, nug_file, new_file] if not os.path.exists(f)]
                if missing:
                    for f in missing:
                        print(f"Missing: {f}")
                    continue

                avg_fs, avg_safe_recall, safe_k, avg_safe_f1 = get_safe_scores(fs_file)
                avg_vital_recall, avg_any_vital_nug_unsup = get_nugget_scores(nug_file)
                avg_vital_prec, avg_any_vital_wrong = get_new_metric_scores(new_file)

                rows.append({
                    "dataset": d,
                    "subset": s,
                    "prompt": p,
                    "factscore": avg_fs,
                    "safe-recall": avg_safe_recall,
                    "safe-k": safe_k,
                    "safe-f1": avg_safe_f1,
                    "vital-precision": avg_vital_prec,
                    "vital-recall": avg_vital_recall,
                    "any-vital-wrong": avg_any_vital_wrong,
                    "any-vital-nuggets-unsupported": avg_any_vital_nug_unsup,
                })

    all_cols = ["factscore", "safe-recall", "safe-k", "safe-f1",
                "vital-precision", "vital-recall", "any-vital-wrong", "any-vital-nuggets-unsupported"]
    df = pd.DataFrame(rows)
    # Average across subsets and datasets exactly as in final_presentation.py (groupby prompt only)
    grouped = df.groupby("prompt")[all_cols].mean().reset_index()
    grouped[all_cols] = grouped[all_cols].round(4)
    print(grouped.to_string(index=False))

    # --- shared style (matches final_presentation.py) ---
    colors = {"normal": "#FFCC00", "missing": "#00A9E0", "wrong": "#3D5B99"}
    width = 0.25  # bars touch: each bar spans ±0.125, next center is ±0.25 away
    out_dir = os.path.dirname(__file__)

    def make_figure(metrics, out_path, figsize=(8, 5), safe_k_col=None):
        """One axes, metrics on x-axis, normal/missing/wrong as touching bar groups.
        If safe_k_col is given, plots that column on a secondary right y-axis."""
        x = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=figsize)

        for prompt, offset in zip(prompts, [-width, 0, width]):
            vals = [
                grouped.loc[grouped["prompt"] == prompt, col].values[0] * 100
                for col, _ in metrics
            ]
            ax.bar(x + offset, vals, width, label=prompt.capitalize(),
                   color=colors[prompt], alpha=0.9)

        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in metrics], rotation=20, ha="right", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))

        if safe_k_col is not None:
            ax2 = ax.twinx()
            safe_k_vals = [
                grouped.loc[grouped["prompt"] == prompt, safe_k_col].values[0]
                for prompt in prompts
            ]
            for prompt, offset, val in zip(prompts, [-width, 0, width], safe_k_vals):
                ax2.bar(len(metrics) + offset, val, width, color=colors[prompt], alpha=0.9)
            # push the top of ax2 to 3× the max value so bars sit in the lower third
            ax2.set_ylim(0, max(safe_k_vals) * 3)
            ax2.set_ylabel("SAFE-K (median # subclaims)", fontsize=9)
            # dashed separator between SAFE F1 and SAFE-K
            ax.axvline(x=len(metrics) - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
            # extend x-axis to include the safe-k group
            ax.set_xlim(-0.5, len(metrics) + 0.5)
            ax.set_xticks(list(x) + [len(metrics)])
            ax.set_xticklabels(
                [label for _, label in metrics] + ["SAFE-K"],
                rotation=20, ha="right", fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(out_path, format="pdf")
        print(f"Figure saved to {out_path}")

    make_figure(
        [
            ("factscore",   "SAFE Precision"),
            ("safe-recall", "SAFE Recall"),
            ("safe-f1",     "SAFE F1"),
        ],
        out_path=os.path.join(out_dir, "safe_scores.pdf"),
        safe_k_col="safe-k",
    )

    make_figure(
        [
            ("vital-precision",               "Vital Precision"),
            ("vital-recall",                  "Vital Recall"),
            ("any-vital-wrong",               "Any vital subclaim wrong"),
            ("any-vital-nuggets-unsupported", "Any vital nugget unsupported"),
        ],
        out_path=os.path.join(out_dir, "vital_scores.pdf"),
        figsize=(9, 5),
    )


if __name__ == "__main__":
    main()
