#!/usr/bin/env python3
"""
Significance testing + effect sizes for the prompt-condition comparisons
(normal / missing / wrong) shown in final_presentation.py.

For each metric used in the precision/recall/error figures, and for each
pairwise prompt comparison (normal vs missing, normal vs wrong,
missing vs wrong), this script:

  - loads PER-EXAMPLE scores (not the pre-averaged numbers final_presentation.py
    plots), aligned by query index within each dataset/subset, since the same
    500-ish queries are reused across the three prompt conditions
  - pools per-example paired differences across subsets within each dataset
    group (Open-Ended: factscore/wildhallucinations/bright,
    Single-Answer: hotpotqa/naturalquestions/triviaqa), and also reports an
    "All" pooled group, matching the grouping used in the bar charts
  - runs a paired t-test and a Wilcoxon signed-rank test for continuous
    metrics (report both since the metrics are bounded in [0, 1] and not
    always normal), with Cohen's d_z / matched-pairs rank-biserial r as
    effect sizes
  - runs McNemar's test for the two binary error metrics (any-vital-wrong,
    any-vital-nuggets-unsupported), with the risk difference as effect size
  - applies Holm-Bonferroni correction within each group across all
    metric x comparison tests

Reported descriptive means/rates (mean_A, mean_B, rate_A, rate_B) are the
unweighted average of each subset's per-example mean, matching the
aggregation in analysis/precision_recall_table.py and
analysis/get_counts_error_tables.py (pandas groupby('prompt').mean() over an
already per-subset-averaged dataframe) -- this is what produced the numbers
in the paper's Tables 2-4, and every value below has been checked to
reproduce those tables exactly. The significance tests themselves still run
on the fully pooled per-example pairs (every subset's examples concatenated
together) since that's the valid, maximum-power way to run a paired test --
so mean_diff/std_diff/Cohen's d_z (computed from the pooled differences) can
differ slightly from mean_A - mean_B (the unweighted subset-of-subsets
difference).

nuggets-strict-all/nuggets-vital-recall use STRICT nugget-support credit
(only "support" counts; "partial_support" does not), matching
precision_recall_table.py exactly. any-vital-nuggets-unsupported uses LENIENT
credit ("partial_support" counts as supported), matching
get_counts_error_tables.py exactly -- the two source scripts disagree on
this on purpose (continuous recall metrics are strict; the binary error flag
is lenient), so the two must NOT share one code path.

Outputs:
  results/stat_sig_continuous.csv
  results/stat_sig_binary.csv
  results/summary.md
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

DATASETS = ["hotpotqa", "naturalquestions", "triviaqa", "bright", "wildhallucinations", "factscore"]
SUBSETS = {
    "bright": ["biology", "earth_science", "economics", "psychology", "robotics",
               "stackoverflow", "sustainable_living"],
    "wildhallucinations": ["cult_ent_1", "cult_ent_2", "cult_ent_3", "cult_ent_4", "geographic"],
    "factscore": ["bios"],
    "hotpotqa": ["val_1", "val_2"],
    "naturalquestions": ["val_1", "val_2"],
    "triviaqa": ["rc_1", "rc_2"],
}
PROMPTS = ["normal", "missing", "wrong"]
PROMPT_PAIRS = [("normal", "missing"), ("normal", "wrong"), ("missing", "wrong")]

GROUPS = {
    "Open-Ended": ["factscore", "wildhallucinations", "bright"],
    "Single-Answer": ["hotpotqa", "naturalquestions", "triviaqa"],
    "All": DATASETS,
}

CONTINUOUS_METRICS = [
    ("factscore", "Factscore (precision)"),
    ("vital-precision", "Vital Precision"),
    ("linear-decay-precision", "Linear-Decay Precision"),
    ("nuggets-strict-all", "Nuggets Recall (strict-all)"),
    ("nuggets-vital-recall", "Vital Recall"),
    ("linear-decay-recall", "Linear-Decay Recall"),
]
BINARY_METRICS = [
    ("any-vital-wrong", "Any vital subclaim wrong"),
    ("any-vital-nuggets-unsupported", "Any vital nugget unsupported"),
]


def load_subset_scores(root_dir, dataset, subset, prompt):
    """Return a dict metric -> np.array of per-example scores (np.nan where
    a metric could not be computed for that example, e.g. factscore is null
    when the response had no decomposable content)."""
    fs_file = os.path.join(root_dir, dataset, subset, prompt, "factscore-out.jsonl")
    nug_file = os.path.join(root_dir, dataset, subset, prompt, "nuggets-out.jsonl")
    new_file = os.path.join(root_dir, dataset, subset, prompt, "importance-rank-new-metric-out.jsonl")

    factscore = []
    with open(fs_file) as f:
        for line in f:
            line = json.loads(line)
            factscore.append(np.nan if line["score"] is None else line["score"])

    nuggets_strict_all, nuggets_vital_recall, any_vital_nuggets_unsupported = [], [], []
    with open(nug_file) as f:
        for line in f:
            line = json.loads(line)
            vital_total = vital_support_strict = vital_support_lenient = 0
            okay_total = okay_support_strict = 0
            for support, n in zip(line["response"]["nugget-assignment"], line["nuggets"]):
                if n["importance"] == "vital":
                    vital_total += 1
                    # "strict" credit (only full support) feeds the continuous
                    # recall metrics, matching precision_recall_table.py's
                    # get_nugget_scores (the code that produced Table 3).
                    if support == "support":
                        vital_support_strict += 1
                    # "lenient" credit (partial support also counts) feeds
                    # only the binary any-vital-nuggets-unsupported flag,
                    # matching get_counts_error_tables.py (Table 4).
                    if support != "not_support":
                        vital_support_lenient += 1
                elif n["importance"] == "okay":
                    okay_total += 1
                    if support == "support":
                        okay_support_strict += 1
            vital_recall_strict = (vital_support_strict / vital_total) if vital_total > 0 else 0.0
            vital_recall_lenient = (vital_support_lenient / vital_total) if vital_total > 0 else 0.0
            strict_all = ((vital_support_strict + okay_support_strict) / len(line["nuggets"])) if line["nuggets"] else 0.0
            nuggets_strict_all.append(strict_all)
            nuggets_vital_recall.append(vital_recall_strict)
            any_vital_nuggets_unsupported.append(0 if vital_recall_lenient >= 1 else 1)

    vital_precision, linear_decay_precision, linear_decay_recall, any_vital_wrong = [], [], [], []
    with open(new_file) as f:
        for line in f:
            scores = json.loads(line)["scores"]
            vital_precision.append(scores["vital-precision"])
            linear_decay_precision.append(scores["linear-decay-precision"])
            linear_decay_recall.append(scores["linear-decay-recall"])
            any_vital_wrong.append(0 if scores["vital-precision"] >= 1 else 1)

    return {
        "factscore": np.array(factscore, dtype=float),
        "vital-precision": np.array(vital_precision, dtype=float),
        "linear-decay-precision": np.array(linear_decay_precision, dtype=float),
        "nuggets-strict-all": np.array(nuggets_strict_all, dtype=float),
        "nuggets-vital-recall": np.array(nuggets_vital_recall, dtype=float),
        "linear-decay-recall": np.array(linear_decay_recall, dtype=float),
        "any-vital-wrong": np.array(any_vital_wrong, dtype=float),
        "any-vital-nuggets-unsupported": np.array(any_vital_nuggets_unsupported, dtype=float),
    }


def load_all_scores(root_dir):
    """scores[dataset][subset][prompt][metric] = np.array"""
    scores = {}
    for d in DATASETS:
        scores[d] = {}
        for s in SUBSETS[d]:
            scores[d][s] = {}
            for p in PROMPTS:
                scores[d][s][p] = load_subset_scores(root_dir, d, s, p)
    return scores


def pooled_group_vectors(scores, group_datasets, prompt, metric):
    """Concatenate per-example vectors for `metric`/`prompt` across all
    subsets of all datasets in `group_datasets`, preserving example order
    (and thus pairing across prompts, since normal/missing/wrong share the
    same query order within each subset)."""
    chunks = []
    for d in group_datasets:
        for s in SUBSETS[d]:
            chunks.append(scores[d][s][prompt][metric])
    return np.concatenate(chunks)


def subset_mean_of_means(scores, group_datasets, prompt, metric):
    """Unweighted average of each subset's per-example mean for `metric`/
    `prompt`, i.e. every subset (val_1, cult_ent_2, bright/biology, ...)
    contributes equally regardless of its example count. This matches the
    aggregation in analysis/precision_recall_table.py and
    analysis/get_counts_error_tables.py (pandas `groupby('prompt').mean()`
    over an already per-subset-averaged dataframe), which is what produced
    the numbers in the paper's Tables 2-4. Used only for the reported
    descriptive means/rates below -- the significance tests themselves run
    on the fully pooled per-example pairs from `pooled_group_vectors`."""
    subset_means = []
    for d in group_datasets:
        for s in SUBSETS[d]:
            arr = scores[d][s][prompt][metric]
            valid = arr[~np.isnan(arr)]
            if len(valid) > 0:
                subset_means.append(valid.mean())
    return float(np.mean(subset_means))


def cohens_dz(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def wilcoxon_rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation effect size, derived from the
    Wilcoxon signed-rank statistic (r = 1 - 2*W_minus/(W_plus+W_minus))."""
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(diff))
    r_plus = ranks[diff > 0].sum()
    r_minus = ranks[diff < 0].sum()
    total = r_plus + r_minus
    return float((r_plus - r_minus) / total) if total > 0 else float("nan")


def run_continuous_tests(scores):
    rows = []
    for group_name, group_datasets in GROUPS.items():
        for metric, metric_label in CONTINUOUS_METRICS:
            for p1, p2 in PROMPT_PAIRS:
                a = pooled_group_vectors(scores, group_datasets, p1, metric)
                b = pooled_group_vectors(scores, group_datasets, p2, metric)
                valid = ~(np.isnan(a) | np.isnan(b))
                a, b = a[valid], b[valid]
                n = len(a)
                diff = a - b

                t_stat, t_p = stats.ttest_rel(a, b)
                dz = cohens_dz(diff)

                if np.all(diff == 0):
                    w_stat, w_p, r_rb = float("nan"), 1.0, 0.0
                else:
                    w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox")
                    r_rb = wilcoxon_rank_biserial(a, b)

                # Reported means match the paper's Table 2-4 aggregation
                # (unweighted average of per-subset means). "mean_diff",
                # "std_diff" and Cohen's d_z below intentionally stay on the
                # pooled per-example differences -- that's the paired
                # quantity the significance test itself is computed from,
                # and the two can differ slightly from mean_p1 - mean_p2.
                mean_p1 = subset_mean_of_means(scores, group_datasets, p1, metric)
                mean_p2 = subset_mean_of_means(scores, group_datasets, p2, metric)

                rows.append({
                    "group": group_name,
                    "metric": metric,
                    "metric_label": metric_label,
                    "comparison": f"{p1} vs {p2}",
                    "n": n,
                    f"mean_{p1}": mean_p1,
                    f"mean_{p2}": mean_p2,
                    "mean_diff": diff.mean(),
                    "std_diff": diff.std(ddof=1),
                    "t_stat": t_stat,
                    "t_pvalue": t_p,
                    "cohens_dz": dz,
                    "wilcoxon_stat": w_stat,
                    "wilcoxon_pvalue": w_p,
                    "rank_biserial_r": r_rb,
                })
    df = pd.DataFrame(rows)

    # Holm-Bonferroni correction within each group, separately for each test type
    for group_name in GROUPS:
        mask = df["group"] == group_name
        df.loc[mask, "t_pvalue_holm"] = multipletests(df.loc[mask, "t_pvalue"], method="holm")[1]
        df.loc[mask, "wilcoxon_pvalue_holm"] = multipletests(df.loc[mask, "wilcoxon_pvalue"], method="holm")[1]
    return df


def run_binary_tests(scores):
    rows = []
    for group_name, group_datasets in GROUPS.items():
        for metric, metric_label in BINARY_METRICS:
            for p1, p2 in PROMPT_PAIRS:
                a = pooled_group_vectors(scores, group_datasets, p1, metric)
                b = pooled_group_vectors(scores, group_datasets, p2, metric)
                n = len(a)

                # 2x2 contingency table of paired binary outcomes
                both1 = int(np.sum((a == 1) & (b == 1)))
                a1_b0 = int(np.sum((a == 1) & (b == 0)))
                a0_b1 = int(np.sum((a == 0) & (b == 1)))
                both0 = int(np.sum((a == 0) & (b == 0)))
                table = [[both1, a1_b0], [a0_b1, both0]]

                exact = (a1_b0 + a0_b1) < 25
                result = mcnemar(table, exact=exact, correction=not exact)

                # Reported rates match the paper's Table 4 aggregation
                # (unweighted average of per-subset rates); the McNemar
                # test itself still runs on the pooled per-example pairs.
                rate_p1 = subset_mean_of_means(scores, group_datasets, p1, metric)
                rate_p2 = subset_mean_of_means(scores, group_datasets, p2, metric)
                risk_diff = rate_p1 - rate_p2

                rows.append({
                    "group": group_name,
                    "metric": metric,
                    "metric_label": metric_label,
                    "comparison": f"{p1} vs {p2}",
                    "n": n,
                    f"rate_{p1}": rate_p1,
                    f"rate_{p2}": rate_p2,
                    "risk_diff": risk_diff,
                    "discordant_10": a1_b0,
                    "discordant_01": a0_b1,
                    "mcnemar_stat": result.statistic,
                    "mcnemar_pvalue": result.pvalue,
                    "exact_test": exact,
                })
    df = pd.DataFrame(rows)
    for group_name in GROUPS:
        mask = df["group"] == group_name
        df.loc[mask, "mcnemar_pvalue_holm"] = multipletests(df.loc[mask, "mcnemar_pvalue"], method="holm")[1]
    return df


def sig_stars(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def write_summary_md(cont_df, bin_df, out_path):
    lines = []
    lines.append("# Significance Testing: Prompt Conditions (normal / missing / wrong)\n")
    lines.append(
        "Per-example scores are paired within each dataset/subset (the same "
        "queries are reused across normal/missing/wrong), then pooled across "
        "subsets within each dataset group before testing. p-values are "
        "Holm-Bonferroni corrected within each group across all metric x "
        "comparison tests of that type (continuous or binary). "
        "`*` p<.05, `**` p<.01, `***` p<.001 (corrected).\n"
    )

    for group_name in GROUPS:
        lines.append(f"\n## {group_name}\n")
        lines.append("### Continuous metrics (paired t-test / Wilcoxon signed-rank)\n")
        lines.append(
            "| Metric | Comparison | n | Mean A | Mean B | Mean diff | "
            "Cohen's d_z | t p (Holm) | Wilcoxon p (Holm) | Rank-biserial r |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        sub = cont_df[cont_df["group"] == group_name]
        for _, row in sub.iterrows():
            p1, p2 = row["comparison"].split(" vs ")
            lines.append(
                f"| {row['metric_label']} | {row['comparison']} | {row['n']} | "
                f"{row[f'mean_{p1}']:.4f} | {row[f'mean_{p2}']:.4f} | "
                f"{row['mean_diff']:+.4f} | {row['cohens_dz']:+.3f} | "
                f"{row['t_pvalue_holm']:.2e}{sig_stars(row['t_pvalue_holm'])} | "
                f"{row['wilcoxon_pvalue_holm']:.2e}{sig_stars(row['wilcoxon_pvalue_holm'])} | "
                f"{row['rank_biserial_r']:+.3f} |"
            )

        lines.append("\n### Binary error metrics (McNemar's test)\n")
        lines.append("| Metric | Comparison | n | Rate A | Rate B | Risk diff | Discordant (1,0)/(0,1) | McNemar p (Holm) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        sub = bin_df[bin_df["group"] == group_name]
        for _, row in sub.iterrows():
            p1, p2 = row["comparison"].split(" vs ")
            lines.append(
                f"| {row['metric_label']} | {row['comparison']} | {row['n']} | "
                f"{row[f'rate_{p1}']:.4f} | {row[f'rate_{p2}']:.4f} | "
                f"{row['risk_diff']:+.4f} | {row['discordant_10']}/{row['discordant_01']} | "
                f"{row['mcnemar_pvalue_holm']:.2e}{sig_stars(row['mcnemar_pvalue_holm'])} |"
            )
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Significance tests for prompt-condition comparisons")
    parser.add_argument(
        "--root_dir", type=str,
        default="/home/mwanner5/scratchmdredze1/mwanner5/ARCHIVE/vitalerrors/data-less-alt",
    )
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading per-example scores...")
    scores = load_all_scores(args.root_dir)

    print("Running continuous-metric tests (paired t-test + Wilcoxon)...")
    cont_df = run_continuous_tests(scores)
    cont_df.to_csv(os.path.join(out_dir, "stat_sig_continuous.csv"), index=False)

    print("Running binary-metric tests (McNemar)...")
    bin_df = run_binary_tests(scores)
    bin_df.to_csv(os.path.join(out_dir, "stat_sig_binary.csv"), index=False)

    write_summary_md(cont_df, bin_df, os.path.join(out_dir, "summary.md"))

    print(f"Wrote {out_dir}/stat_sig_continuous.csv")
    print(f"Wrote {out_dir}/stat_sig_binary.csv")
    print(f"Wrote {out_dir}/summary.md")


if __name__ == "__main__":
    main()
