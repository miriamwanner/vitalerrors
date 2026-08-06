#!/usr/bin/env python3
"""
Compare subclaim importance rankings and labels across models and human annotators.

Each system is a JSONL file in the standard format (revisions/nq-data/*.jsonl or
model-outputs/<slug>/*.jsonl) with:
  - decomposition[*].decomp[*].importance   (vital / okay / less-important / null)
  - subclaim-importance-order               (ranked list of subclaim IDs)

Metrics computed for every pair of systems:
  • Cohen's weighted kappa  — ordinal label agreement (vital > okay > less-important)
  • % exact agreement       — fraction of subclaims with identical label
  • Mean Kendall's tau-b    — agreement on the full subclaim ordering

The pairwise tables are printed to stdout and optionally saved to a file.

Requirements:
    pip install numpy scipy scikit-learn

Usage:
    python compare_rankings.py \\
        --systems gpt-4o:../normal.jsonl \\
                  llama-3.3-70b:../model-outputs/meta-llama_Llama-3.3-70B-Instruct-Turbo/normal.jsonl \\
                  qwen-2.5-72b:../model-outputs/Qwen2.5-72B-Instruct/normal.jsonl \\
                  deepseek-r1:../model-outputs/DeepSeek-R1-Distill-Llama-70B/normal.jsonl \\
                  gpt-oss-120b:../model-outputs/gpt-oss-120b/normal.jsonl \\
                  human:../normal_human-annotated.jsonl \\
        [--output ../results/normal_agreement.txt] \\
        [--per-instance]
"""

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score


# ── Constants ──────────────────────────────────────────────────────────────────

LABELS = ["vital", "okay", "less-important"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}  # vital=0, okay=1, less-important=2


# ── Data loading ───────────────────────────────────────────────────────────────

def load_system(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_labels(instance: dict) -> dict[str, int]:
    """Returns {subclaim_id: label_int} for subclaims with a valid label."""
    out = {}
    for sent in instance["decomposition"]:
        for atom in sent["decomp"]:
            label = atom.get("importance")
            if label in LABEL_TO_INT:
                out[atom["id"]] = LABEL_TO_INT[label]
    return out


def get_order(instance: dict) -> list[str]:
    return instance.get("subclaim-importance-order", [])


# ── Metrics ────────────────────────────────────────────────────────────────────

def kendall_tau(order_a: list[str], order_b: list[str]) -> float | None:
    """
    Kendall's tau-b between two subclaim orderings.
    Only subclaims present in both lists are compared.
    """
    set_b = set(order_b)
    common = [s for s in order_a if s in set_b]
    if len(common) < 2:
        return None
    pos_a = {s: i for i, s in enumerate(order_a)}
    pos_b = {s: i for i, s in enumerate(order_b)}
    vec_a = [pos_a[s] for s in common]
    vec_b = [pos_b[s] for s in common]
    tau, _ = stats.kendalltau(vec_a, vec_b)
    return tau


def pairwise_label_metrics(
    instances_a: list[dict], instances_b: list[dict]
) -> dict:
    """
    Compute label agreement across all matched instances.
    Returns dict with kappa, pct_agree, n_subclaims, n_instances.
    Skips subclaims where either system has a null/missing label
    (e.g. unannotated human instances).
    """
    all_a, all_b = [], []
    n_instances = 0

    for inst_a, inst_b in zip(instances_a, instances_b):
        if inst_a["query"] != inst_b["query"]:
            print(f"  WARNING: query mismatch at index {n_instances}, skipping", file=sys.stderr)
            continue
        labels_a = get_labels(inst_a)
        labels_b = get_labels(inst_b)
        common = sorted(set(labels_a) & set(labels_b))
        for sid in common:
            all_a.append(labels_a[sid])
            all_b.append(labels_b[sid])
        n_instances += 1

    if len(all_a) < 2:
        return {"kappa": None, "pct_agree": None, "n_subclaims": len(all_a), "n_instances": n_instances}

    # Weighted kappa: linear weights over vital=0, okay=1, less-important=2
    # This penalises vital↔less-important disagreements more than adjacent ones.
    kappa = cohen_kappa_score(all_a, all_b, weights="linear",
                              labels=list(range(len(LABELS))))
    pct = sum(a == b for a, b in zip(all_a, all_b)) / len(all_a)
    return {"kappa": kappa, "pct_agree": pct, "n_subclaims": len(all_a), "n_instances": n_instances}


def pairwise_ranking_metrics(
    instances_a: list[dict], instances_b: list[dict]
) -> dict:
    """
    Mean Kendall's tau-b on subclaim-importance-order across all instances.
    Returns dict with mean_tau, std_tau, per_instance list, n_instances.
    """
    taus = []
    for inst_a, inst_b in zip(instances_a, instances_b):
        if inst_a["query"] != inst_b["query"]:
            continue
        tau = kendall_tau(get_order(inst_a), get_order(inst_b))
        if tau is not None:
            taus.append(tau)

    if not taus:
        return {"mean_tau": None, "std_tau": None, "per_instance": [], "n_instances": 0}
    return {
        "mean_tau": float(np.mean(taus)),
        "std_tau": float(np.std(taus)),
        "per_instance": taus,
        "n_instances": len(taus),
    }


def label_distribution(instances: list[dict]) -> dict[str, float]:
    """Fraction of subclaims per label, ignoring nulls."""
    counts = defaultdict(int)
    for inst in instances:
        for sent in inst["decomposition"]:
            for atom in sent["decomp"]:
                if atom.get("importance") in LABEL_TO_INT:
                    counts[atom["importance"]] += 1
    total = sum(counts.values())
    if total == 0:
        return {l: 0.0 for l in LABELS}
    return {l: counts[l] / total for l in LABELS}


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(v, precision=3) -> str:
    if v is None:
        return "  —   "
    return f"{v:+.{precision}f}" if abs(v) < 1 else f" {v:.{precision}f}"


def print_pairwise_table(title: str, names: list[str], matrix: list[list],
                         cell_fn, out) -> None:
    """Print a lower-triangular pairwise matrix."""
    w = max(len(n) for n in names)
    print(f"\n{'='*72}", file=out)
    print(f"  {title}", file=out)
    print(f"{'='*72}", file=out)

    # Header row
    header = " " * (w + 2)
    for n in names[1:]:
        header += f"  {n:>10}"
    print(header, file=out)

    for i in range(1, len(names)):
        row = f"{names[i]:<{w}}  "
        for j in range(i):
            row += f"  {cell_fn(matrix[i][j]):>10}"
        print(row, file=out)


def print_per_instance_table(names_pair: tuple[str, str], taus: list[float], out) -> None:
    a, b = names_pair
    print(f"\n  Per-instance Kendall's tau-b: {a} vs {b}", file=out)
    print(f"  {'idx':>4}  {'tau':>7}", file=out)
    for i, tau in enumerate(taus):
        marker = "  ←" if abs(tau) < 0.3 else ""
        print(f"  {i:>4}  {tau:>+7.3f}{marker}", file=out)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare importance rankings and labels across systems.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--systems", nargs="+", required=True, metavar="NAME:PATH",
        help="Systems to compare, each as name:path",
    )
    parser.add_argument(
        "--output", default=None,
        help="Also write results to this file",
    )
    parser.add_argument(
        "--per-instance", action="store_true",
        help="Print per-instance Kendall's tau-b for every pair",
    )
    args = parser.parse_args()

    # Parse name:path pairs
    systems: dict[str, list[dict]] = {}
    for spec in args.systems:
        if ":" not in spec:
            print(f"ERROR: expected name:path, got {spec!r} — use e.g. gpt-4o:../normal.jsonl",
                  file=sys.stderr)
            sys.exit(1)
        name, path = spec.split(":", 1)
        data = load_system(path)
        systems[name] = data
        n_labeled = sum(
            1 for inst in data
            for sent in inst["decomposition"]
            for atom in sent["decomp"]
            if atom.get("importance") in LABEL_TO_INT
        )
        print(f"  Loaded {len(data):>3} instances  ({n_labeled} labeled subclaims)  [{name}]")

    names = list(systems.keys())
    n = len(names)

    # Compute all pairwise metrics
    label_results  = [[None] * n for _ in range(n)]
    rank_results   = [[None] * n for _ in range(n)]

    print(f"\nComputing pairwise metrics for {n*(n-1)//2} pairs...")
    for i, j in combinations(range(n), 2):
        a, b = names[i], names[j]
        lr = pairwise_label_metrics(systems[a], systems[b])
        rr = pairwise_ranking_metrics(systems[a], systems[b])
        label_results[i][j] = label_results[j][i] = lr
        rank_results[i][j]  = rank_results[j][i]  = rr
        print(f"  {a:>20} vs {b:<20}  "
              f"κ={lr['kappa']:+.3f}  "
              f"%={lr['pct_agree']:.1%}  "
              f"τ={rr['mean_tau']:+.3f}±{rr['std_tau']:.3f}  "
              f"(n_sub={lr['n_subclaims']}, n_inst={lr['n_instances']})")

    # Choose output sink(s)
    sinks = [sys.stdout]
    outfile = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        outfile = open(out_path, "w")
        sinks.append(outfile)

    def write(text=""):
        for s in sinks:
            print(text, file=s)

    # ── Cohen's weighted kappa table ──
    write()
    write("=" * 72)
    write("  Cohen's weighted kappa  (linear weights, vital > okay > less-important)")
    write("  Interpretation:  < 0.20 slight  |  0.21-0.40 fair  |  0.41-0.60 moderate")
    write("                   0.61-0.80 substantial  |  > 0.80 almost perfect")
    write("=" * 72)
    w = max(len(n) for n in names)
    header = " " * (w + 2)
    for n_ in names[1:]:
        header += f"  {n_:>12}"
    write(header)
    for i in range(1, len(names)):
        row = f"{names[i]:<{w}}  "
        for j in range(i):
            v = label_results[i][j]
            cell = f"{v['kappa']:+.3f}" if v and v["kappa"] is not None else "  —   "
            row += f"  {cell:>12}"
        write(row)

    # ── Percent agreement table ──
    write()
    write("=" * 72)
    write("  Percent exact label agreement")
    write("=" * 72)
    header = " " * (w + 2)
    for n_ in names[1:]:
        header += f"  {n_:>12}"
    write(header)
    for i in range(1, len(names)):
        row = f"{names[i]:<{w}}  "
        for j in range(i):
            v = label_results[i][j]
            cell = f"{v['pct_agree']:.1%}" if v and v["pct_agree"] is not None else "  —   "
            row += f"  {cell:>12}"
        write(row)

    # ── Mean Kendall's tau-b table ──
    write()
    write("=" * 72)
    write("  Mean Kendall's tau-b  (subclaim ordering agreement)")
    write("  Format: mean ± std  over instances")
    write("=" * 72)
    header = " " * (w + 2)
    for n_ in names[1:]:
        header += f"  {n_:>16}"
    write(header)
    for i in range(1, len(names)):
        row = f"{names[i]:<{w}}  "
        for j in range(i):
            v = rank_results[i][j]
            if v and v["mean_tau"] is not None:
                cell = f"{v['mean_tau']:+.3f} ± {v['std_tau']:.3f}"
            else:
                cell = "  —   "
            row += f"  {cell:>16}"
        write(row)

    # ── Label distribution ──
    write()
    write("=" * 72)
    write("  Label distribution  (fraction of subclaims per category)")
    write("=" * 72)
    write(f"  {'system':<{w}}  {'vital':>8}  {'okay':>8}  {'less-imp':>8}  {'unlabeled':>10}")
    for name, data in systems.items():
        dist = label_distribution(data)
        total_sub = sum(len(s["decomp"]) for inst in data for s in inst["decomposition"])
        labeled = sum(
            1 for inst in data for sent in inst["decomposition"]
            for atom in sent["decomp"] if atom.get("importance") in LABEL_TO_INT
        )
        unlabeled_pct = (total_sub - labeled) / total_sub if total_sub else 0
        write(f"  {name:<{w}}  {dist['vital']:>8.1%}  {dist['okay']:>8.1%}  "
              f"{dist['less-important']:>8.1%}  {unlabeled_pct:>10.1%}")

    # ── Per-instance tau ──
    if args.per_instance:
        write()
        write("=" * 72)
        write("  Per-instance Kendall's tau-b")
        write("=" * 72)
        for i, j in combinations(range(len(names)), 2):
            v = rank_results[i][j]
            if v and v["per_instance"]:
                write(f"\n  {names[i]} vs {names[j]}:")
                write(f"  {'idx':>4}  {'tau':>7}")
                for idx, tau in enumerate(v["per_instance"]):
                    marker = "  ←" if abs(tau) < 0.3 else ""
                    write(f"  {idx:>4}  {tau:>+7.3f}{marker}")

    if outfile:
        outfile.close()
        print(f"\nResults also saved to {args.output}")


if __name__ == "__main__":
    main()
