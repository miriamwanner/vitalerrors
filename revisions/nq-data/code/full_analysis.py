#!/usr/bin/env python3
"""
Full pairwise agreement analysis across conditions (normal / missing / wrong).

Metrics per pair:
  • Cohen's weighted kappa       — ordinal label agreement
  • % exact label agreement
  • Mean Kendall's τ-b (full)    — agreement on the complete subclaim ordering
  • Mean Kendall's τ-b (within-label) — per label, agreement on relative ordering
    among subclaims that *both* systems assigned to that label

Outputs a single Markdown file.
"""

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

# ── Constants ─────────────────────────────────────────────────────────────────

LABELS      = ["vital", "okay", "less-important"]
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
CONDITIONS  = ["normal", "missing", "wrong"]

BASE = Path(__file__).parent.parent / "model-outputs"
SYSTEMS = {
    "gpt-4o":        BASE / "gpt-4o",
    "llama-3.3-70b": BASE / "meta-llama_Llama-3.3-70B-Instruct-Turbo",
    "qwen3-235b":    BASE / "Qwen_Qwen3-235B-A22B-Instruct-2507-tput",
    "gpt-oss-120b":  BASE / "openai_gpt-oss-120b",
    "human-eval":    BASE / "human-eval",
}

# ── Data helpers ──────────────────────────────────────────────────────────────

def load(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def get_labels(inst: dict) -> dict[str, int]:
    out = {}
    for sent in inst["decomposition"]:
        for atom in sent["decomp"]:
            lbl = atom.get("importance")
            if lbl in LABEL_TO_INT:
                out[atom["id"]] = LABEL_TO_INT[lbl]
    return out

def get_order(inst: dict) -> list[str]:
    return inst.get("subclaim-importance-order", [])

# ── Metrics ───────────────────────────────────────────────────────────────────

def kendall_tau(order_a, order_b) -> float | None:
    set_b = set(order_b)
    common = [s for s in order_a if s in set_b]
    if len(common) < 2:
        return None
    pos_a = {s: i for i, s in enumerate(order_a)}
    pos_b = {s: i for i, s in enumerate(order_b)}
    tau, _ = stats.kendalltau([pos_a[s] for s in common], [pos_b[s] for s in common])
    return tau


def pairwise_label_metrics(insts_a, insts_b) -> dict:
    all_a, all_b, n_inst = [], [], 0
    for ia, ib in zip(insts_a, insts_b):
        if ia["query"] != ib["query"]:
            continue
        la, lb = get_labels(ia), get_labels(ib)
        for sid in sorted(set(la) & set(lb)):
            all_a.append(la[sid])
            all_b.append(lb[sid])
        n_inst += 1
    if len(all_a) < 2:
        return dict(kappa=None, pct_agree=None, n_sub=len(all_a), n_inst=n_inst)
    kappa = cohen_kappa_score(all_a, all_b, weights="linear",
                              labels=list(range(len(LABELS))))
    pct   = sum(a == b for a, b in zip(all_a, all_b)) / len(all_a)
    return dict(kappa=kappa, pct_agree=pct, n_sub=len(all_a), n_inst=n_inst)


def pairwise_ranking_metrics(insts_a, insts_b) -> dict:
    taus = []
    for ia, ib in zip(insts_a, insts_b):
        if ia["query"] != ib["query"]:
            continue
        tau = kendall_tau(get_order(ia), get_order(ib))
        if tau is not None:
            taus.append(tau)
    if not taus:
        return dict(mean=None, std=None, n=0)
    return dict(mean=float(np.mean(taus)), std=float(np.std(taus)), n=len(taus))


def pairwise_within_label_ranking(insts_a, insts_b) -> dict[str, dict]:
    """
    For each label L, restrict to subclaims where both systems assigned L,
    then compute Kendall's τ-b on their relative positions in the full ordering.

    Returns {label: {mean, std, n_instances, total_subclaims}}
    """
    per_label: dict[str, list[float]] = {l: [] for l in LABELS}
    per_label_sub: dict[str, int] = {l: 0 for l in LABELS}

    for ia, ib in zip(insts_a, insts_b):
        if ia["query"] != ib["query"]:
            continue
        la, lb   = get_labels(ia), get_labels(ib)
        ord_a    = get_order(ia)
        ord_b    = get_order(ib)
        pos_a    = {s: i for i, s in enumerate(ord_a)}
        pos_b    = {s: i for i, s in enumerate(ord_b)}

        for lbl, lbl_int in LABEL_TO_INT.items():
            # subclaims that BOTH systems labelled as `lbl` and that appear in both orderings
            agreed = [
                sid for sid in set(la) & set(lb)
                if la[sid] == lbl_int and lb[sid] == lbl_int
                   and sid in pos_a and sid in pos_b
            ]
            per_label_sub[lbl] += len(agreed)
            if len(agreed) < 2:
                continue
            vec_a = [pos_a[s] for s in agreed]
            vec_b = [pos_b[s] for s in agreed]
            tau, _ = stats.kendalltau(vec_a, vec_b)
            if not np.isnan(tau):
                per_label[lbl].append(tau)

    result = {}
    for lbl in LABELS:
        taus = per_label[lbl]
        result[lbl] = dict(
            mean  = float(np.mean(taus)) if taus else None,
            std   = float(np.std(taus))  if taus else None,
            n_inst= len(taus),
            n_sub = per_label_sub[lbl],
        )
    return result


def label_distribution(insts) -> dict:
    counts = defaultdict(int)
    total  = 0
    for inst in insts:
        for sent in inst["decomposition"]:
            for atom in sent["decomp"]:
                total += 1
                if atom.get("importance") in LABEL_TO_INT:
                    counts[atom["importance"]] += 1
    return {l: counts[l] / total if total else 0 for l in LABELS}, total

# ── Markdown helpers ──────────────────────────────────────────────────────────

def md_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    sep  = "| " + " | ".join("-" * w for w in widths) + " |"
    head = "| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |"
        for row in rows
    )
    return "\n".join([head, sep, body])


def fmt_kappa(v):
    if v is None: return "—"
    return f"{v:+.3f}"

def fmt_pct(v):
    if v is None: return "—"
    return f"{v:.1%}"

def fmt_tau(mean, std):
    if mean is None: return "—"
    return f"{mean:+.3f} ± {std:.3f}"

# ── Main ──────────────────────────────────────────────────────────────────────

def analyse_condition(condition: str, sinks) -> None:
    def w(text=""):
        for s in sinks:
            print(text, file=s)

    # Load data
    data = {}
    for name, folder in SYSTEMS.items():
        path = folder / f"{condition}.jsonl"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {name}", file=sys.stderr)
            continue
        data[name] = load(path)

    names = list(data.keys())
    n     = len(names)
    if n < 2:
        w(f"  Not enough systems for condition '{condition}', skipping.")
        return

    # Compute pairwise
    lr_mat = {(i, j): None for i in range(n) for j in range(n)}
    rr_mat = {(i, j): None for i in range(n) for j in range(n)}
    wl_mat = {(i, j): None for i in range(n) for j in range(n)}

    for i, j in combinations(range(n), 2):
        a, b = names[i], names[j]
        lr = pairwise_label_metrics(data[a], data[b])
        rr = pairwise_ranking_metrics(data[a], data[b])
        wl = pairwise_within_label_ranking(data[a], data[b])
        lr_mat[i, j] = lr_mat[j, i] = lr
        rr_mat[i, j] = rr_mat[j, i] = rr
        wl_mat[i, j] = wl_mat[j, i] = wl

    # ── Cohen's weighted kappa ──
    w("### Cohen's Weighted Kappa (linear weights)")
    w()
    w("> Interpretation: < 0.20 slight | 0.21–0.40 fair | 0.41–0.60 moderate | 0.61–0.80 substantial | > 0.80 almost perfect")
    w()
    headers = ["System A \\ System B"] + names
    rows = []
    for i, na in enumerate(names):
        row = [na]
        for j, nb in enumerate(names):
            if i == j:
                row.append("—")
            else:
                v = lr_mat[i, j]
                row.append(fmt_kappa(v["kappa"] if v else None))
        rows.append(row)
    w(md_table(headers, rows))
    w()

    # ── Percent exact agreement ──
    w("### Percent Exact Label Agreement")
    w()
    rows = []
    for i, na in enumerate(names):
        row = [na]
        for j in range(n):
            if i == j:
                row.append("—")
            else:
                v = lr_mat[i, j]
                row.append(fmt_pct(v["pct_agree"] if v else None))
        rows.append(row)
    w(md_table(headers, rows))
    w()

    # ── Mean Kendall tau (full) ──
    w("### Mean Kendall's τ-b — Full Ordering (mean ± std)")
    w()
    rows = []
    for i, na in enumerate(names):
        row = [na]
        for j in range(n):
            if i == j:
                row.append("—")
            else:
                v = rr_mat[i, j]
                row.append(fmt_tau(v["mean"] if v else None, v["std"] if v else None))
        rows.append(row)
    w(md_table(headers, rows))
    w()

    # ── Within-label Kendall tau ──
    w("### Mean Kendall's τ-b — Within-Label Ordering")
    w()
    w("For each pair and label, τ-b is computed only among subclaims **both** systems")
    w("assigned to that label, using their positions in the full ranking order.")
    w()
    for lbl in LABELS:
        w(f"#### Label: `{lbl}`")
        w()
        rows = []
        for i, na in enumerate(names):
            row = [na]
            for j in range(n):
                if i == j:
                    row.append("—")
                else:
                    wl = wl_mat[i, j]
                    if wl is None:
                        row.append("—")
                    else:
                        v = wl[lbl]
                        cell = fmt_tau(v["mean"], v["std"])
                        if v["n_sub"] > 0:
                            cell += f" (n={v['n_sub']})"
                        row.append(cell)
            rows.append(row)
        w(md_table(headers, rows))
        w()

    # ── Label distribution ──
    w("### Label Distribution")
    w()
    dist_headers = ["System", "vital", "okay", "less-important", "total subclaims"]
    dist_rows = []
    for name, insts in data.items():
        dist, total = label_distribution(insts)
        dist_rows.append([
            name,
            fmt_pct(dist["vital"]),
            fmt_pct(dist["okay"]),
            fmt_pct(dist["less-important"]),
            str(total),
        ])
    w(md_table(dist_headers, dist_rows))
    w()

    # ── Instance / subclaim counts ──
    w("### Coverage (instances × labeled subclaims)")
    w()
    cov_headers = ["System", "instances", "labeled subclaims"]
    cov_rows = []
    for name, insts in data.items():
        labeled = sum(
            1 for inst in insts for sent in inst["decomposition"]
            for atom in sent["decomp"] if atom.get("importance") in LABEL_TO_INT
        )
        cov_rows.append([name, str(len(insts)), str(labeled)])
    w(md_table(cov_headers, cov_rows))
    w()


def main():
    out_path = Path(__file__).parent.parent / "results" / "summary.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as outfile:
        sinks = [sys.stdout, outfile]

        def w(text=""):
            for s in sinks:
                print(text, file=s)

        w("# Subclaim Importance Agreement — NQ Data")
        w()
        w(f"Models compared: {', '.join(SYSTEMS.keys())}")
        w()
        w("Metrics:")
        w("- **Cohen's weighted kappa** (linear weights, vital > okay > less-important)")
        w("- **% exact label agreement**")
        w("- **Mean Kendall's τ-b (full)** — full subclaim ordering")
        w("- **Mean Kendall's τ-b (within-label)** — ordering restricted to subclaims")
        w("  where both systems agreed on the label (vital / okay / less-important)")
        w()
        w("---")
        w()

        for condition in CONDITIONS:
            w(f"## Condition: `{condition}`")
            w()
            analyse_condition(condition, sinks)
            w("---")
            w()

    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
