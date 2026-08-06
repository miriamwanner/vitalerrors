# vital

The paper's core contribution (Section 3.1): importance-weighted factuality
metrics built on top of `factscore/` subclaims and `nuggetizer/` nuggets.

- `prompts.py` (Table 8 in the paper):
  - `SUBCLAIM_QUERY_IMPORTANCE` — ranks and labels a response's decomposed
    subclaims as `vital`/`okay`/`less-important`, independent of correctness.
  - `NUGGET_SUBCLAIM_ALIGNMENT` — aligns nuggets to subclaims (used only to
    report the alignment in the output; it does not affect the score).
  - `EVAL_PROMPT` — a batch subclaim-verification prompt kept for reference
    (the shipped pipeline uses `factscore`'s per-fact retrieval-grounded
    scoring instead).
- `metrics.py`:
  - `DataProcessor.get_subclaim_importance_no_alignment` — the
    `SUBCLAIM_QUERY_IMPORTANCE` call that actually drives scoring.
  - `MetricCalculator.calculate_scores` — VITAL_PREC/VITAL_REC: precision/recall
    weighted by importance tier (`--vital-weight`/`--okay-weight`/
    `--less-important-weight`), plus the precision/count breakdown per tier
    used to compute the response-level VITAL_RLP ("any vital subclaim wrong?")
    and VITAL_RLR ("any vital nugget unsupported?") booleans from the paper.
  - `MetricCalculator.linear_decay_weighting` — the positional linear-decay
    alternative from Appendix B (and its top-5 variant).
- `run_vital_metric.py` — the CLI; reads `data.jsonl` + `factscore-out.jsonl`
  + `nuggets-out.jsonl` and writes `new-metric-out.jsonl`. Works for every
  dataset (including wildhallucinations) since it never touches raw documents
  directly — it only needs the query, the FactScore decomposition, and the
  nugget assignments, all of which are already resolved by the earlier stages.

```bash
python -m vital.run_vital_metric --dataset bright --subset biology --prompt normal \
    --vital-weight 1.0 --okay-weight 0.5 --less-important-weight 0.1
```
