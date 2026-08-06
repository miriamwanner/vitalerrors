# revisions

Camera-ready backing material — statistical/robustness checks referenced
during review, kept separate from the main pipeline since they answer
specific reviewer questions rather than produce VITALERRORS itself.

- `stat-sig/` — `stat_sig.py`: paired significance tests (t-test, Wilcoxon,
  McNemar, Holm-Bonferroni corrected) comparing normal/missing/wrong
  conditions across metrics — the statistical backing for the paper's main
  results tables. Results in `results/`.
- `nq-data/` — `code/run_importance_ranking.py`: re-runs the
  `SUBCLAIM_QUERY_IMPORTANCE` step (see `../vital/prompts.py`) on 50 Natural
  Questions instances across multiple models (gpt-4o, Llama-3.3-70B,
  Qwen3-235B, gpt-oss-120b, plus human annotations) to measure inter-model/
  inter-annotator agreement (Cohen's weighted kappa, exact agreement,
  Kendall's tau-b). `code/compare_rankings.py` / `code/full_analysis.py`
  compute the agreement statistics; `code/scratch.sh` has example invocations
  against a local vLLM server or the Together AI API. Results in `results/`;
  raw per-model outputs in `model-outputs/`.
- `safe-ablation/` — `safe_scores_table.py`: compares VITAL scores against a
  SAFE-style baseline recall definition. Tables in `safe-tables/`.

`run_importance_ranking.py` takes `--model-url`/`--api-key`/`--model-name`
directly (any OpenAI-compatible chat completions endpoint), independent of
the `factscore`/`nuggetizer`/`vital` packages elsewhere in this repo.
