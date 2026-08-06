# analysis

Scripts that turn `data/`'s `*-out.jsonl` files into the paper's tables and
figures. None of these make LLM calls — they only read already-computed
scores, so they're free to re-run.

- `motivating_graph.py` — synthetic illustrative precision/recall curves for
  the paper's introductory example (Figure 1/2), not computed from real data.
- `graph_real_data.py` / `graph_real_data_error_bars.py` — cumulative
  FactScore precision by subclaim position, averaged across a dataset group
  (Figure 3/4: single-answer vs. open-ended queries); the `_error_bars`
  variant adds `scipy.stats`-based confidence intervals.
- `precision_recall_table.py` / `precision_recall_one_graph.py` — grouped bar
  charts comparing FactScore/VITAL-precision/linear-decay-precision against
  NuggetRecall/VITAL-recall across dataset pairs (Table 4-style comparisons).
- `get_counts_error_tables.py` / `final_presentation.py` — subclaim/nugget
  count tables (Table 3) and normal/missing/wrong error-detection comparisons
  (Table 5) as grouped bar charts; there's real overlap between these two
  (both build a `create_comparison_chart`-style figure), kept as two files
  since they were used to produce distinct output artifacts historically
  (`counts_errors/` vs `figures/`) — read both before assuming one supersedes
  the other for a given plot.
- `get_ranked_example.py` — `RankedSubclaims`: reconstructs a response's
  original vs. importance-ranked subclaim order for a qualitative example
  (feeds `rank_example/*.txt`, e.g. Table 9).
- `get_data_no_out.py` — lists which dataset/subset/prompt combos are missing
  one of the 4 output files, useful after a partial pipeline run.
- `figures/`, `counts_errors/`, `rank_example/` — pre-generated output
  artifacts from the paper; safe to delete and regenerate.

Most scripts take `--root_dir` (default `data`); run with `--help` to see
each one's specific arguments.
