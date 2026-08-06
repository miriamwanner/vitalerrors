# VITAL / VITALERRORS

Code and data for **"All Claims Are Equal, but Some Claims Are More Equal Than
Others: Importance-Sensitive Factuality Evaluation of LLM Generations"**
([arXiv:2510.07083](https://arxiv.org/abs/2510.07083)).

This repo contains:
- **VITALERRORS**: a benchmark of 6,726 queries across 6 QA/open-ended datasets,
  each with a normal LLM response and two adversarially perturbed versions
  (`missing` a key fact, `wrong` about a key fact) — see [`data/`](data/).
- **VITAL**: a set of importance-weighted factuality metrics (`vital/`) built
  on top of a FactScore-style decomposition pipeline (`factscore/`) and a
  nugget-based recall pipeline (`nuggetizer/`).
- The scripts used to build VITALERRORS from scratch (`get_data/`), aggregate
  scores (`scoring/`), and reproduce the paper's tables/figures (`analysis/`,
  `revisions/`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

`data/` (~880MB) is not tracked in this git repo — download it separately from
**[this link](https://drive.google.com/drive/folders/1SJNMu8Al6uFj11FV1S8TcUTpytl0iPmt?usp=sharing)** and extract it to `data/` at the repo
root. See [`data/README.md`](data/README.md) for the full layout.

All scripts are invoked as modules from the repo root (`python -m package.script ...`),
since they use relative/package-local imports.

**Score an example that's already included in the dataset** (no LLM calls needed
if you just want to inspect existing scores):

```bash
python -m get_data.get_example --dataset bright --subset biology --prompt normal
```

**Run the full pipeline on one dataset/subset/prompt condition** (makes real
OpenAI API calls, results are disk-cached under `.cache/` so re-runs are free):

```bash
python -m factscore.run_factscore   --dataset bright --subset biology --prompt normal
python -m nuggetizer.run_nuggetizer --dataset bright --subset biology --prompt normal
python -m vital.run_vital_metric    --dataset bright --subset biology --prompt normal
```

`bright` is the only dataset with documents inline in each condition's
`data.jsonl`; every other dataset (`factscore`, `wildhallucinations`,
`hotpotqa`, `naturalquestions`, `triviaqa`) stores documents in a shared
`original.jsonl` and needs the `_wh`-suffixed script variant instead:

```bash
python -m factscore.run_factscore_wh   --dataset triviaqa --subset rc_1 --prompt normal
python -m nuggetizer.run_nuggetizer_wh --dataset triviaqa --subset rc_1 --prompt normal
python -m vital.run_vital_metric       --dataset triviaqa --subset rc_1 --prompt normal
```

**Print commands for every dataset/subset/prompt** instead of typing them by hand:

```bash
python -m scripts.print_commands factscore
python -m scripts.print_commands nuggets
python -m scripts.print_commands vital
```

**Aggregate scores across everything** into one table:

```bash
python -m scoring.get_scores --root_dir data
```

## Repo layout

```
data/          VITALERRORS dataset (queries, responses, and cached pipeline outputs)
               -- not in git, download separately (see Quickstart / data/README.md)
factscore/     FactScore-style claim decomposition + verification (OpenAI backend)
nuggetizer/    Nugget extraction/scoring/assignment (AutoNuggetizer, vendored + trimmed)
vital/         The paper's core contribution: importance-weighted precision/recall
get_data/      Scripts to build VITALERRORS from scratch (HF downloads + response generation)
scoring/       Aggregates factscore-out/nuggets-out/new-metric-out into one results table
analysis/      Reproduces the paper's tables and figures from data/
revisions/     Camera-ready backing material: significance tests, an inter-model/
               inter-annotator agreement study, and a SAFE-baseline ablation
scripts/       Misc utilities (print_commands.py)
```

Each of these has its own `README.md` with more detail.

## Pipeline

Each query has three response conditions: `normal` (unperturbed LLM response),
`missing` (a key fact removed), and `wrong` (a key fact falsified). For each
condition:

1. **FactScore** (`factscore/`) decomposes the response into atomic subclaims
   and verifies each against the source documents → `factscore-out.jsonl`.
2. **Nuggetizer** (`nuggetizer/`) extracts importance-labeled (vital/okay)
   nuggets from the source documents and scores how well the response
   supports them → `nuggets-out.jsonl`.
3. **VITAL** (`vital/`) ranks the FactScore subclaims by query-importance
   (vital/okay/less-important, independent of correctness) and computes
   weighted precision/recall/F1, plus the response-level VITAL_RLP/VITAL_RLR
   booleans described in the paper → `new-metric-out.jsonl`.

`scoring/get_scores.py` then averages all three outputs (plus a SAFE-style
recall/F1 computed from the FactScore decomposition) into one table.

## Notes

- All LLM calls go through the OpenAI API (`OPENAI_API_KEY`) and are
  disk-cached (keyed by prompt text) under `.cache/`, so re-running a script
  after a partial failure won't re-pay for already-completed calls.
- `nltk.download('punkt_tab')` runs once on first import of `factscore/atomic_facts.py`
  and needs internet access the first time.
- The paper's model was `gpt-4o`; all scripts default `--model gpt-4o` but accept
  any chat-completions-compatible OpenAI model name.

## Citation

```bibtex
@misc{wanner2025claimsequalclaimsequal,
      title={All Claims Are Equal, but Some Claims Are More Equal Than Others: Importance-Sensitive Factuality Evaluation of LLM Generations}, 
      author={Miriam Wanner and Leif Azzopardi and Paul Thomas and Soham Dan and Benjamin Van Durme and Nick Craswell},
      year={2025},
      eprint={2510.07083},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.07083}, 
}
```
