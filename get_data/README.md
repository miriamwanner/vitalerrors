# get_data

Scripts used to build VITALERRORS from scratch. The full dataset is already
included under [`../data/`](../data/) — you only need these to regenerate it,
extend it to a new HF dataset, or change the generation model.

## Pipeline

1. **Download source data** (`get_hf_data/`) — one script per dataset, each
   downloads from HuggingFace and writes `data/{dataset}/{subset}/original.jsonl`:
   `get_bright_data.py`, `get_fs_data.py`, `get_hotpotqa_data.py`,
   `get_naturalquestions_data.py`, `get_triviaqa_data.py`, `get_wildhallucinations_data.py`.

   ```bash
   python -m get_data.data_setup.make_dirs --root_dir data   # create the directory tree first
   python -m get_data.get_hf_data.get_bright_data --out_dir data
   ```

2. **Generate the normal response** (`get_normal_response.py`) — prompts an
   LLM with each query (Table 6 in the paper) and writes `normal/data.jsonl`.

   ```bash
   python -m get_data.get_normal_response --dataset bright --subset biology
   ```

3. **Generate adversarial responses** (`get_adversarial_response.py`) —
   perturbs the normal response to omit (`missing`) or falsify (`wrong`) the
   key fact needed to answer the query (Table 7 in the paper), writing
   `missing/data.jsonl` and `wrong/data.jsonl`.

   ```bash
   python -m get_data.get_adversarial_response --dataset bright --subset biology --prompt missing
   python -m get_data.get_adversarial_response --dataset bright --subset biology --prompt wrong
   ```

4. **(Optional) spot-check adversarial quality** (`check_adversarial_quality.py`)
   — LLM-judges a sample of normal/missing/wrong responses for completeness/
   correctness, as a diagnostic (not ground truth).

5. **Run FactScore, Nuggetizer, and VITAL** — see the root README; those live
   in `../factscore/`, `../nuggetizer/`, `../vital/`.

`get_example.py` prints one query/response/subclaims/nuggets example for
manual inspection (no LLM calls).

## Notes

- `data_setup/data_info.py` is the single source of truth for which
  datasets/subsets exist; `../scoring/get_scores.py` and
  `../scripts/print_commands.py` both import it.
- For every dataset except `bright`, `get_normal_response.py` sets
  `documents: null` in the per-condition `data.jsonl` — see the root README's
  note on `_wh`-suffixed scripts for why.
