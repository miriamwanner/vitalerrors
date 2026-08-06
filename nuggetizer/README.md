# nuggetizer

A trimmed, OpenAI/Azure-OpenAI-only vendored copy of
[AutoNuggetizer](https://arxiv.org/abs/2411.09607) (Pradeep et al., TREC 2024
RAG Track), originally implementing the nugget framework from Voorhees (2003).
Upstream: [castorini/nuggetizer](https://github.com/castorini/nuggetizer).

This is a fork rather than a `pip install nuggetizer` dependency because the
upstream `core/llm.py` imports a Microsoft-internal package unconditionally
(it would fail to import at all without it); `core/llm.py` here is rewritten
to keep only the already-present OpenAI/Azure OpenAI code paths.

- `models/nuggetizer.py` — `Nuggetizer.create(request)`: extracts and
  vital/okay-scores nuggets from a request's documents. `.assign(query,
  context, nuggets)`: judges whether a piece of text (here, an LLM response)
  supports each nugget (`support`/`partial_support`/`not_support`).
- `core/metrics.py` — `calculate_nugget_scores`: strict/partial-credit
  vital/all recall scores. This paper reports the "All Strict" variant as
  NUGGETRECALL.
- `core/llm.py` — `LLMHandler`: the OpenAI/Azure OpenAI client used internally
  by `Nuggetizer` (separate from `factscore/openai_agent.py` so this package
  stays usable standalone).
- `run_nuggetizer.py` / `run_nuggetizer_wh.py` — the CLI used to build
  VITALERRORS (see root README for when to use which).

```bash
python -m nuggetizer.run_nuggetizer --dataset bright --subset biology --prompt normal
```
