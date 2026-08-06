# factscore

A trimmed, OpenAI-only fork of [FActScore](https://arxiv.org/abs/2305.14251)-style
claim decomposition and verification (Min et al., 2023), adapted for this paper.

- `atomic_facts.py` — `AtomicFactGenerator`: decomposes a response into
  sentences, then atomic facts per sentence, via few-shot prompting
  (`data/atomic_facts_demons.json`).
- `fact_scorer.py` — `FactScorer`: verifies each atomic fact against source
  documents. `get_score_with_retrieval` (used by `run_factscore*.py`) retrieves
  the top-5 BM25 passages per fact before judging; other `get_score_*` methods
  are alternative grounding strategies (no retrieval, full document, etc.)
  kept for reference but not used by the main pipeline.
- `openai_agent.py` — `OpenAIAgent`: the shared LLM client (`OPENAI_API_KEY`,
  disk-cached by prompt text under `.cache/`). Reused directly by `get_data/`
  and `vital/` so there's exactly one LLM-calling code path in the repo.
- `factscore.py` — `FactScore`: a convenience wrapper (`get_factscore(generations,
  knowledge_sources)`) for simple one-off use outside this paper's pipeline.
- `run_factscore.py` / `run_factscore_wh.py` — the CLI used to build VITALERRORS
  (see root README for when to use which).

```bash
python -m factscore.run_factscore --dataset bright --subset biology --prompt normal
```
