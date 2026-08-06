# VITALERRORS dataset

6,726 queries across 6 datasets, each with a `normal`, `missing`, and `wrong`
response condition, plus the outputs of every stage of the VITAL pipeline
already computed (so you can inspect/reuse results without spending on LLM
calls).

## Download

This directory (~880MB) is not tracked in git. Download it from:

**[PLACEHOLDER: data download link]**

and extract it so this file ends up at `data/README.md` relative to the repo
root (i.e. extract directly into the repo root, or extract elsewhere and move
the resulting `data/` folder there).

## Layout

```
data/{dataset}/{subset}/
  original.jsonl            # shared input: query (+ gold answer where available) + documents
  normal/
    data.jsonl               # query + normal LLM response (+ documents, bright only)
    factscore-out.jsonl       # FactScore decomposition + per-subclaim judgment
    nuggets-out.jsonl         # nuggets + importance + response support
    new-metric-out.jsonl      # VITAL: subclaim importance + weighted scores
  missing/                    # same 4 files, response has a key fact removed
  wrong/                      # same 4 files, response has a key fact falsified
```

`original.jsonl` is response-independent (query, documents, and — for `bright`
— a gold answer); each condition's `data.jsonl` is `original.jsonl` with
`response.text` filled in by that condition's LLM generation. For every
dataset except `bright`, `data.jsonl`'s own `documents` field is `null` —
`factscore`/`nuggetizer` read documents from the subset-level `original.jsonl`
instead (see the root README's note on `_wh`-suffixed scripts).

## Datasets

| Dataset | Subsets | Queries | Source |
|---|---|---|---|
| BRIGHT | biology, earth_science, economics, psychology, robotics, stackoverflow, sustainable_living | 749 | Real-world StackExchange questions + linked sources |
| FActScore Bios | bios | 500 | Generated people biographies + corresponding Wikipedia page |
| WildHallucinations | cult_ent_1-4, geographic | 2,477 | Entities from user conversations + curated web pages |
| HotpotQA | val_1, val_2 | 1,000 | Multi-hop reasoning questions + supporting documents |
| Natural Questions | val_1, val_2 | 1,000 | Real Google search queries + results |
| TriviaQA | rc_1, rc_2 | 1,000 | Trivia questions + reference documents |

BRIGHT and FActScore Bios/WildHallucinations are open-ended queries (many
valid answers); HotpotQA/NQ/TriviaQA are single-answer queries (one specific
piece of information is required). See Section 3.2 / Table 1 of the paper.

A `truthfulqa` dataset exists in the original research repo but is not one of
the 6 datasets reported in the paper, so it's excluded here to keep this
release matching the paper exactly.

## Schema notes

- `response`/`gold_response`: `{"id": str, "text": str}`.
- `documents`: `[{"id": str, "text": str}, ...]`, except `factscore/bios` (no
  `id`, single document) and `wildhallucinations/*` (`{"status_code", "text", "url"}`,
  plus a `wiki` field with the source entity's Wikipedia page).
- `factscore-out.jsonl`: `{"query", "response", "decomposition": [{"sentence", "decomp": [{"text", "judgment"}]}], "score"}`.
- `nuggets-out.jsonl`: `{"query", "response": {..., "nugget-assignment": [...]}, "nuggets": [{"id", "text", "importance"}], "documents"}`.
- `new-metric-out.jsonl`: `{"query", "response", "decomposition": [... same as factscore-out, plus "importance" per subclaim ...], "N2S", "subclaim-importance-order", "scores": {...}}`.

To regenerate any of this from scratch (e.g. to extend to a new dataset or a
different model), see [`../get_data/README.md`](../get_data/README.md).
