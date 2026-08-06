#!/usr/bin/env python3
"""
Re-runs the SUBCLAIM_QUERY_IMPORTANCE ranking step on nq-data instances using
any OpenAI-compatible model endpoint (vLLM, Together, OpenAI, etc.).

Takes a JSONL file that already has decomposition+judgments and outputs a new
JSONL with updated decomposition[*].decomp[*].importance and
subclaim-importance-order filled in by the specified model.

Requirements:
    pip install openai tqdm

Usage examples:
    # Local vLLM server
    python run_importance_ranking.py \
        --input ../normal.jsonl \
        --model-url http://localhost:8000/v1 \
        --model-name meta-llama/Llama-3.1-70B-Instruct

    # Together AI
    python run_importance_ranking.py \
        --input ../normal.jsonl \
        --model-url https://api.together.xyz/v1 \
        --model-name meta-llama/Llama-3.1-70B-Instruct-Turbo \
        --api-key $TOGETHER_API_KEY

    # OpenAI
    python run_importance_ranking.py \
        --input ../normal.jsonl \
        --model-url https://api.openai.com/v1 \
        --model-name gpt-4o \
        --api-key $OPENAI_API_KEY

Output goes to:  ../model-outputs/<model-name-slug>/<input-filename>
Cache goes to:   ../model-outputs/<model-name-slug>/<input-stem>_cache.pkl
"""

import argparse
import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


# ── Prompt (matches experiments/our-metric/prompts.py SUBCLAIM_QUERY_IMPORTANCE) ──

SUBCLAIM_QUERY_IMPORTANCE = '''You are performing step two of a four part fact-checking process:
(1) Decompose a paragraph into individual claims.
(2) Given a query and set of claims, rank by decreasing query-importance (this step).
(3) Check the correctness of each claim.
(4) Score the paragraph, weighting by importance.
This step is completely independent of factual correctness, and only focuses on the query-importance of claims for answering the query. Even factually incorrect claims should be ranked highly if they directly answer the query.

Instructions: You are provided with a query and set of claims. Rank the claims in decreasing order of query-importance. A claim exhibits high query-importance when it addresses a central aspect of the query, and low query-importance when it contributes only peripheral or background information. Rank claims independent of correctness, instead only based on query-importance. A later step will check for correctness of claims.

Assign query-importance labels using exactly these three categories:
- "vital" - Essential claims that directly address the core query
- "okay" - Supporting claims that provide useful but non-essential information
- "less-important" - Background or tangentially related claims with minimal relevance

Ordering Rules:
- All "vital" claims must appear first, then all "okay" claims come second, and "less-important" claims come last.
- Within each category, order by decreasing importance.
- If two or more claims address the same aspect of the query, keep them grouped in the order they appear, even if their answers contradict. For example:
    ...
    [S3] Washington, D.C. is the capital of Canada.: "vital"
    [S8] Washington, D.C. is the capital of the United States.: "vital"
    ...
- Do not adjust rankings based on factual correctness, this will be handled in step 3.

Output Format:
[Claim ID] <claim text>: "label"
[Claim ID] <claim text>: "label"
...

Requirements:
- Label every claim exactly once
- Use only the three specified labels
- Maintain the original claim count
- Return only the labeled, ordered list (no explanations)
Below is your task.

###Your task:
Query: [QUERY]
Claims:
[SUBCLAIMS]
Ranked Claims:'''

VALID_LABELS = {"vital", "okay", "less-important"}


# ── Cache helpers ──────────────────────────────────────────────────────────────

def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)


def cache_key(model_name: str, prompt: str) -> str:
    return hashlib.sha256((model_name + "|||" + prompt).encode()).hexdigest()


# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, model_name: str, prompt: str,
             temperature: float, max_tokens: int, max_retries: int = 5) -> str:
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exc = e
            wait = 2 ** attempt
            print(f"  LLM error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait}s.")
            time.sleep(wait)
    raise RuntimeError(f"All {max_retries} LLM attempts failed") from last_exc


# ── Prompt building & parsing ─────────────────────────────────────────────────

def build_subclaim_str(instance: dict) -> tuple[list[str], str]:
    """Returns (subclaim_ids_in_order, formatted_subclaim_string)."""
    ids = []
    parts = []
    for sent in instance["decomposition"]:
        for atom in sent["decomp"]:
            ids.append(atom["id"])
            parts.append(f"[{atom['id']}] {atom['text']}")
    return ids, "\n".join(parts)


def parse_response(text: str, subclaim_ids: list[str]) -> tuple[dict, list[str]]:
    """
    Parses lines of the form:  [Sx] claim text: "label"
    Returns (subclaim2importance, ordered_subclaim_ids).
    """
    id_set = set(subclaim_ids)
    pattern = re.compile(r'\[S(\d+)\]\s*.+?:\s*"([^"]+)"')

    subclaim2importance = {}
    subclaim_order = []

    for match in pattern.finditer(text):
        sid = f"S{match.group(1)}"
        label = match.group(2).strip()
        if sid in id_set and label in VALID_LABELS and sid not in subclaim2importance:
            subclaim2importance[sid] = label
            subclaim_order.append(sid)

    # Fall back: any missed subclaims get less-important at the end
    for sid in subclaim_ids:
        if sid not in subclaim2importance:
            subclaim2importance[sid] = "less-important"
            subclaim_order.append(sid)

    return subclaim2importance, subclaim_order


# ── Per-instance importance ranking ──────────────────────────────────────────

def rank_instance(client: OpenAI, model_name: str, instance: dict,
                  temperature: float, max_tokens: int,
                  cache: dict, max_retries: int = 5) -> tuple[dict, list[str]]:
    """
    Runs the SUBCLAIM_QUERY_IMPORTANCE prompt on one instance.
    Returns (subclaim2importance, subclaim_order).
    """
    subclaim_ids, subclaim_str = build_subclaim_str(instance)
    prompt = (SUBCLAIM_QUERY_IMPORTANCE
              .replace("[QUERY]", instance["query"])
              .replace("[SUBCLAIMS]", subclaim_str))

    ck = cache_key(model_name, prompt)

    # Pull from cache or call the model
    if ck in cache:
        response_text = cache[ck]
    else:
        response_text = call_llm(client, model_name, prompt, temperature, max_tokens, max_retries)
        cache[ck] = response_text

    # Parse; if we don't get all subclaims back, retry the LLM call
    for attempt in range(max_retries):
        subclaim2importance, subclaim_order = parse_response(response_text, subclaim_ids)
        # Check that every id was found in the parsed output (before the fallback filled gaps)
        parsed_ids = set()
        for match in re.finditer(r'\[S(\d+)\]', response_text):
            parsed_ids.add(f"S{match.group(1)}")
        missing = [s for s in subclaim_ids if s not in parsed_ids]
        if not missing:
            break
        if attempt < max_retries - 1:
            print(f"  Missing {len(missing)} subclaims in parse, re-calling LLM (attempt {attempt + 2})")
            del cache[ck]
            response_text = call_llm(client, model_name, prompt, temperature, max_tokens, max_retries)
            cache[ck] = response_text

    return subclaim2importance, subclaim_order


# ── Output record construction ────────────────────────────────────────────────

def build_output_record(instance: dict, subclaim2importance: dict,
                        subclaim_order: list[str]) -> dict:
    """Builds the output JSONL record with updated importance labels and ordering."""
    new_decomp = []
    for sent in instance["decomposition"]:
        new_sent = {"sentence": sent["sentence"], "decomp": []}
        for atom in sent["decomp"]:
            new_sent["decomp"].append({
                "id": atom["id"],
                "text": atom["text"],
                "judgment": atom.get("judgment"),
                "importance": subclaim2importance.get(atom["id"], "less-important"),
            })
        new_decomp.append(new_sent)

    return {
        "query": instance["query"],
        "response": instance["response"],
        "decomposition": new_decomp,
        "N2S": instance.get("N2S", {}),
        "subclaim-importance-order": subclaim_order,
    }


# ── Filesystem helpers ────────────────────────────────────────────────────────

def model_slug(model_name: str) -> str:
    """Converts a model name to a filesystem-safe directory name."""
    return re.sub(r"[^\w.-]", "_", model_name).strip("_")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Re-rank subclaim importance using any OpenAI-compatible model endpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL (e.g. ../normal.jsonl)")
    parser.add_argument("--model-url", required=True,
                        help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model-name", required=True,
                        help="Model name as expected by the endpoint")
    parser.add_argument("--api-key", default="EMPTY",
                        help="API key. Defaults to 'EMPTY' (fine for local vLLM)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=4000)
    parser.add_argument("--output", default=None,
                        help="Output file path. Default: ../model-outputs/<model-slug>/<input-name>")
    parser.add_argument("--resume", action="store_true",
                        help="Append to output and skip already-processed instances")
    parser.add_argument("--test", action="store_true",
                        help="Dry-run: process one instance, print prompt + response + parsed result, write nothing")
    parser.add_argument("--test-idx", type=int, default=0,
                        help="Which instance to use in --test mode (default: 0)")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    slug = model_slug(args.model_name)

    client = OpenAI(base_url=args.model_url, api_key=args.api_key)

    # Load instances
    with open(input_path) as f:
        instances = [json.loads(line) for line in f if line.strip()]

    # ── Test mode ──────────────────────────────────────────────────────────────
    if args.test:
        instance = instances[args.test_idx]
        subclaim_ids, subclaim_str = build_subclaim_str(instance)
        prompt = (SUBCLAIM_QUERY_IMPORTANCE
                  .replace("[QUERY]", instance["query"])
                  .replace("[SUBCLAIMS]", subclaim_str))

        print("=" * 70)
        print(f"TEST MODE  —  instance {args.test_idx}  —  {args.model_name}")
        print("=" * 70)
        print(f"\nQUERY:\n  {instance['query']}\n")
        print(f"SUBCLAIMS ({len(subclaim_ids)} total, original order):")
        for sid in subclaim_ids:
            for sent in instance["decomposition"]:
                for atom in sent["decomp"]:
                    if atom["id"] == sid:
                        print(f"  [{sid}] {atom['text']}")
        print("\n" + "-" * 70)
        print("PROMPT SENT TO MODEL:\n")
        print(prompt)
        print("\n" + "-" * 70)
        print("CALLING MODEL...")

        response_text = call_llm(client, args.model_name, prompt, args.temperature, args.max_tokens)

        print("\nRAW RESPONSE:\n")
        print(response_text)
        print("\n" + "-" * 70)

        subclaim2importance, subclaim_order = parse_response(response_text, subclaim_ids)
        missing = [s for s in subclaim_ids if s not in set(subclaim_order[:len(subclaim_ids)])]

        print("\nPARSED RESULT:")
        print(f"  Order ({len(subclaim_order)} subclaims):")
        for rank, sid in enumerate(subclaim_order, 1):
            label = subclaim2importance.get(sid, "?")
            for sent in instance["decomposition"]:
                for atom in sent["decomp"]:
                    if atom["id"] == sid:
                        print(f"  {rank:>3}. [{sid}] {label:<16}  {atom['text'][:80]}")
        if missing:
            print(f"\n  WARNING: {len(missing)} subclaims missing from response (fell back to less-important): {missing}")
        else:
            print(f"\n  All {len(subclaim_ids)} subclaims accounted for.")
        return
    # ── End test mode ──────────────────────────────────────────────────────────

    # Resolve output path
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = input_path.parent.parent / "model-outputs" / slug / input_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache_path = out_path.with_name(out_path.stem + "_cache.pkl")

    print(f"Input:      {input_path}")
    print(f"Model:      {args.model_name}  @  {args.model_url}")
    print(f"Output:     {out_path}")
    print(f"Cache:      {cache_path}")

    # Load cache and client
    cache = load_cache(cache_path)

    print(f"Instances:  {len(instances)}")

    # Resume: count already-written lines
    already_done = 0
    if args.resume and out_path.exists():
        with open(out_path) as f:
            already_done = sum(1 for line in f if line.strip())
        print(f"Resuming:   skipping first {already_done} instances")

    write_mode = "a" if args.resume else "w"
    errors = 0

    with open(out_path, write_mode) as writer:
        for i, instance in enumerate(tqdm(instances, desc=slug)):
            if i < already_done:
                continue

            try:
                subclaim2importance, subclaim_order = rank_instance(
                    client, args.model_name, instance,
                    args.temperature, args.max_tokens, cache,
                )
                result = build_output_record(instance, subclaim2importance, subclaim_order)
            except Exception as e:
                print(f"\n  Error on instance {i} (query: {instance['query'][:60]}...): {e}")
                # Fall back: original order, all less-important
                fallback_ids = [a["id"] for s in instance["decomposition"] for a in s["decomp"]]
                fallback_imp = {sid: "less-important" for sid in fallback_ids}
                result = build_output_record(instance, fallback_imp, fallback_ids)
                errors += 1

            writer.write(json.dumps(result) + "\n")
            writer.flush()

            if (i + 1) % 10 == 0:
                save_cache(cache, cache_path)

        save_cache(cache, cache_path)

    total = len(instances) - already_done
    print(f"\nDone. Processed {total} instances ({errors} errors).")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
