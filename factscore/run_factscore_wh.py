"""Same as run_factscore.py, but for datasets whose documents live in a shared
`original.jsonl` at the subset level rather than inline in each condition's
data.jsonl (currently only needed for wildhallucinations). Documents over
50,000 characters are dropped to keep prompts a reasonable size.

Example:
    python -m factscore.run_factscore_wh --dataset wildhallucinations --subset geographic --prompt normal
"""
import argparse
import json
import os

from tqdm import tqdm

from . import configs
from .atomic_facts import AtomicFactGenerator
from .fact_scorer import FactScorer
from .openai_agent import OpenAIAgent

import logging
logging.disable(logging.WARNING)

MAX_DOC_CHARS = 50000


def main():
    parser = argparse.ArgumentParser(description="Run FactScore over a dataset that stores documents in original.jsonl.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True, choices=["normal", "missing", "wrong"])
    parser.add_argument("--cache_path", type=str, default=".cache")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--demons_path", type=str, default=None,
                         help="Path to demonstrations for atomic fact generation. Defaults to the packaged demons.")
    args = parser.parse_args()

    in_file = os.path.join(args.root_dir, args.dataset, args.subset, args.prompt, "data.jsonl")
    in_file_docs = os.path.join(args.root_dir, args.dataset, args.subset, "original.jsonl")
    out_file = os.path.join(args.root_dir, args.dataset, args.subset, args.prompt, "factscore-out.jsonl")
    cache_path = os.path.join(args.cache_path, args.dataset, args.subset, args.prompt, "fs.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if args.demons_path:
        configs.atomic_facts_demons_path = args.demons_path
    configs.model_name = args.model

    client = OpenAIAgent(cache_path=cache_path)

    with open(in_file, "r") as input, open(in_file_docs, "r") as input_docs, open(out_file, "w") as output:
        for line, line_docs in tqdm(zip(input, input_docs)):
            line = json.loads(line)
            line_docs = json.loads(line_docs)

            query = line["query"]
            generation_dict = line["response"]
            generation = generation_dict["text"]
            docs = [d["text"] for d in line_docs["documents"] if len(d["text"]) < MAX_DOC_CHARS]

            # atomic fact generation
            atomicfg = AtomicFactGenerator(cache_path, client)
            facts = atomicfg.run(generation)
            afs = []
            for sent in facts:
                for af in sent[1]:
                    afs.append(af)

            # fact scoring
            fscorer = FactScorer(cache_path, client)
            if len(docs) == 0:
                fs_score = None
                decomposition = []
                for sent in facts:
                    temp_dict = {"sentence": sent[0], "decomp": []}
                    for af in sent[1]:
                        temp_dict["decomp"].append({"text": af, "judgment": None})
                    decomposition.append(temp_dict)
            else:
                scores = fscorer.get_score_with_retrieval(afs, docs)

                fact_to_judgment = {}
                tot = 0
                sup = 0
                for fact_dict in scores:
                    fact_to_judgment[fact_dict["fact"]] = fact_dict["is_supported"]
                    tot += 1
                    if fact_dict["is_supported"]:
                        sup += 1
                fs_score = sup / tot

                decomposition = []
                for sent in facts:
                    temp_dict = {"sentence": sent[0], "decomp": []}
                    for af in sent[1]:
                        temp_dict["decomp"].append({"text": af, "judgment": fact_to_judgment[af]})
                    decomposition.append(temp_dict)

            for_json = {
                "query": query,
                "response": generation_dict,
                "decomposition": decomposition,
                "score": fs_score,
            }
            output.write(json.dumps(for_json) + "\n")

    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
