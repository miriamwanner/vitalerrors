"""Generate the 'normal' (unperturbed) LLM response for each query in a
dataset/subset's original.jsonl.

Example:
    python -m get_data.get_normal_response --dataset bright --subset biology
"""
import argparse
import json
import os

from tqdm import tqdm

from factscore import configs
from factscore.openai_agent import OpenAIAgent

PROMPTS = {
    "bright": "Answer the following question in a short paragraph:\n[QUERY]",
    "factscore": "In a paragraph, tell me a bio of [TOPIC]",
    "wildhallucinations": "In a paragraph, could you tell me what you know about [TOPIC]?",
    "hotpotqa": "In a paragraph, could you answer: [QUERY]",
    "naturalquestions": "In a paragraph, could you answer: [QUERY]",
    "triviaqa": "In a paragraph, could you answer: [QUERY]",
}


def get_prompt(dataset, query):
    if dataset not in PROMPTS:
        raise ValueError(f"No normal-response prompt registered for dataset {dataset!r}")
    placeholder = "[TOPIC]" if dataset in ("factscore", "wildhallucinations") else "[QUERY]"
    return PROMPTS[dataset].replace(placeholder, query)


def main():
    parser = argparse.ArgumentParser(description="Generate normal responses for a dataset/subset.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--cache_path", type=str, default=".cache")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    args = parser.parse_args()

    in_file = os.path.join(args.root_dir, args.dataset, args.subset, "original.jsonl")
    out_file = os.path.join(args.root_dir, args.dataset, args.subset, "normal", "data.jsonl")
    cache_path = os.path.join(args.cache_path, args.dataset, args.subset, "normal", "cache.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    configs.model_name = args.model
    configs.temp = 0.2
    configs.max_tokens = 2000
    client = OpenAIAgent(cache_path=cache_path)

    count = 0
    with open(in_file, "r") as input, open(out_file, "w") as output:
        for line in tqdm(input):
            line = json.loads(line)
            query = line["query"]

            prompt = get_prompt(args.dataset, query)
            response = client.generate(prompt)

            # store the fully-rendered prompt as the query, matching what the model saw
            line["query"] = prompt
            line["response"] = {"id": str(count), "text": response}
            if args.dataset != "bright":
                # For every dataset except bright, factscore/nuggetizer read documents
                # from the subset-level original.jsonl (see run_factscore_wh.py /
                # run_nuggetizer_wh.py), so the per-condition data.jsonl doesn't need
                # its own copy. bright is the exception: its documents are inline and
                # small enough to carry through unchanged.
                line["documents"] = None

            output.write(json.dumps(line) + "\n")
            count += 1
            if count % 10 == 0:
                client.cache.save_cache()
    client.cache.save_cache()
    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
