"""Adversarially perturb each normal response to omit ('missing') or falsify
('wrong') the key information needed to answer the query (Table 7 prompts).

Example:
    python -m get_data.get_adversarial_response --dataset bright --subset biology --prompt missing
"""
import argparse
import json
import os

from tqdm import tqdm

from factscore import configs
from factscore.openai_agent import OpenAIAgent

MISSING_PROMPT = """You are given a query and an answer. Your task is to modify the answer by removing the most important information for answering the question. You are first given an example, and then your task.

EXAMPLE
Query: Which is denser, water vapor or air?
Answer: Air is denser than water vapor. This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation. At the same temperature and pressure, a volume of moist air will weigh less than the same volume of dry air, confirming that water vapor is less dense than the air it mixes with.
Modified answer: This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation.


YOUR TASK
Query: [QUERY]
Answer: [ANSWER]
Modified answer:"""

WRONG_PROMPT = """You are given a query and its corresponding answer. Your task is to make a modification to one sentence from the answer by changing the key piece of information required to answer the question correctly, thereby making the answer factually incorrect. This will most likely be a change to the first sentence. Do not alter the remainder of the response. An example is provided first, followed by your task.

EXAMPLE
Query: Which is denser, water vapor or air?
Answer: Air is denser than water vapor. This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation. At the same temperature and pressure, a volume of moist air will weigh less than the same volume of dry air, confirming that water vapor is less dense than the air it mixes with.
Modified answer: Water vapor is denser than air. This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation. At the same temperature and pressure, a volume of moist air will weigh less than the same volume of dry air, confirming that water vapor is less dense than the air it mixes with.


YOUR TASK
Query: [QUERY]
Answer: [ANSWER]
Modified answer:"""


def get_prompt(prompt_name, query, answer):
    if prompt_name == "missing":
        return MISSING_PROMPT.replace("[QUERY]", query).replace("[ANSWER]", answer)
    elif prompt_name == "wrong":
        return WRONG_PROMPT.replace("[QUERY]", query).replace("[ANSWER]", answer)
    else:
        raise ValueError("prompt option not from list of choices")


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial (missing/wrong) responses for a dataset/subset.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True, choices=["missing", "wrong"])
    parser.add_argument("--cache_path", type=str, default=".cache")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    args = parser.parse_args()

    in_file = os.path.join(args.root_dir, args.dataset, args.subset, "normal", "data.jsonl")
    out_file = os.path.join(args.root_dir, args.dataset, args.subset, args.prompt, "data.jsonl")
    cache_path = os.path.join(args.cache_path, args.dataset, args.subset, args.prompt, "cache.pkl")
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
            answer = line["response"]["text"]

            prompt = get_prompt(args.prompt, query, answer)
            response = client.generate(prompt)

            line["response"]["text"] = response

            output.write(json.dumps(line) + "\n")
            count += 1
            if count % 10 == 0:
                client.cache.save_cache()
    client.cache.save_cache()
    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
