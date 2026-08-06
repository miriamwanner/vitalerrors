"""Spot-check the quality of adversarially generated missing/wrong responses:
for each of normal/missing/wrong, ask an LLM whether the response is missing
key information, sufficiently answers the query, and is factually correct.
Useful as a sanity check after running get_adversarial_response.py, on a
sample of the data (LLM-judged, so treat as a diagnostic, not ground truth).

Example:
    python -m get_data.check_adversarial_quality --dataset bright --subset biology --limit 100
"""
import argparse
import json
import os

from tqdm import tqdm

from factscore import configs
from factscore.openai_agent import OpenAIAgent

MISSING_INFO_CHECK = """You are provided with a query and an answer. Your task is to determine if the answer is incomplete and missing any key information. Respond with INCOMPLETE if the answer is missing information and COMPLETE if the answer is complete. Respond with nothing else. You are provided with an example and then your task.

EXAMPLE
Query: Which is denser, water vapor or air?
Answer: This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation. At the same temperature and pressure, a volume of moist air will weigh less than the same volume of dry air, confirming that water vapor is less dense than the air it mixes with.
Complete or incomplete: INCOMPLETE

YOUR TASK
Query: [QUERY]
Answer: [ANSWER]
Complete or incomplete:"""

ANSWER_CHECK = """You are provided with a query and an answer. Your task is to determine if the answer sufficiently answers the question. Respond with SUFFICIENT if the answer is sufficient for answering the question and INSUFFICIENT if the answer is insufficient for answering the question. Respond with nothing else. You are provided with an example and then your task.

EXAMPLE
Query: Which is denser, water vapor or air?
Answer: This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation. At the same temperature and pressure, a volume of moist air will weigh less than the same volume of dry air, confirming that water vapor is less dense than the air it mixes with.
Complete or incomplete: INSUFFICIENT

YOUR TASK
Query: [QUERY]
Answer: [ANSWER]
Complete or incomplete:"""

ANSWER_WRONG_CHECK = """You are provided with a query and an answer. Your task is to determine if the answer contains factually incorrect information. Respond with CORRECT if the answer is correct and INCORRECT if the answer contains incorrect information. Respond with nothing else. You are provided with an example and then your task.

EXAMPLE
Query: Which is denser, water vapor or air?
Answer: Water vapor is denser than air. This is because the molecular weight of the primary components of air mainly nitrogen and oxygen is higher than that of water vapor. Nitrogen has a molecular weight of approximately 28 g/mol, and oxygen about 32 g/mol, whereas water vapor has a molecular weight of only about 18 g/mol. When water vapor is added to air, it actually displaces some of the heavier nitrogen and oxygen molecules, making the resulting humid air less dense than dry air. This principle is why humid air tends to rise and is a key factor in weather patterns and cloud formation. At the same temperature and pressure, a volume of moist air will weigh less than the same volume of dry air, confirming that water vapor is less dense than the air it mixes with.
Complete or incomplete: INCORRECT

YOUR TASK
Query: [QUERY]
Answer: [ANSWER]
Complete or incomplete:"""


def eval_response(query, answer, client):
    missing_info_response = client.generate(MISSING_INFO_CHECK.replace("[QUERY]", query).replace("[ANSWER]", answer))
    answer_response = client.generate(ANSWER_CHECK.replace("[QUERY]", query).replace("[ANSWER]", answer))
    answer_wrong_response = client.generate(ANSWER_WRONG_CHECK.replace("[QUERY]", query).replace("[ANSWER]", answer))
    return missing_info_response, answer_response, answer_wrong_response


def optional_eval(query, normal, missing, wrong, client):
    to_ret = {}
    for label, answer in [("normal", normal), ("missing", missing), ("wrong", wrong)]:
        missing_info_response, answer_response, answer_wrong_response = eval_response(query, answer, client)
        to_ret[f"{label}-missing"] = missing_info_response.strip() == "COMPLETE"
        to_ret[f"{label}-answer"] = answer_response.strip() == "SUFFICIENT"
        to_ret[f"{label}-answer-wrong"] = answer_wrong_response.strip() == "CORRECT"
    return to_ret


def main():
    parser = argparse.ArgumentParser(description="Spot-check quality of adversarial missing/wrong responses.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--cache_path", type=str, default=".cache")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to sample.")
    args = parser.parse_args()

    normal_file = os.path.join(args.root_dir, args.dataset, args.subset, "normal", "data.jsonl")
    missing_file = os.path.join(args.root_dir, args.dataset, args.subset, "missing", "data.jsonl")
    wrong_file = os.path.join(args.root_dir, args.dataset, args.subset, "wrong", "data.jsonl")
    cache_path = os.path.join(args.cache_path, args.dataset, args.subset, "adversarial_check", "cache.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    configs.model_name = args.model
    configs.temp = 0.2
    configs.max_tokens = 2000
    client = OpenAIAgent(cache_path=cache_path)

    count = 0
    eval_subtotals = {f"{label}-{kind}": 0 for label in ("normal", "missing", "wrong") for kind in ("missing", "answer", "answer-wrong")}
    with open(normal_file, "r") as normal, open(missing_file, "r") as missing, open(wrong_file, "r") as wrong:
        for normal_line, missing_line, wrong_line in tqdm(zip(normal, missing, wrong), total=args.limit):
            if count >= args.limit:
                break
            normal_line = json.loads(normal_line)
            missing_line = json.loads(missing_line)
            wrong_line = json.loads(wrong_line)
            query = normal_line["query"]
            normal_answer = normal_line["response"]["text"]
            missing_answer = missing_line["response"]["text"]
            wrong_answer = wrong_line["response"]["text"]

            eval_dict = optional_eval(query, normal_answer, missing_answer, wrong_answer, client)
            for key, passed in eval_dict.items():
                if passed:
                    eval_subtotals[key] += 1

            count += 1
            if count % 10 == 0:
                client.cache.save_cache()

    client.cache.save_cache()
    print(f"Out of {count} examples:")
    print(json.dumps(eval_subtotals, indent=2))


if __name__ == "__main__":
    main()
