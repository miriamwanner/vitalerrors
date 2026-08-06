"""Compute VITAL precision/recall by aligning FactScore subclaims with
importance-ranked nuggets (Section 3.1 of the paper).

Example:
    python -m vital.run_vital_metric --dataset bright --subset biology --prompt normal \\
        --vital-weight 1.0 --okay-weight 0.5 --less-important-weight 0.1
"""
import argparse
import json
import os
from typing import Tuple

from tqdm import tqdm

from factscore import configs
from factscore.openai_agent import OpenAIAgent

from .metrics import DataProcessor, MetricCalculator, MetricWeights

import logging
logging.disable(logging.WARNING)


class MetricProcessor:
    """Orchestrates alignment, importance labeling, evaluation, and scoring for one entry."""

    def __init__(self, client, weights: MetricWeights):
        self.client = client
        self.weights = weights
        self.data_processor = DataProcessor(client)
        self.metric_calculator = MetricCalculator()

    def process_single_entry(self, data_line, nuggets_line, factscore_line):
        alignment = self.data_processor.get_alignment(nuggets_line, factscore_line)

        subclaim2importance, subclaim_order = self.data_processor.get_subclaim_importance_no_alignment(
            factscore_line["query"], alignment.id_to_str_dict
        )

        subclaim_eval = self.data_processor.get_eval(factscore_line, subclaim2importance, alignment.str_to_id_dict)

        scores = self.metric_calculator.calculate_scores(subclaim_eval, nuggets_line, self.weights)
        linear_decay_scores = self.metric_calculator.linear_decay_weighting(subclaim_eval, nuggets_line, subclaim_order)

        return {
            "query": data_line["query"],
            "response": data_line["response"],
            "decomposition": subclaim_eval,
            "N2S": alignment.N2S,
            "subclaim-importance-order": subclaim_order,
            "scores": {
                "weighted-precision": scores.weighted_precision,
                "weighted-recall": scores.weighted_recall,
                "weighted-f1": scores.weighted_f1,
                "vital-precision": scores.vital_precision,
                "vital-subclaims": scores.vital_subclaims,
                "okay-precision": scores.okay_precision,
                "okay-subclaims": scores.okay_subclaims,
                "less-important-precision": scores.less_important_precision,
                "less-important-subclaims": scores.less_important_subclaims,
                "linear-decay-precision": linear_decay_scores.weighted_precision,
                "linear-decay-recall": linear_decay_scores.weighted_recall,
                "linear-decay-f1": linear_decay_scores.weighted_f1,
                "linear-decay-precision-topk": linear_decay_scores.weighted_precision_topk,
                "linear-decay-recall-topk": linear_decay_scores.weighted_recall_topk,
                "linear-decay-f1-topk": linear_decay_scores.weighted_f1_topk,
            },
        }


def get_file_paths(root_dir: str, dataset: str, subset: str, prompt: str, cache_path: str) -> Tuple[str, str, str, str, str]:
    base_dir = os.path.join(root_dir, dataset, subset, prompt)

    in_file = os.path.join(base_dir, "data.jsonl")
    factscore_file = os.path.join(base_dir, "factscore-out.jsonl")
    nuggets_file = os.path.join(base_dir, "nuggets-out.jsonl")
    out_file = os.path.join(base_dir, "new-metric-out.jsonl")
    cache_file = os.path.join(cache_path, dataset, subset, prompt, "vital_metric.pkl")

    return in_file, factscore_file, nuggets_file, out_file, cache_file


def main():
    parser = argparse.ArgumentParser(description="Compute VITAL precision/recall over a dataset/subset/prompt condition.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True, choices=["normal", "missing", "wrong"])
    parser.add_argument("--cache_path", type=str, default=".cache")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--vital-weight", type=float, default=1.0)
    parser.add_argument("--okay-weight", type=float, default=0.5)
    parser.add_argument("--less-important-weight", type=float, default=0.1)
    args = parser.parse_args()

    weights = MetricWeights(args.vital_weight, args.okay_weight, args.less_important_weight)
    in_file, factscore_file, nuggets_file, out_file, cache_file = get_file_paths(
        args.root_dir, args.dataset, args.subset, args.prompt, args.cache_path
    )
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    configs.model_name = args.model
    configs.temp = 0.2
    configs.max_tokens = 4000
    client = OpenAIAgent(cache_path=cache_file)

    processor = MetricProcessor(client, weights)

    with open(in_file, "r") as data_reader, \
         open(nuggets_file, "r") as nuggets_reader, \
         open(factscore_file, "r") as factscore_reader, \
         open(out_file, "w") as writer:

        count = 0
        for data_line, nuggets_line, factscore_line in tqdm(zip(data_reader, nuggets_reader, factscore_reader)):
            data_line = json.loads(data_line)
            nuggets_line = json.loads(nuggets_line)
            factscore_line = json.loads(factscore_line)

            result = processor.process_single_entry(data_line, nuggets_line, factscore_line)
            writer.write(json.dumps(result) + "\n")

            if count % 10 == 0:
                client.cache.save_cache()
            count += 1

        client.cache.save_cache()

    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
