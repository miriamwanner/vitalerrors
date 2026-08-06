#!/usr/bin/env python3
"""Print ready-to-run shell commands for one pipeline stage across every
dataset/subset/prompt combination in get_data/data_setup/data_info.py.

`bright` is the only dataset whose documents are inline in each condition's
data.jsonl, so it uses the plain factscore/nuggetizer scripts; every other
dataset stores documents in a shared original.jsonl and uses the _wh variant.

Examples:
    python -m scripts.print_commands factscore
    python -m scripts.print_commands nuggets --model gpt-4o
    python -m scripts.print_commands vital
    python -m scripts.print_commands normal_response      # get_data stage, no --prompt loop
    python -m scripts.print_commands adversarial_response
"""
import argparse

from get_data.data_setup.data_info import datasets, subsets, gen_type

PROMPTS = [p for p in gen_type if p != "normal"]  # ["missing", "wrong"]


def factscore_command(dataset, subset, prompt, model):
    script = "factscore.run_factscore" if dataset == "bright" else "factscore.run_factscore_wh"
    return f"python -m {script} --dataset {dataset} --subset {subset} --prompt {prompt} --model {model}"


def nuggets_command(dataset, subset, prompt, model):
    script = "nuggetizer.run_nuggetizer" if dataset == "bright" else "nuggetizer.run_nuggetizer_wh"
    return f"python -m {script} --dataset {dataset} --subset {subset} --prompt {prompt} --model {model}"


def vital_command(dataset, subset, prompt, model):
    return f"python -m vital.run_vital_metric --dataset {dataset} --subset {subset} --prompt {prompt} --model {model}"


def normal_response_command(dataset, subset, prompt, model):
    return f"python -m get_data.get_normal_response --dataset {dataset} --subset {subset} --model {model}"


def adversarial_response_command(dataset, subset, prompt, model):
    return f"python -m get_data.get_adversarial_response --dataset {dataset} --subset {subset} --prompt {prompt} --model {model}"


STAGES = {
    "factscore": (factscore_command, gen_type),
    "nuggets": (nuggets_command, gen_type),
    "vital": (vital_command, gen_type),
    "normal_response": (normal_response_command, ["normal"]),  # no --prompt loop, run once
    "adversarial_response": (adversarial_response_command, PROMPTS),
}


def main():
    parser = argparse.ArgumentParser(description="Print ready-to-run commands for a pipeline stage.")
    parser.add_argument("stage", choices=list(STAGES.keys()))
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--dataset", type=str, default=None, choices=datasets, help="Restrict to one dataset.")
    args = parser.parse_args()

    command_fn, prompts = STAGES[args.stage]
    target_datasets = [args.dataset] if args.dataset else datasets

    for d in target_datasets:
        for s in subsets[d]:
            for p in prompts:
                print(command_fn(d, s, p, args.model))


if __name__ == "__main__":
    main()
