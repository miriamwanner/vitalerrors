"""Print a single query/response/subclaims/nuggets example for manual inspection.
Reads already-generated *-out.jsonl files; makes no LLM calls.

Example:
    python -m get_data.get_example --dataset bright --subset biology --prompt normal
"""
import argparse
import json
import os


def print_example(factscore_file, nuggets_file, new_metric_file):
    with open(factscore_file, "r", encoding="utf-8") as fs_f, \
         open(nuggets_file, "r", encoding="utf-8") as nugget_f, \
         open(new_metric_file, "r", encoding="utf-8") as new_metric_f:
        fs_line, nugget_line, new_metric_line = next(zip(fs_f, nugget_f, new_metric_f))
        fs_line = json.loads(fs_line)
        nugget_line = json.loads(nugget_line)
        new_metric_line = json.loads(new_metric_line)

        query = fs_line["query"]
        response = fs_line["response"]["text"]

        subclaims = ""
        for sent in new_metric_line["decomposition"]:
            for atom in sent["decomp"]:
                subclaims += f"- {atom['text']}\t\timportance: {atom['importance']}\t\tjudgment: {atom['judgment']}\n"

        nuggets = ""
        for n in nugget_line["nuggets"]:
            nuggets += f"- {n['text']}\t\timportance: {n['importance']}\n"

        print("------------------- Query -------------------")
        print(query)
        print("------------------- Response -------------------")
        print(response)
        print("------------------- Subclaims -------------------")
        print(subclaims)
        print("------------------- Nuggets -------------------")
        print(nuggets)


def main():
    parser = argparse.ArgumentParser(description="Print one query/response/subclaims/nuggets example.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--prompt", type=str, default="normal", choices=["normal", "missing", "wrong"])
    args = parser.parse_args()

    base_dir = os.path.join(args.root_dir, args.dataset, args.subset, args.prompt)
    factscore_file = os.path.join(base_dir, "factscore-out.jsonl")
    nuggets_file = os.path.join(base_dir, "nuggets-out.jsonl")
    new_metric_file = os.path.join(base_dir, "new-metric-out.jsonl")

    print_example(factscore_file, nuggets_file, new_metric_file)


if __name__ == "__main__":
    main()
