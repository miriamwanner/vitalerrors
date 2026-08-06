#!/usr/bin/env python3
"""Same as run_nuggetizer.py, but for datasets whose documents live in a shared
`original.jsonl` at the subset level rather than inline in each condition's
data.jsonl (currently only needed for wildhallucinations). Documents over
50,000 characters are dropped to keep prompts a reasonable size.

Example:
    python -m nuggetizer.run_nuggetizer_wh --dataset wildhallucinations --subset geographic --prompt normal
"""
import argparse
import json
import os

from tqdm import tqdm

from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores

MAX_DOC_CHARS = 50000


def get_document_assignment_metrics(query, doc, scored_nuggets, nuggetizer) -> None:
    """Given nuggets and a document, get level of support and scores."""
    to_ret = {"id": doc.docid, "text": doc.segment, "nugget-assignment": []}
    assigned_nuggets = nuggetizer.assign(query.text, doc.segment, scored_nuggets)
    for nugget in assigned_nuggets:
        to_ret["nugget-assignment"].append(nugget.assignment)

    nugget_list = [
        {"text": n.text, "importance": n.importance, "assignment": n.assignment}
        for n in assigned_nuggets
    ]
    metrics = calculate_nugget_scores(query.qid, nugget_list)
    to_ret["strict-vital-score"] = metrics.strict_vital_score
    to_ret["strict-all-score"] = metrics.strict_all_score
    to_ret["vital-score"] = metrics.vital_score
    to_ret["all-score"] = metrics.all_score

    return to_ret


def create_request(query: str, docs: list) -> Request:
    query = Query(qid="0", text=query)
    documents = [Document(docid=d["id"], segment=d["text"]) for d in docs]
    return Request(query=query, documents=documents)


def process_request(request: Request, generation: dict, model: str, use_azure_openai: bool, log_level: int, cache_path: str):
    """Process a request through the nuggetizer pipeline."""
    nuggetizer = Nuggetizer(model=model, use_azure_openai=use_azure_openai, log_level=log_level, cache_path=cache_path)

    scored_nuggets = nuggetizer.create(request)
    nuggets_list = [
        {"id": str(i), "text": nugget.text, "importance": nugget.importance}
        for i, nugget in enumerate(scored_nuggets)
    ]

    document_list = []  # per-source-document assignment is skipped for time; only the response is assigned below

    generation_document = Document(docid=generation["id"], segment=generation["text"])
    response_assignment = get_document_assignment_metrics(request.query, generation_document, scored_nuggets, nuggetizer)

    return nuggets_list, document_list, response_assignment


def main():
    parser = argparse.ArgumentParser(description="Run nuggetizer over a dataset that stores documents in original.jsonl.")
    parser.add_argument("--root_dir", type=str, default="data")
    parser.add_argument("--dataset", type=str, required=True,
                         choices=["bright", "factscore", "wildhallucinations", "hotpotqa", "naturalquestions", "triviaqa"])
    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--prompt", type=str, required=True, choices=["normal", "missing", "wrong"])
    parser.add_argument("--cache_path", type=str, default=".cache")
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--use_azure_openai", action="store_true", help="Use Azure OpenAI instead of the OpenAI API.")
    parser.add_argument("--log_level", type=int, default=0, help="Log level")
    args = parser.parse_args()

    in_file = os.path.join(args.root_dir, args.dataset, args.subset, args.prompt, "data.jsonl")
    in_file_docs = os.path.join(args.root_dir, args.dataset, args.subset, "original.jsonl")
    out_file = os.path.join(args.root_dir, args.dataset, args.subset, args.prompt, "nuggets-out.jsonl")
    cache_path = os.path.join(args.cache_path, args.dataset, args.subset, "nuggets")
    os.makedirs(cache_path, exist_ok=True)

    with open(in_file, "r") as input, open(in_file_docs, "r") as input_docs, open(out_file, "w") as output:
        for line, line_docs in tqdm(zip(input, input_docs)):
            line = json.loads(line)
            line_docs = json.loads(line_docs)

            query = line["query"]
            generation = line["response"]
            docs = []
            count = 0
            for d in line_docs["documents"]:
                if len(d["text"]) < MAX_DOC_CHARS:
                    docs.append({"text": d["text"], "id": str(count)})
                    count += 1
            request = create_request(query, docs)

            nuggets_list, document_list, response_assignment = process_request(
                request, generation, args.model, args.use_azure_openai, args.log_level, cache_path
            )

            for_json = {
                "query": query,
                "response": response_assignment,
                "nuggets": nuggets_list,
                "documents": document_list,
            }
            output.write(json.dumps(for_json) + "\n")

    print(f"Wrote results to {out_file}")


if __name__ == "__main__":
    main()
