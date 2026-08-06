"""Core VITAL metric logic: subclaim/nugget importance labeling, alignment,
and weighted precision/recall/F1 (Section 3.1 of the paper).

This module is LLM-client agnostic: `DataProcessor` just needs an object with
a `.generate(prompt) -> str` method (e.g. `factscore.openai_agent.OpenAIAgent`).
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .prompts import NUGGET_SUBCLAIM_ALIGNMENT, SUBCLAIM_QUERY_IMPORTANCE


@dataclass
class MetricWeights:
    """Configuration for importance weights."""
    vital: float
    okay: float
    less_important: float


@dataclass
class AlignmentResult:
    """Result of nugget-subclaim alignment."""
    N2S: Dict[str, List[str]]
    S2N: Dict[str, str]
    id_to_str_dict: Dict[str, str]
    str_to_id_dict: Dict[str, str]
    nugget2importance: Dict[str, str]


@dataclass
class EvaluationResult:
    """Result of subclaim evaluation."""
    subclaim_eval: List[Dict[str, Any]]
    subclaim2importance: Dict[str, str]


@dataclass
class MetricScores:
    """VITAL_PREC-style scores: overall weighted precision/recall/F1 plus a
    precision/count breakdown per importance tier."""
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    vital_precision: float
    vital_subclaims: float
    okay_precision: float
    okay_subclaims: float
    less_important_precision: float
    less_important_subclaims: float


@dataclass
class DecayMetricScores:
    """Linear-decay-weighted scores (Appendix B), overall and restricted to
    the top-k highest-ranked subclaims/nuggets."""
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float
    weighted_precision_topk: float
    weighted_recall_topk: float
    weighted_f1_topk: float


class TextProcessor:
    """Handles text processing and parsing operations."""

    @staticmethod
    def extract_mappings(text: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """Extract N2S and S2N mappings from alignment text."""
        N2S = {}
        S2N = {}

        pattern = re.compile(r"\(N(\d+),\s*(None|\[?[S\d,\s]+\]?)\)")

        for match in pattern.finditer(text):
            n_id = f"N{match.group(1)}"
            s_part = match.group(2).strip()

            if s_part == "None":
                s_ids = []
            elif s_part.startswith("["):
                s_ids = [s.strip() for s in s_part[1:-1].split(",")]
            else:
                s_ids = [s_part]

            N2S[n_id] = s_ids
            for s_id in s_ids:
                S2N[s_id] = n_id

        return N2S, S2N

    @staticmethod
    def parse_eval(text: str) -> Dict[str, bool]:
        """Parse evaluation results from text."""
        result = {}
        pattern = re.compile(r"\[(S\d+)]\s*(.+?):\s*(Supported|Unsupported)")

        for match in pattern.finditer(text):
            index = match.group(1)
            label = match.group(3)
            result[index] = label == "Supported"

        return result


class DataProcessor:
    """Handles nugget/subclaim alignment, importance labeling, and evaluation assembly."""

    def __init__(self, client):
        self.client = client
        self.text_processor = TextProcessor()

    def get_alignment(self, nugget_line: Dict, factscore_line: Dict) -> AlignmentResult:
        """Get alignment between nuggets and subclaims (used only to report N2S
        in the output; importance labels come from get_subclaim_importance_no_alignment)."""
        id_to_str_dict = {}
        str_to_id_dict = {}
        nugget2importance = {}

        nuggets_str = ""
        for i, n in enumerate(nugget_line["nuggets"]):
            nugget_id = f"N{i}"
            nugget2importance[nugget_id] = n["importance"]
            nuggets_str += f"{nugget_id}: {n['text']}\n"
            id_to_str_dict[nugget_id] = n["text"]

        subclaims_str = ""
        subclaim_counter = 1
        for sent in factscore_line["decomposition"]:
            for atom in sent["decomp"]:
                subclaim_id = f"S{subclaim_counter}"
                subclaims_str += f"{subclaim_id}: {atom['text']}\n"
                id_to_str_dict[subclaim_id] = atom["text"]
                str_to_id_dict[atom["text"] + "_" + str(subclaim_counter)] = subclaim_id
                subclaim_counter += 1

        prompt = NUGGET_SUBCLAIM_ALIGNMENT.replace("[NUGGETS]", nuggets_str.strip()).replace("[SUBCLAIMS]", subclaims_str.strip())
        response = self.client.generate(prompt)
        N2S, S2N = self.text_processor.extract_mappings(response)

        return AlignmentResult(N2S, S2N, id_to_str_dict, str_to_id_dict, nugget2importance)

    def get_subclaim_importance_no_alignment(self, query: str, id_to_str_dict: Dict[str, str]) -> Tuple[Dict[str, str], List[str]]:
        """Rank and label subclaims by query-importance (Table 8 SUBCLAIM_QUERY_IMPORTANCE prompt)."""
        subclaim2importance = {}

        subclaim_str = ""
        subclaim_count = 0
        subclaim_ids = []
        for subclaim_id in id_to_str_dict.keys():
            if subclaim_id.startswith("S"):
                subclaim_str += "[" + subclaim_id + "] " + id_to_str_dict[subclaim_id] + "\n"
                subclaim_count += 1
                subclaim_ids.append(subclaim_id)
        prompt = SUBCLAIM_QUERY_IMPORTANCE.replace("[QUERY]", query).replace("[SUBCLAIMS]", subclaim_str)

        subclaim_order = []
        remaining_tries = 5
        while remaining_tries > 0:
            try:
                response = self.client.generate(prompt)

                pattern = re.compile(r'\[S(\d+)\]\s*(.+?):\s*"([^"]+)"')
                subclaim_order = []
                for match in pattern.finditer(response):
                    subclaim_id = f"S{match.group(1)}"
                    subclaim_importance = match.group(3)
                    assert subclaim_importance in ["vital", "okay", "less-important"]
                    if subclaim_id in subclaim_ids:
                        subclaim2importance[subclaim_id] = subclaim_importance
                        subclaim_order.append(subclaim_id)

                for sid in subclaim_ids:
                    if sid not in subclaim2importance:
                        # if not mentioned in the ordering, fall back on "less important"
                        subclaim2importance[sid] = "less-important"
                        subclaim_order.append(sid)
                assert subclaim_count == len(subclaim2importance)
                break
            except AssertionError:
                print("Assertion error, trying again")
            remaining_tries -= 1

        return subclaim2importance, subclaim_order

    def get_eval(self, factscore_line: Dict, subclaim2importance: Dict[str, str], str_to_id_dict: Dict[str, str]) -> List[Dict[str, Any]]:
        """Attach importance labels to the FactScore decomposition/judgments."""
        subclaim_eval = []
        subclaim_counter = 1
        for sent in factscore_line["decomposition"]:
            temp_dict = {"sentence": sent["sentence"], "decomp": []}
            for atom in sent["decomp"]:
                subclaim_id = str_to_id_dict[atom["text"] + "_" + str(subclaim_counter)]
                subclaim_counter += 1
                importance = subclaim2importance.get(subclaim_id, "less-important")
                temp_dict["decomp"].append({
                    "id": subclaim_id,
                    "text": atom["text"],
                    "judgment": atom["judgment"],
                    "importance": importance,
                })

            subclaim_eval.append(temp_dict)

        return subclaim_eval


class MetricCalculator:
    """Calculates weighted precision, recall, and F1 scores."""

    def calculate_scores(self, subclaim_eval: List[Dict[str, Any]], nuggets_line: Dict,
                          weights: MetricWeights) -> MetricScores:
        """Calculate weighted precision, recall, and F1 scores."""

        precision_counters = {
            "vital": {"supported": 0, "total": 0},
            "okay": {"supported": 0, "total": 0},
            "less-important": {"supported": 0, "total": 0},
        }

        for sent in subclaim_eval:
            for atom in sent["decomp"]:
                importance = atom["importance"]
                precision_counters[importance]["total"] += 1
                if atom["judgment"]:
                    precision_counters[importance]["supported"] += 1

        precision_numerator = (
            weights.vital * precision_counters["vital"]["supported"] +
            weights.okay * precision_counters["okay"]["supported"] +
            weights.less_important * precision_counters["less-important"]["supported"]
        )
        precision_denominator = (
            weights.vital * precision_counters["vital"]["total"] +
            weights.okay * precision_counters["okay"]["total"] +
            weights.less_important * precision_counters["less-important"]["total"]
        )
        weighted_precision = precision_numerator / precision_denominator if precision_denominator > 0 else 0

        vital_precision = precision_counters["vital"]["supported"] / precision_counters["vital"]["total"] if precision_counters["vital"]["total"] > 0 else 0
        vital_subclaims = precision_counters["vital"]["total"]
        okay_precision = precision_counters["okay"]["supported"] / precision_counters["okay"]["total"] if precision_counters["okay"]["total"] > 0 else 0
        okay_subclaims = precision_counters["okay"]["total"]
        less_important_precision = precision_counters["less-important"]["supported"] / precision_counters["less-important"]["total"] if precision_counters["less-important"]["total"] > 0 else 0
        less_important_subclaims = precision_counters["less-important"]["total"]

        recall_counters = {"vital": {"supported": 0, "total": 0}, "okay": {"supported": 0, "total": 0}}
        for support, nugget in zip(nuggets_line["response"]["nugget-assignment"], nuggets_line["nuggets"]):
            importance = nugget["importance"]
            if importance in recall_counters:
                recall_counters[importance]["total"] += 1
                if support != "not_support":
                    recall_counters[importance]["supported"] += 1

        recall_numerator = (
            weights.vital * recall_counters["vital"]["supported"] +
            weights.okay * recall_counters["okay"]["supported"]
        )
        recall_denominator = (
            weights.vital * recall_counters["vital"]["total"] +
            weights.okay * recall_counters["okay"]["total"]
        )
        weighted_recall = recall_numerator / recall_denominator if recall_denominator > 0 else 0

        if weighted_precision + weighted_recall > 0:
            weighted_f1 = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)
        else:
            weighted_f1 = 0

        return MetricScores(weighted_precision, weighted_recall, weighted_f1, vital_precision, vital_subclaims,
                             okay_precision, okay_subclaims, less_important_precision, less_important_subclaims)

    def linear_decay_weighting(self, subclaim_eval: List[Dict[str, Any]], nuggets_line: Dict,
                                subclaim_order: List[str] = None) -> DecayMetricScores:
        """Linear-decay-weighted precision/recall/F1 (Appendix B), with a top-k variant."""
        k = 5

        if subclaim_order is None:  # if no ordering provided, fall back to response order
            subclaim_judgments = []
            for sent in subclaim_eval:
                for atom in sent["decomp"]:
                    subclaim_judgments.append(1 if atom["judgment"] else 0)
        else:
            subclaimid2judgment = {}
            for sent in subclaim_eval:
                for atom in sent["decomp"]:
                    subclaimid2judgment[atom["id"]] = 1 if atom["judgment"] else 0
            subclaim_judgments = [subclaimid2judgment[s] for s in subclaim_order]

        weighted_precision = self.linear_decay_scoring_helper(subclaim_judgments)
        weighted_precision_topk = self.linear_decay_scoring_helper(subclaim_judgments[:k])

        nugget_judgments = [
            1 if support != "not_support" else 0
            for support in nuggets_line["response"]["nugget-assignment"]
        ]
        weighted_recall = self.linear_decay_scoring_helper(nugget_judgments)
        weighted_recall_topk = self.linear_decay_scoring_helper(nugget_judgments[:k])

        weighted_f1 = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if weighted_precision + weighted_recall > 0 else 0
        weighted_f1_topk = (2 * weighted_precision_topk * weighted_recall_topk) / (weighted_precision_topk + weighted_recall_topk) if weighted_precision_topk + weighted_recall_topk > 0 else 0

        return DecayMetricScores(weighted_precision, weighted_recall, weighted_f1,
                                  weighted_precision_topk, weighted_recall_topk, weighted_f1_topk)

    def linear_decay_scoring_helper(self, judgments):
        num = len(judgments)
        if num == 0:
            return 0
        weights = [(num - i) / i for i in range(1, num + 1)]

        numerator = sum(w * correct for w, correct in zip(weights, judgments))
        total_weight = sum(weights)

        return numerator / total_weight if total_weight > 0 else 0
