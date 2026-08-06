import string
from .openai_agent import OpenAIAgent
from . import configs
import json
import random
from rank_bm25 import BM25Okapi
import numpy as np

class FactScorer:
    def __init__(self, cache_path="/llm_cache/factscore.pkl", openai_agent=None):
        # Examples (demonstrations) that is used in prompt generation
        self.demons = self.load_demons()
        # To interact with OpenAI APIs
        if openai_agent:
            self.openai_agent = openai_agent
        else:
            self.openai_agent = OpenAIAgent(cache_path)


    def load_demons(self):
        """
        Load examples (demonstrations) from a JSON file.
        This will be used in the prompt generation.

        Returns:
            list: A list of examples (demonstrations).
        """
        with open(configs.fact_scorer_demons_path, "r") as file:
            demons = json.load(file)

        return demons

    def get_instructions(self) -> str:
        """
        Prepare instructions for the prompt generation.
        Instructions include the examples given in the fact_scorer_demons.json file.

        Returns:
            str: The instructions for the prompt generation.
        """

        # instructions = "Evaluate the truthfulness of the statement based solely on the provided context.\n\n"
        instructions = "Instruction:\nOnly consider the statement true if it can be directly verified by the information in the context. If the information in the statement cannot be found in the context or differs from it, label it as false.\n\n"
        true_example = self.demons[0]
        false_example = random.choice(self.demons[1:])

        for demon in [true_example, false_example]:
            instructions += f"Context:\n{demon['knowledge_source']}\n"
            instructions += f"Statement:\n{demon['fact']} True or False?\n"
            instructions += f"Output:\n{demon['is_supported']}\n\n"
            # TODO: add reason (+change parsing)
            # instructions += f"Reason:\n{demon['reason']}\n\n"

        return instructions

    def get_score(self, facts: list, knowledge_source: str) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (str): The knowledge source to be used for scoring.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        count = 0
        for atom in facts:
            atom = atom.strip()

            # Prompt that will be sent to GPT
            prompt = self.get_instructions()
            prompt += f"Context:\n{knowledge_source}\n"
            prompt += f"Statement:\n{atom} True or False?\n"
            prompt += "Output:\n"

            output = self.openai_agent.generate(prompt)

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )
            count += 1
            if count % 10 == 0:
                self.openai_agent.cache.save_cache()
        self.openai_agent.cache.save_cache()
        return decisions


    def get_score_with_retrieval(self, facts: list, knowledge_source: list) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (list): A list of documents to be retrieved from to score each fact.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        count = 0
        for atom in facts:
            atom = atom.strip()

            # retrieve passages
            if knowledge_source[0] != "":
                passages = get_bm25_passages(atom, knowledge_source, k=5)
                context = ""
                for psg in reversed(passages):
                    context += psg + "\n\n"
            else: 
                context = ""

            # # Prompt that will be sent to GPT
            # prompt = f"Context:\n{context}"
            # prompt += f"Statement:\n{atom} True or False?\n"
            # prompt += "Output:\n"

            # Prompt that will be sent to GPT
            prompt = f"Answer the question based on the given context. Reply with True or False and nothing else.\n\nContext: {context}\n\n"
            prompt += f"Input:\n{atom} True or False?\n"
            prompt += "Output:"


            output = self.openai_agent.generate(prompt)

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )
            # print({"fact": atom, "is_supported": is_supported, "output": output})
            count += 1
            if count % 10 == 0:
                self.openai_agent.cache.save_cache()
        self.openai_agent.cache.save_cache()

        return decisions


    def get_score_no_docs(self, facts: list) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (list): A list of documents to be retrieved from to score each fact.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        count = 0
        for atom in facts:
            atom = atom.strip()

            # Prompt that will be sent to GPT
            prompt = f"You are a fact-checking agent. Your task is to assess whether a given statement is factually accurate. Respond only with True or False. Do not provide explanations, reasoning, or any additional text.\n\n"
            prompt += f"Input: {atom}\nOutput:"

            output = self.openai_agent.generate(prompt)

            # print(prompt)
            # print(output)
            # raise NotImplementedError

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )
            # print({"fact": atom, "is_supported": is_supported, "output": output})
            count += 1
            if count % 10 == 0:
                self.openai_agent.cache.save_cache()
        self.openai_agent.cache.save_cache()
        return decisions

    def get_score_grounded(self, facts: list, knowledge_source: list) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (list): A list of documents to be retrieved from to score each fact.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        count = 0
        for atom in facts:
            atom = atom.strip()

            # retrieve passages
            passages = get_bm25_passages(atom, knowledge_source, k=5)
            context = ""
            for psg in reversed(passages):
                context += psg + "\n\n"

            # # Prompt that will be sent to GPT
            # prompt = f"Context:\n{context}"
            # prompt += f"Statement:\n{atom} True or False?\n"
            # prompt += "Output:\n"

            # Prompt that will be sent to GPT
            # prompt = f"Answer the question based on the given context. Reply with True or False and nothing else.\n\nContext: {context}\n\n"
            # prompt += f"Input:\n{atom} True or False?\n"
            # prompt += "Output:"

            prompt = f"You are a grounding agent. Your task is to determine whether the input statement is grounded or ungrounded based solely on the information provided in the grounding.\n- Do not use any background knowledge, external information, or prior training.\n- Only consider the statement True if it can be directly verified by the grounding information.\n- If the grounding does not contain sufficient evidence to determine the grounding of the input, respond with False.\n- Respond only with True or False.\n- Do not provide explanations, reasoning, or any additional text."
            prompt += f"\n\nGrounding: {context}"
            prompt += f"\n\nInput: {atom}"
            prompt += "\nOutput:"

            output = self.openai_agent.generate(prompt)

            # print(prompt)
            # print(output)
            # raise NotImplementedError

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )
            # print({"fact": atom, "is_supported": is_supported, "output": output})
            count += 1
            if count % 10 == 0:
                self.openai_agent.cache.save_cache()
        self.openai_agent.cache.save_cache()

        return decisions



    def get_score_nugget_doc(self, facts: list, nugget_doc: str) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (list): A list of documents to be retrieved from to score each fact.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        count = 0
        for atom in facts:
            atom = atom.strip()

            # retrieve passages
            context = nugget_doc

            # Prompt that will be sent to GPT
            prompt = f"Answer the question based on the given context. Reply with True or False and nothing else.\n\nContext: {context}\n\n"
            prompt += f"Input:\n{atom} True or False?\n"
            prompt += "Output:"


            output = self.openai_agent.generate(prompt)

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )
            # print({"fact": atom, "is_supported": is_supported, "output": output})
            count += 1
            if count % 10 == 0:
                self.openai_agent.cache.save_cache()
        self.openai_agent.cache.save_cache()

        return decisions



    def get_score_nugget_doc_grounded(self, facts: list, nugget_doc: str) -> list:
        """
        Calculates the score of each atomic fact based on the knowledge source.
        The score is caclulated by using the OpenAI API.

        Args:
            facts (list): A list of atomic  to be scored.
            knowledge_source (list): A list of documents to be retrieved from to score each fact.

        Returns:
            list: A list of dictionaries containing the atomic fact and its score.
        """

        decisions = []

        count = 0
        for atom in facts:
            atom = atom.strip()

            context = nugget_doc

            prompt = f"You are a grounding agent. Your task is to determine whether the input statement is grounded or ungrounded based solely on the information provided in the grounding.\n- Do not use any background knowledge, external information, or prior training.\n- Only consider the statement True if it can be directly verified by the grounding information.\n- If the grounding does not contain sufficient evidence to determine the grounding of the input, respond with False.\n- Respond only with True or False.\n- Do not provide explanations, reasoning, or any additional text."
            prompt += f"\n\nGrounding: {context}"
            prompt += f"\n\nInput: {atom}"
            prompt += "\nOutput:"

            output = self.openai_agent.generate(prompt)

            # print(prompt)
            # print(output)
            # raise NotImplementedError

            generated_answer = output.lower()
            is_supported = None

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index(
                        "true"
                    ) > generated_answer.index("false")
            else:
                is_supported = all(
                    [
                        keyword
                        not in generated_answer.lower()
                        .translate(str.maketrans("", "", string.punctuation))
                        .split()
                        for keyword in [
                            "not",
                            "cannot",
                            "unknown",
                            "information",
                        ]
                    ]
                )

            decisions.append(
                {"fact": atom, "is_supported": is_supported, "output": output}
            )
            # print({"fact": atom, "is_supported": is_supported, "output": output})
            count += 1
            if count % 10 == 0:
                self.openai_agent.cache.save_cache()
        self.openai_agent.cache.save_cache()

        return decisions



def get_bm25_passages(query, passages, k=5):
    bm25 = BM25Okapi(passages)
    scores = bm25.get_scores(query.split())
    indices = np.argsort(-scores)[:k]
    return [passages[i] for i in indices]

