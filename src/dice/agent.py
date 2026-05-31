"""DICE agent loop."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from .datasets import HotPotQATask
from .llm import LanguageModel
from .prompts import (
    QUESTION_KNOWLEDGE_TEMPLATE,
    QUESTION_TYPE_TEMPLATE,
    REACT_INSTRUCTION,
    clean_question_knowledge,
    clean_question_type,
)
from .retriever import DemoRetriever, format_demo_trajectory


@dataclass
class AgentResult:
    reward: int
    info: dict


class DICEAgent:
    """Dynamic in-context example retrieval plus ReAct execution."""

    def __init__(
        self,
        task: HotPotQATask,
        llm: LanguageModel,
        retriever: DemoRetriever | None = None,
        retrieval_mode: str = "knowledge",
        top_k: int = 6,
        max_steps: int = 7,
        prompt: str = REACT_INSTRUCTION,
        retry_limit: int = 10,
    ):
        if retrieval_mode not in {"knowledge", "type", "none"}:
            raise ValueError("retrieval_mode must be one of: knowledge, type, none.")
        self.task = task
        self.llm = llm
        self.retriever = retriever
        self.retrieval_mode = retrieval_mode
        self.top_k = top_k
        self.max_steps = max_steps
        self.prompt = prompt
        self.retry_limit = retry_limit

    def _extract_retrieval_query(self, question: str) -> str:
        if self.retrieval_mode == "knowledge":
            raw = self.llm.generate(f"{QUESTION_KNOWLEDGE_TEMPLATE} {question}", stop=["\n"])
            return clean_question_knowledge(raw)
        if self.retrieval_mode == "type":
            raw = self.llm.generate(f"{QUESTION_TYPE_TEMPLATE} {question}", stop=["\n"])
            return clean_question_type(raw)
        return ""

    def _retrieve_examples(self, question: str) -> tuple[str, str]:
        if self.retrieval_mode == "none" or self.retriever is None:
            return "", ""

        query = self._extract_retrieval_query(question)
        retrieved = self.retriever.retrieve(query, top_k=self.top_k)
        examples = "".join(format_demo_trajectory(item.document) for item in retrieved)
        return query, examples

    def _environment_step(self, action: str):
        attempts = 0
        while True:
            try:
                return self.task.step(action)
            except requests.exceptions.Timeout:
                attempts += 1
                if attempts >= self.retry_limit:
                    raise

    def run(self, idx: int | None = None, verbose: bool = False) -> AgentResult:
        question = self.task.reset(idx=idx)
        retrieval_query, retrieved_examples = self._retrieve_examples(question)
        prompt = self.prompt + retrieved_examples + question + "\n"

        n_calls = 0
        n_badcalls = 0
        done = False
        result = None

        if verbose:
            print(idx, question)
            if retrieval_query:
                print(f"Retrieval query: {retrieval_query}")

        for step_idx in range(1, self.max_steps + 1):
            n_calls += 1
            thought_action = self.llm.generate(prompt + f"Thought {step_idx}:", stop=[f"\nObservation {step_idx}:"])
            try:
                thought, action = thought_action.strip().split(f"\nAction {step_idx}: ", 1)
            except ValueError:
                n_badcalls += 1
                n_calls += 1
                thought = thought_action.strip().split("\n")[0]
                action = self.llm.generate(
                    prompt + f"Thought {step_idx}: {thought}\nAction {step_idx}:",
                    stop=["\n"],
                ).strip()

            if not action:
                action = "finish[]"
            normalized_action = action[0].lower() + action[1:]
            result = self._environment_step(normalized_action)
            observation = result.observation.replace("\n", "")
            step_text = (
                f"Thought {step_idx}: {thought}\n"
                f"Action {step_idx}: {action}\n"
                f"Observation {step_idx}: {observation}\n"
            )
            prompt += step_text

            if verbose:
                print(step_text)
            if result.done:
                done = True
                break

        if not done:
            result = self._environment_step("finish[]")

        assert result is not None
        result.info.update(
            {
                "n_calls": n_calls,
                "n_badcalls": n_badcalls,
                "traj": prompt,
                "retrieval_mode": self.retrieval_mode,
                "retrieval_query": retrieval_query,
            }
        )
        return AgentResult(reward=result.reward, info=result.info)

