"""Dataset loaders and task adapters."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .env import StepResult, WikiEnv
from .metrics import exact_match, f1_score


HOTPOTQA_FILES = {
    "train": "hotpot_train_v1.1_simplified.json",
    "dev": "hotpot_dev_v1_simplified.json",
    "test": "hotpot_test_v1_simplified.json",
}

FEVER_FILES = {
    "dev": "paper_dev.jsonl",
}


@dataclass(frozen=True)
class QAExample:
    prompt: str
    answer: str | None
    index: int


class HotPotQATask:
    """HotPotQA task wrapper around the Wikipedia action environment."""

    def __init__(self, env: WikiEnv, data_dir: Path, split: str = "dev"):
        if split not in HOTPOTQA_FILES:
            raise ValueError(f"Unknown HotPotQA split {split!r}. Expected one of {sorted(HOTPOTQA_FILES)}.")
        self.env = env
        self.split = split
        self.data_path = data_dir / "hotpotqa" / HOTPOTQA_FILES[split]
        with self.data_path.open() as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def reset(self, idx: int | None = None, rng: random.Random | None = None) -> str:
        self.env.reset()
        if idx is None:
            generator = rng if rng is not None else random
            idx = generator.randrange(len(self.data))
        self.data_idx = idx
        return f"Question: {self.data[idx]['question']}"

    def step(self, action: str) -> StepResult:
        result = self.env.step(action)
        if result.done:
            row = self.data[self.data_idx]
            gt_answer = row.get("answer")
            prediction = result.info.get("answer") or ""
            em = exact_match(prediction, gt_answer) if gt_answer is not None else False
            f1 = f1_score(prediction, gt_answer)[0] if gt_answer is not None else 0.0
            result.reward = int(em)
            result.observation = f"Episode finished, reward = {result.reward}\n"
            result.info.update(
                {
                    "gt_answer": gt_answer,
                    "question": row["question"],
                    "question_idx": self.data_idx,
                    "hotpot_split": self.split,
                    "reward": em,
                    "em": em,
                    "f1": f1,
                }
            )
        return result


class FeverTask:
    """FEVER claim verification task wrapper around the Wikipedia action environment."""

    def __init__(self, env: WikiEnv, data_dir: Path, split: str = "dev"):
        if split not in FEVER_FILES:
            raise ValueError(f"Unknown FEVER split {split!r}. Expected one of {sorted(FEVER_FILES)}.")
        self.env = env
        self.split = split
        self.data_path = data_dir / "fever" / FEVER_FILES[split]
        self.data = []
        with self.data_path.open() as f:
            for line in f:
                item = json.loads(line)
                self.data.append({"question": item["claim"], "answer": item["label"]})

    def __len__(self) -> int:
        return len(self.data)

    def reset(self, idx: int | None = None, rng: random.Random | None = None) -> str:
        self.env.reset()
        if idx is None:
            generator = rng if rng is not None else random
            idx = generator.randrange(len(self.data))
        self.data_idx = idx
        return f"Claim: {self.data[idx]['question']}"

    def step(self, action: str) -> StepResult:
        result = self.env.step(action)
        if result.done:
            row = self.data[self.data_idx]
            prediction = result.info.get("answer") or ""
            em = exact_match(prediction, row["answer"])
            result.reward = int(em)
            result.observation = f"Episode finished, reward = {result.reward}\n"
            result.info.update(
                {
                    "gt_answer": row["answer"],
                    "question": row["question"],
                    "question_idx": self.data_idx,
                    "fever_split": self.split,
                    "reward": em,
                    "em": em,
                    "f1": float(em),
                }
            )
        return result

