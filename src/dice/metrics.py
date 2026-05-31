"""Evaluation utilities for QA-style tasks."""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(char for char in value if char not in exclude)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Return token-level F1, precision, and recall."""

    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    zero = (0.0, 0.0, 0.0)

    if (
        normalized_prediction in {"yes", "no", "noanswer"}
        and normalized_prediction != normalized_ground_truth
    ):
        return zero
    if (
        normalized_ground_truth in {"yes", "no", "noanswer"}
        and normalized_prediction != normalized_ground_truth
    ):
        return zero

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return zero

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(ground_truth_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Return normalized exact match."""

    return normalize_answer(prediction) == normalize_answer(ground_truth)

