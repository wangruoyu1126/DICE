#!/usr/bin/env python3
"""Build a multi-level demo pool from successful ReAct trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dice.llm import OpenAIChatModel
from dice.prompts import (
    QUESTION_KNOWLEDGE_TEMPLATE,
    QUESTION_TYPE_TEMPLATE,
    clean_question_knowledge,
    clean_question_type,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/demos/hotpotqa_success_traj_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("data/demos/doc_multi_level_knowledge_v2.json"))
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def extract_traj(item: dict) -> str:
    if "traj" in item:
        return item["traj"]
    if "traj_full" in item:
        return item["traj_full"]
    raise KeyError("Input item must contain either 'traj' or 'traj_full'.")


def extract_question(traj: str) -> str:
    return traj.split("Question:", 1)[1].split("\n", 1)[0].strip()


def extract_high_level_method(traj: str) -> str:
    try:
        return traj.split("Thought 1:", 1)[1].split("\n", 1)[0].strip()
    except IndexError:
        return ""


def main() -> None:
    args = parse_args()
    llm = OpenAIChatModel(model=args.model)

    with args.input.open() as f:
        source = json.load(f)
    if args.limit is not None:
        source = source[: args.limit]

    documents = []
    for item in tqdm(source):
        traj = extract_traj(item)
        question = extract_question(traj)
        question_type = clean_question_type(
            llm.generate(f"{QUESTION_TYPE_TEMPLATE} {question}", stop=["\n"])
        )
        question_knowledge = clean_question_knowledge(
            llm.generate(f"{QUESTION_KNOWLEDGE_TEMPLATE} {question}", stop=["\n"])
        )
        documents.append(
            {
                "traj_full": traj,
                "high_level_method": extract_high_level_method(traj),
                "question": question,
                "question_type": question_type,
                "question_knowledge": question_knowledge,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(documents, f, indent=2)
    print(f"Saved {len(documents)} demo documents to {args.output}")


if __name__ == "__main__":
    main()

