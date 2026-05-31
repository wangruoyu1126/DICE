#!/usr/bin/env python3
"""Run DICE on HotPotQA."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from tqdm import tqdm

from dice.agent import DICEAgent
from dice.datasets import HotPotQATask
from dice.env import WikiEnv
from dice.llm import GemmaModel, OpenAIChatModel
from dice.retriever import DemoRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/hotpotqa_dice.json"))
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--demo-pool", type=Path, default=Path("data/demos/doc_multi_level_knowledge_v2.json"))
    parser.add_argument("--retrieval-mode", choices=["knowledge", "type", "none"], default="knowledge")
    parser.add_argument("--encode-field", default=None, help="Defaults to question_knowledge or question_type.")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-examples", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--llm-provider", choices=["openai", "gemma"], default="openai")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_llm(args: argparse.Namespace):
    if args.llm_provider == "openai":
        return OpenAIChatModel(model=args.llm_model or "gpt-4o-mini")
    return GemmaModel(model_name=args.llm_model or "google/gemma-2-2b-it")


def main() -> None:
    args = parse_args()
    llm = build_llm(args)
    task = HotPotQATask(WikiEnv(), data_dir=args.data_dir, split=args.split)

    retriever = None
    if args.retrieval_mode != "none":
        encode_field = args.encode_field
        if encode_field is None:
            encode_field = "question_knowledge" if args.retrieval_mode == "knowledge" else "question_type"
        retriever = DemoRetriever.from_json(
            args.demo_pool,
            encode_field=encode_field,
            model_name=args.retriever_model,
        )

    agent = DICEAgent(
        task=task,
        llm=llm,
        retriever=retriever,
        retrieval_mode=args.retrieval_mode,
        top_k=args.top_k,
    )

    indices = list(range(len(task)))
    random.Random(args.seed).shuffle(indices)
    indices = indices[args.start : args.start + args.num_examples]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    infos = []
    scores = []
    start_time = time.time()
    for idx in tqdm(indices):
        result = agent.run(idx=idx, verbose=args.verbose)
        infos.append(result.info)
        scores.append(float(result.info.get("em", 0.0)))
        running_accuracy = sum(scores) / len(scores)
        tqdm.write(
            f"idx={idx} em={result.info.get('em')} "
            f"running_em={running_accuracy:.3f} "
            f"sec/example={(time.time() - start_time) / len(scores):.1f}"
        )
        with args.output.open("w") as f:
            json.dump(infos, f, indent=2)

    print(f"Saved {len(infos)} results to {args.output}")


if __name__ == "__main__":
    main()
