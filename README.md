# DICE: Dynamic In-Context Example Selection

This repository contains cleaned code and data for the paper `DICE: Dynamic In-Context Example Selection in LLM Agents via Efficient Knowledge Transfer`.

DICE retrieves demonstrations dynamically at each agent run. For a current question/context, a knowledge extractor first summarizes the general reasoning knowledge needed for the task. The retriever embeds that extracted knowledge and selects the closest demonstrations from a demo pool. The selected trajectories are then inserted into a ReAct-style prompt for question answering with Wikipedia search and lookup actions.

## Repository layout

```text
DICE_clean/
  data/
    demos/       # successful trajectories and processed multi-level demo pool
    fever/       # FEVER dev data used in the original experiments
    hotpotqa/    # HotPotQA train/dev/test simplified files
    prompts/     # original ReAct prompt examples
  scripts/
    build_demo_pool.py  # extract question type/knowledge fields for demo retrieval
    run_hotpotqa.py     # run DICE/ReAct on HotPotQA
  src/dice/
    agent.py      # ReAct + dynamic in-context retrieval loop
    datasets.py   # HotPotQA/FEVER task adapters
    env.py        # Wikipedia search/lookup environment
    llm.py        # OpenAI and local Gemma model wrappers
    metrics.py    # EM/F1 utilities
    prompts.py    # extraction and ReAct prompt templates
    retriever.py  # sentence-transformer nearest-neighbor demo retrieval
```

## Setup

Create an environment and install dependencies:

```bash
cd DICE_clean
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For OpenAI-backed experiments, set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

The cleaned code never stores API keys in source files.

## Run HotPotQA with DICE

The default setting uses knowledge-based retrieval, matching the high-level method in the paper:

```bash
python scripts/run_hotpotqa.py \
  --split dev \
  --retrieval-mode knowledge \
  --num-examples 100 \
  --output outputs/hotpotqa_dice_knowledge.json
```

To retrieve by question type instead:

```bash
python scripts/run_hotpotqa.py \
  --split dev \
  --retrieval-mode type \
  --encode-field question_type \
  --num-examples 100 \
  --output outputs/hotpotqa_dice_type.json
```

To run the ReAct baseline without retrieved demos:

```bash
python scripts/run_hotpotqa.py \
  --split dev \
  --retrieval-mode none \
  --num-examples 100 \
  --output outputs/hotpotqa_react.json
```

## Rebuild the demo pool

The repository already includes `data/demos/doc_multi_level_knowledge_v2.json`. To regenerate it from successful trajectories:

```bash
python scripts/build_demo_pool.py \
  --input data/demos/hotpotqa_success_traj_v2.json \
  --output data/demos/doc_multi_level_knowledge_v2.json
```

## Notes

- The Wikipedia environment performs live web requests, so runs require network access.
- `sentence-transformers/all-MiniLM-L6-v2` is used by default for demo retrieval.
- `gpt-4o-mini` is used by default for extraction and agent actions. You can switch to the local Gemma wrapper with `--llm-provider gemma --llm-model google/gemma-2-2b-it`.
- Outputs are written under `outputs/` and are ignored by git by default.

