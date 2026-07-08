#!/usr/bin/env python3
"""
Step B: Generate evaluation prompts from eval_ready.json.
Output: eval_prompts.json — list of {id, task, source_file, prompt, ground_truth}
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "eval_ready.json"
OUTPUT_PATH = BASE_DIR / "eval_prompts.json"

TASK1_TEMPLATE = """You are a software architect. Given the following context describing a problem or constraint in a real system, propose an architectural decision and list its consequences.

Context:
{context}

Respond in this format:
Decision: <your proposed decision>
Consequences: <the trade-offs, both positive and negative>"""


def main():
    with open(INPUT_PATH) as f:
        records = json.load(f)

    print(f"Loaded {len(records)} eval-ready records")

    prompts = []
    for r in records:
        prompts.append({
            "id": f"{r['source_file']}_task1",
            "task": 1,
            "source_file": r["source_file"],
            "prompt": TASK1_TEMPLATE.format(context=r["context"]),
            "ground_truth": {
                "decision": r["decision"],
                "consequences": r["consequences"],
            },
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Generated {len(prompts)} prompts")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
