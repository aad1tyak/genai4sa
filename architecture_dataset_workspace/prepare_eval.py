#!/usr/bin/env python3
"""
Step A: Filter ADR dataset to records with all 3 core fields
(Context, Decision, Consequences) meaningfully populated.
Output: eval_ready.json
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent
JSONL_PATH = BASE_DIR / "adr_dataset.jsonl"
OUTPUT_PATH = BASE_DIR / "eval_ready.json"


def is_filled(val, min_len=30):
    return bool(val) and len(val.strip()) > min_len


def main():
    with open(JSONL_PATH) as f:
        records = [json.loads(l) for l in f if l.strip()]

    print(f"Total records in dataset: {len(records)}")

    eval_ready = []
    for r in records:
        if not is_filled(r.get("context")):
            continue
        if not is_filled(r.get("decision")):
            continue
        if not is_filled(r.get("consequences")):
            continue
        eval_ready.append({
            "source_file": r["source_file"],
            "context": r["context"].strip(),
            "decision": r["decision"].strip(),
            "consequences": r["consequences"].strip(),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(eval_ready, f, indent=2)

    print(f"Eval-ready records: {len(eval_ready)}")
    from collections import Counter
    srcs = Counter(r["source_file"].split("_")[0] for r in eval_ready)
    for s, c in srcs.most_common():
        print(f"  {s}: {c}")
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
