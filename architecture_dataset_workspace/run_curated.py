#!/usr/bin/env python3
"""
Run Task 1 eval on a subset of the curated ADRs.
Usage:
  python run_curated.py                     # run all curated ADRs
  python run_curated.py --source backstage  # filter by source prefix
  python run_curated.py --dry-run           # preview only
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
CURATED_PATH = BASE_DIR / "eval_curated.json"

TEMPLATE = """You are a software architect. Given the following context describing a problem or constraint in a real system, propose an architectural decision and list its consequences.

Context:
{context}

Respond in this format:
Decision: <your proposed decision>
Consequences: <the trade-offs, both positive and negative>"""


def load_env(env_path=BASE_DIR.parent / ".env"):
    if env_path.exists():
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="Filter by source prefix (e.g. backstage, edx, govuk, memorizer)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    with open(CURATED_PATH) as f:
        records = json.load(f)

    if args.source:
        records = [r for r in records if r["source_file"].startswith(args.source)]
        print(f"Filtered to source='{args.source}': {len(records)} records")

    if not records:
        print("No records to process.")
        sys.exit(1)

    # Generate prompts
    prompts = []
    for r in records:
        prompts.append({
            "id": f"{r['source_file']}_task1",
            "source_file": r["source_file"],
            "topic": r.get("topic", ""),
            "prompt": TEMPLATE.format(context=r["context"]),
            "ground_truth": {
                "decision": r["decision"],
                "consequences": r["consequences"],
            },
        })

    if args.dry_run:
        for p in prompts:
            print("─" * 60)
            print(f"ID: {p['id']}")
            print(p["prompt"])
        print(f"\nTotal: {len(prompts)}")
        return

    # Send to API
    api_url = os.environ.get("LLM_API_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o")
    api_key = os.environ.get("LLM_API_KEY", "")

    if not api_key:
        print("ERROR: LLM_API_KEY not set in .env")
        sys.exit(1)

    print(f"Model: {model}")
    print(f"Prompts: {len(prompts)}\n")

    import requests

    results = []
    for i, p in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {p['source_file']} ...", end=" ", flush=True)
        try:
            resp = requests.post(
                f"{api_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": p["prompt"]}],
                    "temperature": 0.0,
                },
                timeout=180,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
            results.append({**p, "response": reply, "error": None})
            print("OK")
        except Exception as e:
            results.append({**p, "response": None, "error": str(e)})
            print(f"ERROR: {e}")

        if args.delay and i < len(prompts) - 1:
            time.sleep(args.delay)

    out_path = BASE_DIR / f"eval_results_{args.source or 'all'}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Quick summary
    for r in results:
        dec_gt = r["ground_truth"]["decision"][:80]
        if r["response"]:
            dec_pred = r["response"][:80]
            print(f"\n{r['source_file']}:")
            print(f"  GT:     {dec_gt}...")
            print(f"  LLM:    {dec_pred}...")
        else:
            print(f"\n{r['source_file']}: ERROR - {r['error']}")


if __name__ == "__main__":
    main()
