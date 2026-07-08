#!/usr/bin/env python3
"""
Step C: Run evaluation prompts against an LLM API.
Reads eval_prompts.json, sends prompts, saves eval_results.json.

Usage:
  export LLM_API_KEY="your-key"
  export LLM_MODEL="gemini-3-pro"     # or gpt-4o, claude-3, etc.
  export LLM_API_URL="https://api.example.com/v1"  # OpenAI-compatible

  python run_eval.py                   # run all prompts
  python run_eval.py --dry-run         # preview prompts only
  python run_eval.py --start 0 --end 10   # run a subset
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "eval_prompts.json"
OUTPUT_PATH = BASE_DIR / "eval_results.json"


def load_prompts():
    with open(INPUT_PATH) as f:
        return json.load(f)


def save_results(results):
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to {OUTPUT_PATH}")


def send_prompt(prompt_text, api_url, model, api_key):
    import requests
    resp = requests.post(
        f"{api_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.0,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Run ADR evaluation prompts against LLM")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without sending")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End index (exclusive)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between calls (seconds)")
    args = parser.parse_args()

    prompts = load_prompts()
    total = len(prompts)
    start = args.start
    end = args.end if args.end is not None else total
    batch = prompts[start:end]
    print(f"Loaded {total} prompts, running {len(batch)} ({start}:{end})")

    if args.dry_run:
        for p in batch[:3]:
            print("─" * 60)
            print(f"ID: {p['id']}")
            print(p["prompt"])
            print()
        print(f"Total that would be sent: {len(batch)}")
        return

    api_url = os.environ.get("LLM_API_URL", "")
    model = os.environ.get("LLM_MODEL", "")
    api_key = os.environ.get("LLM_API_KEY", "")

    if not api_key or not model or not api_url:
        print("ERROR: Set LLM_API_URL, LLM_MODEL, and LLM_API_KEY environment variables.")
        print("  export LLM_API_URL='https://generativelanguage.googleapis.com/v1beta/openai'")
        print("  export LLM_MODEL='gemini-3-pro'")
        print("  export LLM_API_KEY='your-key'")
        sys.exit(1)

    results = []
    errors = 0
    for i, p in enumerate(batch):
        idx = start + i
        print(f"  [{idx+1}/{total}] {p['id']} ...", end=" ", flush=True)
        try:
            reply = send_prompt(p["prompt"], api_url, model, api_key)
            results.append({
                "id": p["id"],
                "task": p["task"],
                "source_file": p["source_file"],
                "prompt": p["prompt"],
                "ground_truth": p["ground_truth"],
                "response": reply,
            })
            print("OK")
        except Exception as e:
            results.append({
                "id": p["id"],
                "task": p["task"],
                "source_file": p["source_file"],
                "prompt": p["prompt"],
                "ground_truth": p["ground_truth"],
                "response": None,
                "error": str(e),
            })
            print(f"ERROR: {e}")
            errors += 1

        if args.delay > 0:
            time.sleep(args.delay)

    save_results(results)
    print(f"Done. {len(results)-errors} ok, {errors} errors")


if __name__ == "__main__":
    main()
