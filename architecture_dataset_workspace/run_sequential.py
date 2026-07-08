#!/usr/bin/env python3
"""
Sequential evaluation: give N-1 ADRs fully, test on the withheld one.
Usage:
  python run_sequential.py --clusters 1,4          # GOV.UK + Backstage
  python run_sequential.py --clusters 1,4 --dry-run
"""
import json
import os
import sys
import time
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).parent
CURATED_PATH = BASE_DIR / "eval_curated.json"
CLUSTER_PATH = BASE_DIR / "adr_clusters_semantic.json"

TEMPLATE = """You are a software architect. Below are architectural decisions made in a real project. Study them carefully — they represent the architectural patterns and constraints of this system.

{examples}

Now consider the following new problem in the same system:

{context}

Based on the architectural patterns and constraints shown in the previous decisions, what architectural decision would you make here, and what would be the consequences?

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
    parser.add_argument("--clusters", required=True, help="Comma-separated cluster IDs to run (e.g. 1,4)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    cluster_ids = [int(c.strip()) for c in args.clusters.split(",")]

    # Load cluster info
    with open(CLUSTER_PATH) as f:
        clusters_data = json.load(f)
    cluster_map = {r["source_file"]: r.get("cluster") for r in clusters_data}

    # Load curated ADRs
    with open(CURATED_PATH) as f:
        records = json.load(f)

    # Group by cluster, filter to target clusters
    from collections import defaultdict
    by_cluster = defaultdict(list)
    for r in records:
        cid = cluster_map.get(r["source_file"], -99)
        if cid in cluster_ids:
            by_cluster[cid].append(r)

    print(f"Target clusters: {cluster_ids}")
    for cid in sorted(by_cluster):
        print(f"  Cluster {cid}: {len(by_cluster[cid])} ADRs — {[r['source_file'] for r in by_cluster[cid]]}")

    # Build all test pairs (A→B and B→A for each cluster)
    test_cases = []
    for cid in sorted(by_cluster):
        members = by_cluster[cid]
        for i, test_adr in enumerate(members):
            examples = [m for j, m in enumerate(members) if j != i]
            test_cases.append({
                "id": f"cluster{cid}_test{test_adr['source_file']}",
                "cluster": cid,
                "source_file": test_adr["source_file"],
                "examples": examples,
                "test_adr": test_adr,
            })

    print(f"\nTest cases: {len(test_cases)}")

    if args.dry_run:
        for tc in test_cases:
            example_texts = []
            for ex in tc["examples"]:
                example_texts.append(f"--- ADR: {ex['source_file']} ---\n{ex['context']}\n\nDecision: {ex['decision']}\n\nConsequences: {ex['consequences']}")

            prompt = TEMPLATE.format(
                examples="\n\n".join(example_texts),
                context=tc["test_adr"]["context"],
            )
            print("─" * 60)
            print(f"ID: {tc['id']}")
            print(f"Examples: {[e['source_file'] for e in tc['examples']]}")
            print(f"Test: {tc['test_adr']['source_file']}")
            print()
            print(prompt[:500])
            print("...")
        print(f"\nTotal: {len(test_cases)}")
        return

    # API
    api_url = os.environ.get("LLM_API_URL", "https://api.openai.com/v1")
    model = os.environ.get("LLM_MODEL", "gpt-4o")
    api_key = os.environ.get("LLM_API_KEY", "")

    if not api_key:
        print("ERROR: LLM_API_KEY not set")
        sys.exit(1)

    import requests
    results = []

    for i, tc in enumerate(test_cases):
        example_texts = []
        for ex in tc["examples"]:
            example_texts.append(f"--- ADR: {ex['source_file']} ---\n{ex['context']}\n\nDecision: {ex['decision']}\n\nConsequences: {ex['consequences']}")

        prompt = TEMPLATE.format(
            examples="\n\n".join(example_texts),
            context=tc["test_adr"]["context"],
        )

        print(f"[{i+1}/{len(test_cases)}] {tc['source_file']} (cluster {tc['cluster']}) ...", end=" ", flush=True)
        try:
            resp = requests.post(
                f"{api_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
                timeout=180,
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
            results.append({
                "id": tc["id"],
                "cluster": tc["cluster"],
                "source_file": tc["source_file"],
                "examples": [e["source_file"] for e in tc["examples"]],
                "prompt": prompt,
                "ground_truth": {
                    "decision": tc["test_adr"]["decision"],
                    "consequences": tc["test_adr"]["consequences"],
                },
                "response": reply,
            })
            print("OK")
        except Exception as e:
            results.append({
                "id": tc["id"],
                "cluster": tc["cluster"],
                "source_file": tc["source_file"],
                "examples": [e["source_file"] for e in tc["examples"]],
                "prompt": prompt,
                "ground_truth": {
                    "decision": tc["test_adr"]["decision"],
                    "consequences": tc["test_adr"]["consequences"],
                },
                "response": None,
                "error": str(e),
            })
            print(f"ERROR: {e}")

        if args.delay and i < len(test_cases) - 1:
            time.sleep(args.delay)

    out_path = BASE_DIR / f"eval_results_sequential_clusters_{'_'.join(str(c) for c in cluster_ids)}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {out_path}")

    # Quick compare
    for r in results:
        print(f"\n=== {r['source_file']} ===")
        print(f"GT:  {r['ground_truth']['decision'][:100]}...")
        if r["response"]:
            print(f"LLM: {r['response'][:100]}...")
        else:
            print(f"LLM: ERROR - {r['error']}")


if __name__ == "__main__":
    main()
