#!/usr/bin/env python3
"""
Cluster ADR records by topic using TF-IDF + UMAP + HDBSCAN.
Outputs cluster assignments to adr_clusters.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import hdbscan
import umap
from sklearn.feature_extraction.text import TfidfVectorizer


def load_records(jsonl_path: str) -> list[dict]:
    with open(jsonl_path) as f:
        return [json.loads(l) for l in f]


def clean_text(text: str) -> str:
    text = re.sub(r'[=\-~]{3,}', ' ', text)
    text = re.sub(r'`{2,}', ' ', text)
    return text


def cluster_records(records: list[dict], min_cluster_size: int = 3):
    docs = [clean_text(r["raw_text"]) for r in records]

    vec = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(docs)

    reducer = umap.UMAP(n_components=5, random_state=42, n_neighbors=15)
    X_umap = reducer.fit_transform(X.toarray())

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=2, metric="euclidean")
    labels = clusterer.fit_predict(X_umap)

    return labels, clusterer.probabilities_


def main():
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "./adr_dataset.jsonl"
    records = load_records(jsonl_path)
    print(f"Loaded {len(records)} records")

    labels, probs = cluster_records(records)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise = list(labels).count(-1)
    print(f"Clusters: {n_clusters}, Noise: {noise}/{len(records)}")

    # Save assignments
    output = []
    for i, r in enumerate(records):
        output.append({
            "source_file": r["source_file"],
            "cluster": int(labels[i]),
            "cluster_prob": float(probs[i]) if probs[i] is not None else 0.0,
            "context_preview": (r.get("context") or "")[:120],
        })

    out_path = Path(jsonl_path).parent / "adr_clusters.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}")

    # Print summary
    for label in sorted(set(labels)):
        members = [i for i, l in enumerate(labels) if l == label]
        srcs = Counter(records[i]["source_file"].split("_")[0] for i in members)
        print(f"\n  Cluster {label} ({len(members)}): {dict(srcs)}")
        for i in members[:4]:
            print(f"    {records[i]['source_file']}")
        if len(members) > 4:
            print(f"    ... +{len(members)-4} more")


if __name__ == "__main__":
    main()
