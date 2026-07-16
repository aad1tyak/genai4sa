import json
import sys
import os
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from minicheck.minicheck import MiniCheck
from nltk.tokenize import sent_tokenize
import re


def split_sentences(text: str) -> list[str]:
    text = text.replace("\\n", "\n")
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            continue
        if re.match(r'^\[.+\]:\s*https?://', stripped):
            continue
        if stripped.startswith("`") and len(stripped) < 30:
            continue
        cleaned.append(stripped)
    joined = " ".join(cleaned)
    joined = re.sub(r'\s+', ' ', joined).strip()
    sents = sent_tokenize(joined)
    return [s.strip() for s in sents if len(s.strip()) > 10]


def format_sent(s: str, maxlen: int = 70) -> str:
    return s[:maxlen] + "..." if len(s) > maxlen else s


def test_case(scorer, case: dict):
    gt_d = case["ground_truth"].get("decision", "")
    gt_c = case["ground_truth"].get("consequences", "")
    gt_text = f"{gt_d}\n\n{gt_c}"
    response = case["response"]

    gt_sents = split_sentences(gt_text)
    resp_sents = split_sentences(response)

    result = {
        "id": case["id"],
        "cluster": case.get("cluster"),
        "num_gt_sents": len(gt_sents),
        "num_resp_sents": len(resp_sents),
        "gt_sentences": gt_sents,
        "response_sentences": resp_sents,
        "precision": {"results": [], "aggregate": {}},
        "recall": {"results": [], "aggregate": {}},
    }

    # Direction 1 (Precision): GT -> Response
    # For each response sentence, check if it's supported by the full GT
    t0 = time.time()
    docs_prec = [gt_text] * len(resp_sents)
    if resp_sents:
        pred_label, raw_prob, _, _ = scorer.score(docs=docs_prec, claims=resp_sents)
        supported = sum(pred_label)
        for i in range(len(resp_sents)):
            result["precision"]["results"].append({
                "resp_idx": i,
                "response_sent": resp_sents[i],
                "supported": bool(pred_label[i]),
                "confidence": round(float(raw_prob[i]), 4),
            })
        result["precision"]["aggregate"] = {
            "sentences_checked": len(resp_sents),
            "sentences_supported": int(supported),
            "precision_ratio": round(supported / len(resp_sents), 4) if resp_sents else 0,
            "time_seconds": round(time.time() - t0, 1),
        }

    # Direction 2 (Recall): Response -> GT
    # For each GT sentence, check if it's covered by the response
    t0 = time.time()
    docs_rec = [response] * len(gt_sents)
    if gt_sents:
        pred_label, raw_prob, _, _ = scorer.score(docs=docs_rec, claims=gt_sents)
        covered = sum(pred_label)
        for i in range(len(gt_sents)):
            result["recall"]["results"].append({
                "gt_idx": i,
                "gt_sent": gt_sents[i],
                "covered": bool(pred_label[i]),
                "confidence": round(float(raw_prob[i]), 4),
            })
        result["recall"]["aggregate"] = {
            "sentences_checked": len(gt_sents),
            "sentences_covered": int(covered),
            "recall_ratio": round(covered / len(gt_sents), 4) if gt_sents else 0,
            "time_seconds": round(time.time() - t0, 1),
        }

    return result


def main():
    files = [
        ("/home/bigWheel/code/gai4sa/genai4sa/eval/results/sequential_clusters_1_4.json", "clusters_1_4"),
        ("/home/bigWheel/code/gai4sa/genai4sa/eval/results/eval_results_sequential_clusters_0_9.json", "clusters_0_9"),
    ]

    print("Loading MiniCheck-Flan-T5-Large...", file=sys.stderr)
    t0 = time.time()
    scorer = MiniCheck(model_name="deberta-v3-large", cache_dir="./ckpts")
    print(f"  done in {time.time()-t0:.1f}s\n", file=sys.stderr)

    all_results = {"config": {"model": "MiniCheck-Flan-T5-Large"}, "cases": []}

    for fpath, label in files:
        with open(fpath) as f:
            cases = json.load(f)
        for case in cases:
            result = test_case(scorer, case)
            all_results["cases"].append(result)

    print(f"\n{'='*80}")
    print(f"{'Case':<60} {'Resp#':>5} {'Prec':>6} {'GT#':>5} {'Recall':>7}")
    print(f"{'-'*80}")
    for r in all_results["cases"]:
        prec = r["precision"]["aggregate"].get("precision_ratio", 0)
        rec = r["recall"]["aggregate"].get("recall_ratio", 0)
        name = r["id"][:58]
        print(f"{name:<60} {r['num_resp_sents']:>5} {prec:>6.3f} {r['num_gt_sents']:>5} {rec:>7.3f}")
    print(f"{'='*80}")

    outpath = "/home/bigWheel/code/gai4sa/genai4sa/eval/results/minicheck_results.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {outpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
