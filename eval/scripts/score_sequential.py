import json
import re
import sys
import time
from pathlib import Path

from nltk.tokenize import sent_tokenize
from sentence_transformers import CrossEncoder
import numpy as np


SIM_MODEL = "cross-encoder/stsb-distilroberta-base"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


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


def load_models():
    print("Loading similarity model...", file=sys.stderr)
    t0 = time.time()
    sim_model = CrossEncoder(SIM_MODEL)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    print("Loading NLI model...", file=sys.stderr)
    t0 = time.time()
    nli_model = CrossEncoder(NLI_MODEL)
    print(f"  done in {time.time()-t0:.1f}s", file=sys.stderr)

    return sim_model, nli_model


def run_alignment(sim_model, resp_sents: list[str], gt_sents: list[str], threshold: float = 0.5):
    if not resp_sents or not gt_sents:
        return [], [], {}

    pairs = [(r, g) for r in resp_sents for g in gt_sents]
    scores = sim_model.predict(pairs)
    scores = np.array(scores).reshape(len(resp_sents), len(gt_sents))

    evidence_map = {}
    for i, r_sent in enumerate(resp_sents):
        matches = []
        for j, g_sent in enumerate(gt_sents):
            sim = float(scores[i][j])
            if sim >= threshold:
                matches.append({"gt_idx": j, "gt_sent": g_sent, "similarity": round(sim, 4)})
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        evidence_map[i] = {
            "response_sent": r_sent,
            "matches": matches if matches else None,
        }

    return evidence_map, scores, gt_sents


def run_nli(nli_model, evidence_map: dict):
    pairs_for_nli = []
    pair_index = []
    for r_idx, entry in evidence_map.items():
        if entry["matches"]:
            for m in entry["matches"]:
                pairs_for_nli.append((entry["response_sent"], m["gt_sent"]))
                pair_index.append((r_idx, m["gt_idx"]))

    if not pairs_for_nli:
        return evidence_map, []

    logits = nli_model.predict(pairs_for_nli)
    label_map = {0: "CONTRADICTION", 1: "ENTAILMENT", 2: "NEUTRAL"}
    nli_results = []
    for (r_idx, g_idx), lbl_logits in zip(pair_index, logits):
        label_id = int(np.argmax(lbl_logits))
        label = label_map[label_id]
        confidence = round(float(np.max(lbl_logits)), 4)
        nli_results.append({
            "response_idx": r_idx,
            "gt_idx": g_idx,
            "nli_label": label,
            "confidence": confidence,
        })
        for m in evidence_map[r_idx]["matches"]:
            if m["gt_idx"] == g_idx:
                m["nli_label"] = label
                m["nli_confidence"] = confidence
                break

    return evidence_map, nli_results


def compute_aggregates(evidence_map: dict, nli_results: list[dict]):
    n_resp = len(evidence_map)

    gt_aligned = set()
    gt_entailed = set()
    n_entail = 0
    n_contra = 0
    n_neutral = 0
    n_unsupported = 0

    for r_idx, entry in evidence_map.items():
        if not entry["matches"]:
            n_unsupported += 1
            continue
        labels = [m.get("nli_label") for m in entry["matches"] if m.get("nli_label")]
        if not labels:
            n_unsupported += 1
            continue
        for m in entry["matches"]:
            gt_aligned.add(m["gt_idx"])
            if m.get("nli_label") == "ENTAILMENT":
                gt_entailed.add(m["gt_idx"])
        if "ENTAILMENT" in labels:
            n_entail += 1
        elif all(l == "CONTRADICTION" for l in labels):
            n_contra += 1
        else:
            n_neutral += 1

    actual_gt = 0
    for entry in evidence_map.values():
        if entry["matches"]:
            for m in entry["matches"]:
                actual_gt = max(actual_gt, m["gt_idx"] + 1)

    n_total_pairs = len(nli_results)
    n_entail_pairs = sum(1 for nr in nli_results if nr["nli_label"] == "ENTAILMENT")
    n_contra_pairs = sum(1 for nr in nli_results if nr["nli_label"] == "CONTRADICTION")
    n_neutral_pairs = sum(1 for nr in nli_results if nr["nli_label"] == "NEUTRAL")

    return {
        "response_sentences": n_resp,
        "ground_truth_sentences": actual_gt,
        "gt_sentences_aligned": len(gt_aligned),
        "gt_alignment_ratio": round(len(gt_aligned) / actual_gt, 4) if actual_gt > 0 else 0,
        "gt_sentences_entailed": len(gt_entailed),
        "gt_entailment_ratio": round(len(gt_entailed) / actual_gt, 4) if actual_gt > 0 else 0,
        "response_entailed": n_entail,
        "response_contradicted": n_contra,
        "response_neutral": n_neutral,
        "response_unsupported": n_unsupported,
        "entailment_rate": round(n_entail / n_resp, 4) if n_resp > 0 else 0,
        "contradiction_rate": round(n_contra / n_resp, 4) if n_resp > 0 else 0,
        "neutral_rate": round(n_neutral / n_resp, 4) if n_resp > 0 else 0,
        "unsupported_rate": round(n_unsupported / n_resp, 4) if n_resp > 0 else 0,
        "aligned_pairs": n_total_pairs,
        "pair_entailments": n_entail_pairs,
        "pair_contradictions": n_contra_pairs,
        "pair_neutrals": n_neutral_pairs,
    }


def score_one(sim_model, nli_model, gt_text: str, response_text: str, threshold: float = 0.5):
    gt_sents = split_sentences(gt_text)
    resp_sents = split_sentences(response_text)

    evidence_map, _, _ = run_alignment(sim_model, resp_sents, gt_sents, threshold)
    evidence_map, nli_results = run_nli(nli_model, evidence_map)
    aggregates = compute_aggregates(evidence_map, nli_results)

    return {
        "num_gt_sentences": len(gt_sents),
        "num_response_sentences": len(resp_sents),
        "gt_sentences": gt_sents,
        "response_sentences": resp_sents,
        "evidence_map": evidence_map,
        "nli_results": nli_results,
        "aggregates": aggregates,
    }


def score_file(results_path: str, output_path: str | None = None, threshold: float = 0.5):
    sim_model, nli_model = load_models()

    with open(results_path) as f:
        results = json.load(f)

    scored = []
    for idx, case in enumerate(results):
        print(f"\n--- Scoring case {idx+1}/{len(results)}: {case['id']} ---", file=sys.stderr)
        t0 = time.time()

        gt_d = case["ground_truth"].get("decision", "")
        gt_c = case["ground_truth"].get("consequences", "")
        gt_text = f"{gt_d}\n\n{gt_c}"
        response = case["response"]

        score = score_one(sim_model, nli_model, gt_text, response, threshold)
        score["id"] = case["id"]
        score["cluster"] = case.get("cluster")
        score["source_file"] = case.get("source_file")
        score["examples"] = case.get("examples")

        agg = score["aggregates"]
        print(f"  GT sents: {score['num_gt_sentences']}, Response sents: {score['num_response_sentences']}", file=sys.stderr)
        print(f"  Align: {agg['gt_alignment_ratio']} ({agg['gt_sentences_aligned']}/{agg['ground_truth_sentences']})", file=sys.stderr)
        print(f"  Entail: {agg['gt_entailment_ratio']} ({agg['gt_sentences_entailed']}/{agg['ground_truth_sentences']})", file=sys.stderr)
        print(f"  Resp -> Entail {agg['entailment_rate']} Contra {agg['contradiction_rate']} Neutral {agg['neutral_rate']} Unsupp {agg['unsupported_rate']}", file=sys.stderr)
        print(f"  Time: {time.time()-t0:.1f}s", file=sys.stderr)

        scored.append(score)

    summary = {
        "config": {"similarity_threshold": threshold, "sim_model": SIM_MODEL, "nli_model": NLI_MODEL},
        "results": scored,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {output_path}", file=sys.stderr)

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to eval results JSON")
    parser.add_argument("-o", "--output", help="Output path (default: input with _scored suffix)")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Similarity threshold (default: 0.5)")
    args = parser.parse_args()

    inp = Path(args.input)
    out = args.output or str(inp.with_stem(inp.stem + "_scored"))
    score_file(str(inp), out, args.threshold)
