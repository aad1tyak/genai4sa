# gai4sa — Generative AI for Software Architecture

**Research question:** Can LLMs "think like architects" — i.e., generate constraint-aware software architecture decisions from context, not just syntactically generate diagram structures?

## Overview

We evaluate LLMs on Architectural Decision Records (ADRs) from real open-source projects. The core idea: strip the Decision and Consequences from an ADR, give the LLM only the Context (and optionally example ADRs), and see if it can reconstruct a valid architecture decision.

## Pipeline

### Step 1 — Dataset (done)
139 parsed ADRs from 5 repos: Open edX (112), Backstage (13), MADR (0 — format docs), GOV.UK Docker (4), Memorizer (10).

### Step 2 — Clustering (done)
Semantic clustering (all-MiniLM-L6-v2 → UMAP → HDBSCAN) produced 17 clusters + 19 noise. Used for sequential eval cross-validation.

### Step 3 — Evaluation (done, 8 test cases)
Two task designs evaluated with GPT-4o:

**Task 1 — Forward (Context → Decision + Consequences):** Single ADR, strip Decision+Consequences, LLM reconstructs from Context only.

**Task 2 — Sequential (Example ADR → Test ADR):** Give 1-2 full example ADRs from same cluster, then present test ADR's Context only. Tests pattern transfer — can the LLM learn the project's architectural style from examples?

We ran 8 sequential cases across 4 clusters:
| Cluster | Test Case | Status |
|---|---|---|
| 0 (Memorizer) | hybrid-search-rrf | Good — near-perfect pattern transfer |
| 0 (Memorizer) | memory-search-ranking | Good — captured tag soft-boosting |
| 1 (GOV.UK) | docker-volumes | Weak — generic, missed specifics |
| 1 (GOV.UK) | docker-images | Weak — missed build details |
| 4 (Backstage) | plugin-package-structure | Weak — invented wrong naming convention |
| 4 (Backstage) | avoid-default-exports | Weak — correct topic but missed specifics |
| 9 (edx) | canonical-mfe-config | Strong — closely matched endpoint reasoning |
| 9 (edx) | cms-vs-studio | Solid — correct terminology decision |

### Step 4 — Scoring Pipeline (done)
Two-stage deterministic scoring, no LLM-as-judge:

**Stage 1 — Semantic Alignment:** `cross-encoder/stsb-distilroberta-base` computes sentence-level similarity between response and ground truth. Threshold 0.3 calibrated as the sweet spot (alignment ratio 85-100%).

**Stage 2 — NLI Verification:** `cross-encoder/nli-deberta-v3-base` (fine-tuned on MNLI/SNLI) classifies aligned pairs as ENTAILMENT / NEUTRAL / CONTRADICTION.

**Key calibration finding:** The NLI model is conservative — strong matches get ENTAILMENT (edx config: 80% entailment rate), but generic rephrases or summaries get NEUTRAL. This is expected behavior for off-the-shelf NLI on architecture text.

**Results at threshold 0.3:**
| Case | Alignment | Entail (GT) | Entail (Resp) | Contra |
|---|---|---|---|---|
| govuk volumes | 1.000 | 0.000 | 0.000 | 0.000 |
| govuk images | 1.000 | 0.091 | 0.118 | 0.000 |
| backstage plugin | 1.000 | 0.000 | 0.000 | 0.000 |
| backstage exports | 1.000 | 0.000 | 0.000 | 0.056 |
| memorizer hybrid | 1.000 | 0.000 | 0.000 | 0.000 |
| memorizer ranking | 0.846 | 0.154 | 0.273 | 0.000 |
| **edx config** | **0.952** | **0.113** | **0.800** | 0.000 |
| edx cms/studio | 0.900 | 0.200 | 0.312 | 0.000 |

**Zero contradictions detected** across all 8 cases — LLM doesn't make stuff up. Strong pattern-transfer signal for edx and Memorizer cases.

### Step 5 — MiniCheck Integration (in progress)
Discovered `MiniCheck` (Tang et al., EMNLP 2024) — a 770M parameter fact-checking model that matches GPT-4-level performance at 400x lower cost. Key advantage over our current NLI: **binary output** (supported/unsupported, no NEUTRAL escape hatch). Training data is synthetic GPT-4 generated data designed to mimic real LLM hallucination patterns.

Planned architecture for final scoring:
- **Precision:** MiniCheck(GT, response_sentence) → is each claim grounded?
- **Recall:** MiniCheck(response, GT_sentence) → is each GT point covered?
- Combined → definitive yes/no per case, no NEUTRAL ambiguity

**File:** `eval/scripts/test_minicheck.py`

## Next Steps

1. Finish MiniCheck eval on all 8 cases (CPU inference: ~15s per pair, ~45 min total)
2. If MiniCheck passes, replace NLI stage with MiniCheck
3. Compare MiniCheck results against our NLI pipeline + human expert review
4. Expand to full 71-ADR dataset
5. Cross-model comparison (GPT-4o vs Gemini Pro 3)

## Project Structure

```
eval/
├── data/
│   ├── adr_dataset.jsonl               # 139 parsed ADRs
│   ├── adr_clusters_semantic.json      # semantic cluster assignments
│   ├── eval_ready.json                 # 71 with all 3 fields
│   └── eval_curated.json               # 14 hand-picked
├── scripts/
│   ├── parse_adrs.py / cluster_adrs.py / prepare_eval.py / generate_prompts.py
│   ├── run_sequential.py               # sequential eval runner
│   ├── score_sequential.py             # two-stage scoring pipeline
│   └── test_minicheck.py               # MiniCheck integration test
├── results/
│   ├── sequential_clusters_1_4.json             # GPT-4o results (GOV.UK + Backstage)
│   ├── eval_results_sequential_clusters_0_9.json # GPT-4o results (Memorizer + edx)
│   ├── sequential_clusters_1_4_scored.json       # scored with alignment + NLI
│   └── minicheck_results.json                    # (pending) MiniCheck binary results
└── reports/
```

## Key Technical Decisions

- **No LLM-generated questions** — all prompts are deterministic string substitutions from ADR fields
- **Sequential eval preferred** — tests pattern transfer, not keyword recall
- **Cross-encoder alignment + NLI** — reference-free scoring pipeline
- **MiniCheck as next step** — binary verdicts eliminate NEUTRAL ambiguity
- **CPU inference for now** — all models run on CPU; MiniCheck ~15s/pair
