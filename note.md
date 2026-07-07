# gai4sa — Generative AI for Software Architecture

**Research question:** Can LLMs generate software architectures or "think like architects"?

## Overview

This project evaluates whether LLMs are capable of constraint-aware reasoning about software architecture — as opposed to just syntactically generating diagram structures. It critiques existing benchmarks (SADU, R2ABench) for measuring topological compliance rather than genuine architectural cognition.

## Pipeline

### Step 1 — ADR Dataset Collection (done)

Architectural Decision Records (ADRs) are extracted from open-source projects and
parsed into a structured JSONL dataset for LLM evaluation.

**Source repositories** (all shallow-cloned via `extract_adrs.sh`):

| Repo | ADRs found | Parsed |
|---|---|---|
| openedx/edx-platform | 141 | 112 |
| backstage/backstage | 17 | 13 |
| adr/madr | 21 | 0* |
| alphagov/govuk-docker | 6 | 4 |
| petabridge/memorizer | 12 | 10 |
| **Total** | **195** | **139** |

*\* MADR files describe the ADR format itself, not actual architecture decisions.*

**Scripts:**
- `architecture_dataset_workspace/extract_adrs.sh` — clones repos and copies ADR files into `compiled_adr_dataset/`
- `architecture_dataset_workspace/parse_adrs.py` — parses raw ADRs into structured JSONL (`adr_dataset.jsonl`)

**Output:** `architecture_dataset_workspace/adr_dataset.jsonl` — 139 records with fields `source_file`, `status`, `context`, `decision`, `consequences`, `raw_text`.

**Coverage:**
- Context: 94%
- Decision: 71%
- Consequences: 71%
- Status: 73%

### Step 2.1 — Semantic Clustering (done)

Two clustering approaches compared:

**TF-IDF** (local, fast): `cluster_adrs.py` → `adr_clusters.json` — 14 clusters + 10 noise
**Semantic** (Colab, sentence-transformers): `cluster_adrs_colab.ipynb` → `adr_clusters_semantic.json` — 17 clusters + 19 noise

The semantic approach gives cleaner separation (pure govuk cluster, pure Backstage sub-clusters) at the cost of more noise points.

| Cluster | Size | Source | Topic |
|---|---|---|---|
| -1 | 19 | mixed | Noise — hard-to-classify |
| 0 | 11 | memorizer (10) + edx (1) | Vector embeddings, chunking, LLM service |
| 1 | 3 | govuk | Docker infrastructure |
| 2 | 15 | edx | Auth / JWT / OAuth / scopes |
| 3 | 3 | backstage | HTTP/mocking infra |
| 4 | 10 | backstage (8) + edx + govuk | ADR meta, plugins, general architecture |
| 5 | 5 | edx | Permissions, ratelimiting, SSO |
| 6 | 9 | edx | Discussions, teams, course features |
| 7 | 4 | edx | Waffle flags / toggles |
| 8 | 5 | edx | XBlocks, content structure, modulestore |
| 9 | 11 | edx | Translations, plugins, Django infra |
| 10 | 7 | edx | Learning goals, calendar, email, events |
| 11 | 6 | edx | Certificates, cert dates |
| 12 | 4 | edx | Logging, monitoring, idempotency |
| 13 | 8 | edx | API standardization (serializers, errors, filtering) |
| 14 | 5 | edx | Persistent grades, grade overrides |
| 15 | 8 | edx | Enrollment APIs, Studio APIs, ORA |
| 16 | 6 | edx | Program enrollments, Georgia Tech, access

### Step 3 — Ablation Evaluation Pipeline (in progress)

Instead of manually writing questions or using LLMs to generate them, we built a **deterministic ablation pipeline** that generates evaluation prompts programmatically from the ADR data itself.

**Core idea:** Strip one or two fields from each ADR and ask the LLM to reconstruct them. The ground truth is the original field — scored with ROUGE-L (deterministic, no LLM-as-judge needed).

**Task 1 — Forward (Context → Decision + Consequences):**
Given only the context, the LLM must propose an architectural decision and its trade-offs. Tests whether the LLM can design a valid architecture from constraints.

**Data:** 71 ADRs with all 3 core fields (Context, Decision, Consequences) sufficiently filled.
- edx: 45 | backstage: 12 | memorizer: 10 | govuk: 4

**Pipeline (modular scripts, one step each):**

| Step | Script | Input | Output |
|---|---|---|---|
| A — Filter | `prepare_eval.py` | `adr_dataset.jsonl` | `eval_ready.json` (71 records) |
| B — Generate prompts | `generate_prompts.py` | `eval_ready.json` | `eval_prompts.json` (71 prompts) |
| C — Run LLM | `run_eval.py` | `eval_prompts.json` | `eval_results.json` |
| D — Score | *(pending)* | `eval_results.json` | ROUGE-L scores |

**Usage (Step C):**
```bash
export LLM_API_URL="https://your-api/v1"
export LLM_MODEL="gemini-3-pro"
export LLM_API_KEY="your-key"
python run_eval.py
python run_eval.py --dry-run           # preview without sending
python run_eval.py --start 0 --end 10  # run a subset
```
