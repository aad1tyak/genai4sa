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

### Step 2 — Evaluation (not started)

Planned: apply LLM evaluation using the IHUM rubric (Insightful / Helpful / Uncertain / Misleading) and frameworks like ATAM, SAAM, CBAM, and Script Concordance Testing.
