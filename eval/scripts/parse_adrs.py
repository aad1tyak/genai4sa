#!/usr/bin/env python3
import os
import re
import json

DATASET_DIR = "./compiled_adr_dataset"
OUTPUT_FILE = "./adr_dataset.jsonl"


def parse_adr_content(file_path, file_content):
    """
    Parses Markdown and RST files to extract core ADR components.
    Uses regex lookaheads to capture multi-line paragraphs up to the next major header.
    """
    context_match = re.search(
        r'(?i)(?:#+\s*Context|Context\n[=\-\~]+)\s*\n(.*?)(?=\n(?:#+\s*[A-Z]|[A-Z][a-z]+\n[=\-\~]+)|$)',
        file_content, re.DOTALL
    )
    decision_match = re.search(
        r'(?i)(?:#+\s*Decision|Decision\n[=\-\~]+)\s*\n(.*?)(?=\n(?:#+\s*[A-Z]|[A-Z][a-z]+\n[=\-\~]+)|$)',
        file_content, re.DOTALL
    )
    consequences_match = re.search(
        r'(?i)(?:#+\s*Consequences|Consequences\n[=\-\~]+)\s*\n(.*?)(?=\n(?:#+\s*[A-Z]|[A-Z][a-z]+\n[=\-\~]+)|$)',
        file_content, re.DOTALL
    )
    status_match = re.search(
        r'(?i)(?:#+\s*Status|Status\n[=\-\~]+)\s*\n(.*?)(?=\n(?:#+\s*[A-Z]|[A-Z][a-z]+\n[=\-\~]+)|$)',
        file_content, re.DOTALL
    )

    context = context_match.group(1).strip() if context_match else None
    decision = decision_match.group(1).strip() if decision_match else None
    consequences = consequences_match.group(1).strip() if consequences_match else None
    status = status_match.group(1).strip() if status_match else None

    return {
        "source_file": os.path.basename(file_path),
        "status": status,
        "context": context,
        "decision": decision,
        "consequences": consequences,
        "raw_text": file_content,
    }


def compile_dataset():
    parsed_records = []
    for filename in os.listdir(DATASET_DIR):
        file_path = os.path.join(DATASET_DIR, filename)
        if not os.path.isfile(file_path):
            continue
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        parsed_data = parse_adr_content(file_path, content)
        if parsed_data["decision"] or parsed_data["context"]:
            parsed_records.append(parsed_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for record in parsed_records:
            out_f.write(json.dumps(record) + "\n")

    print(f"Successfully compiled {len(parsed_records)} structured ADRs into {OUTPUT_FILE}")


if __name__ == "__main__":
    compile_dataset()
