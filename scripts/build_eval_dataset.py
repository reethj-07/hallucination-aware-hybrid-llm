"""Build an expanded evaluation set with NQ-Open plus domain QA pairs."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset

NQ_SAMPLES = 200
DOMAIN_PAIRS = Path("data/eval/qa_pairs.jsonl")
OUTPUT = Path("data/eval/qa_pairs_v2.jsonl")


def main() -> None:
    nq = load_dataset("google-research-datasets/nq_open", split="validation")
    nq_subset = [
        {
            "id": f"nq_{idx}",
            "question": row["question"],
            "expected_answers": row["answer"],
            "source": "nq_open",
        }
        for idx, row in enumerate(nq.shuffle(seed=42).select(range(NQ_SAMPLES)))
    ]

    domain = [json.loads(line) for line in DOMAIN_PAIRS.read_text(encoding="utf-8").splitlines() if line.strip()]
    for pair in domain:
        pair["source"] = "domain"
        if "expected_answers" not in pair:
            pair["expected_answers"] = [pair.pop("answer", "")]
        if "question" not in pair and "query" in pair:
            pair["question"] = pair.pop("query")

    all_pairs = domain + nq_subset
    OUTPUT.write_text("\n".join(json.dumps(pair) for pair in all_pairs), encoding="utf-8")
    print(f"Wrote {len(all_pairs)} QA pairs to {OUTPUT}")


if __name__ == "__main__":
    main()
