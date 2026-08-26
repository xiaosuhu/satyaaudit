"""One-off diagnostic: verify Step 3a (query_distillation) + Step 3b
(claim_comparison) of the Replicability module's cross-paper claim
comparator against real Claude + Asta APIs.

Not part of the satyarepro package — this is a throwaway script, not
collected by pytest, and makes no pass/fail assertions. It only prints
and saves output for a human to review.

Usage:
    python scripts/verify_claim_comparison.py
"""
from dotenv import load_dotenv

load_dotenv()

import asyncio
import json
from collections import Counter

from satyarepro.client.asta import AstaClient
from satyarepro.client.claude import ClaudeClient
from satyarepro.tools.layer2.replicability.claim_comparison import extract_and_compare
from satyarepro.tools.layer2.replicability.query_distillation import (
    distill_and_retrieve,
    distill_query,
)

# Ground truth: Vandewiele et al. 2021, Artif Intell Med 111:101987 —
# the EHG/SMOTE-before-split-leakage paper.
TASK_DESCRIPTION = (
    "predicting preterm birth using SMOTE-balanced XGBoost classifiers "
    "on electrohysterogram (EHG) signals"
)
TARGET_CLAIM = (
    "Applying SMOTE oversampling to EHG-based preterm birth prediction "
    "improves classifier AUC compared to no oversampling"
)

_OUTPUT_PATH = "/tmp/verify_claim_comparison_output.json"
_DIVIDER = "=" * 80


def _print_divider(title: str) -> None:
    print(f"\n{_DIVIDER}\n{title}\n{_DIVIDER}")


async def main() -> None:
    claude_client = ClaudeClient()
    asta_client = AstaClient()

    _print_divider("(a) task_description")
    print(TASK_DESCRIPTION)

    _print_divider("(b) distilled query (Step 3a)")
    distilled_query = await distill_query(TASK_DESCRIPTION, claude_client)
    print(distilled_query)

    _print_divider("(c) distill_and_retrieve candidates (Step 3a, limit=8)")
    candidates = await distill_and_retrieve(
        TASK_DESCRIPTION,
        client=claude_client,
        asta_client=asta_client,
        limit=8,
        exclude_corpus_id="210714142",
    )
    for i, c in enumerate(candidates, 1):
        print(f"\n--- candidate {i} ---")
        print(f"corpus_id: {c.get('corpus_id')}")
        print(f"title: {c.get('title')}")
        print(f"score: {c.get('score')}")
        print(f"snippet: {c.get('snippet')}")

    _print_divider("(d) target_claim")
    print(TARGET_CLAIM)

    _print_divider("(e) extract_and_compare judgments (Step 3b)")
    judgments = await extract_and_compare(TARGET_CLAIM, candidates, client=claude_client)
    for i, j in enumerate(judgments, 1):
        print(f"\n--- judgment {i} ---")
        if "error" in j:
            print(f"corpus_id: {j.get('corpus_id')}")
            print(f"title: {j.get('title')}")
            print(f"error: {j.get('error')}")
            continue
        print(f"corpus_id: {j.get('corpus_id')}")
        print(f"title: {j.get('title')}")
        print(f"population_modality: {j.get('population_modality')}")
        print(f"extracted_claim: {j.get('extracted_claim')}")
        print(f"comparison_type: {j.get('comparison_type')}")
        print(f"agreement: {j.get('agreement')}")
        print(f"evidence: {j.get('evidence')}")
        print(f"confidence: {j.get('confidence')}")

    _print_divider("(f) summary")
    total = len(judgments)
    comparison_counts = Counter(j.get("comparison_type") for j in judgments if "error" not in j)
    agreement_counts = Counter(j.get("agreement") for j in judgments if "error" not in j)
    error_count = sum(1 for j in judgments if "error" in j)
    print(f"total candidates: {total}")
    print(f"by comparison_type: {dict(comparison_counts)}")
    print(f"by agreement: {dict(agreement_counts)}")
    print(f"error entries: {error_count}")

    output = {
        "task_description": TASK_DESCRIPTION,
        "target_claim": TARGET_CLAIM,
        "distilled_query": distilled_query,
        "candidates": candidates,
        "judgments": judgments,
    }
    with open(_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    _print_divider("(g) output file")
    print(f"Full structured result written to {_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
