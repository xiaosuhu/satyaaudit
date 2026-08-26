from __future__ import annotations

from typing import Any

from satyarepro.client.base import ModelClient

from ..._llm_utils import _extract_json_object

_JUDGE_SYSTEM = (
    "You are a biomedical ML methodologist comparing a target research claim against a "
    "candidate paper's reported findings, using a structured taxonomy for how directly "
    "comparable the two are and whether they agree."
)

_JUDGE_PROMPT = """\
Target claim:
\"\"\"
{target_claim}
\"\"\"

Candidate paper title:
\"\"\"
{title}
\"\"\"

Candidate paper snippet:
\"\"\"
{snippet}
\"\"\"

Do the following, in order:

1. Determine "population_modality": what population and what input modality/data type \
the candidate paper studies. Base this primarily on the snippet text; the title may be \
used as auxiliary context but is not sufficient on its own if the snippet doesn't support \
it. If the snippet gives no information at all about population/modality, say so \
explicitly rather than guessing from the title alone.

2. Extract "extracted_claim": the directional claim the candidate paper makes that is \
comparable to the target claim. If the snippet does not contain a directional claim (e.g. \
it only mentions the same topic or dataset without stating an effect, direction, or \
outcome), leave this as an empty string "" — do not force a claim into existence.

3. Determine "comparison_type":
- "direct": same population + same modality, testing the same underlying claim as the \
target claim.
- "conceptual": different population or modality, but testing the same underlying claim \
(the comparison is still meaningful, just not a literal replication).
- "not_comparable": the candidate is not actually testing the same underlying claim at \
all (e.g. same general topic/dataset, but a different research question).

4. Determine "agreement" — but ONLY if extracted_claim is non-empty AND comparison_type \
is not "not_comparable". If extracted_claim is empty, agreement MUST be \
"no_comparable_evidence" regardless of what comparison_type was determined to be. The \
agreement categories:
- "agree": both claims are directionally consistent with no meaningful qualification.
- "partially_agree": both claims have a stated direction, but with a meaningful \
qualification or partial divergence (e.g. consistent in one subgroup but not another, \
same direction but different significance/robustness caveats noted by the candidate's own \
authors).
- "conflict": the candidate's directional claim contradicts the target claim.
- "no_comparable_evidence": no directional claim could be extracted from the snippet \
(comparison_type may still be direct/conceptual — e.g. same population/modality, but the \
retrieved snippet just doesn't happen to cover the outcome direction).

Calibration examples for the partially_agree vs. no_comparable_evidence boundary:
- "mentions the same oversampling approach was tried on a similar dataset, no effect \
direction stated" -> no_comparable_evidence (no direction was ever stated).
- "reports improved recall in one subgroup but not another" -> partially_agree (a \
direction is stated, but with a qualification).
- "reports the same direction of effect as the target claim but notes it was not \
statistically significant" -> partially_agree (direction stated, with a caveat).

5. Provide "evidence": a short quote or paraphrase from the snippet supporting the \
comparison_type + agreement determination.

6. Provide "confidence": "high", "medium", or "low", reflecting how much the judgment is \
limited by the snippet being a short excerpt rather than the full paper.

Respond with ONLY a JSON object (no prose, no markdown fences) with these fields:
- "population_modality": string
- "extracted_claim": string
- "comparison_type": "direct" | "conceptual" | "not_comparable"
- "agreement": "agree" | "partially_agree" | "conflict" | "no_comparable_evidence"
- "evidence": string
- "confidence": "high" | "medium" | "low\""""


_TITLE_ONLY_POPULATION_MODALITY = "insufficient snippet text — title only"
_TITLE_ONLY_EVIDENCE = (
    "Candidate snippet contains only the paper title, with no body or abstract "
    "text to judge against."
)


def _title_only_result(candidate: dict[str, Any]) -> dict[str, Any]:
    """Hard floor for candidates with has_body_text=False.

    Enforced in Python, not left to the LLM to infer from a bare title —
    mirrors the empty-extracted_claim -> no_comparable_evidence rule already
    enforced below. No LLM call is made for these candidates.
    """
    return {
        "corpus_id": candidate.get("corpus_id"),
        "title": candidate.get("title"),
        "population_modality": _TITLE_ONLY_POPULATION_MODALITY,
        "extracted_claim": "",
        "comparison_type": "not_comparable",
        "agreement": "no_comparable_evidence",
        "evidence": _TITLE_ONLY_EVIDENCE,
        "confidence": "low",
    }


async def _judge_candidate(
    target_claim: str, candidate: dict[str, Any], client: ModelClient
) -> dict[str, Any]:
    """Make one LLM call to judge a single candidate against the target claim.

    Short-circuits before the LLM call when has_body_text is explicitly
    False — title-only candidates get a fixed not_comparable/
    no_comparable_evidence result instead of asking the LLM to guess from a
    bare title. Absent or True has_body_text goes through the normal path.
    """
    if candidate.get("has_body_text") is False:
        return _title_only_result(candidate)

    response = await client.complete(
        messages=[
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    target_claim=target_claim,
                    title=candidate.get("title"),
                    snippet=candidate.get("snippet"),
                ),
            }
        ],
        system=_JUDGE_SYSTEM,
        max_tokens=1024,
    )
    parsed = _extract_json_object(response.content)

    extracted_claim = str(parsed.get("extracted_claim", ""))
    agreement = str(parsed.get("agreement", ""))
    if not extracted_claim:
        agreement = "no_comparable_evidence"

    return {
        "corpus_id": candidate.get("corpus_id"),
        "title": candidate.get("title"),
        "population_modality": str(parsed.get("population_modality", "")),
        "extracted_claim": extracted_claim,
        "comparison_type": str(parsed.get("comparison_type", "")),
        "agreement": agreement,
        "evidence": str(parsed.get("evidence", "")),
        "confidence": str(parsed.get("confidence", "low")),
    }


async def extract_and_compare(
    target_claim: str,
    candidates: list[dict[str, Any]],
    client: ModelClient,
) -> list[dict[str, Any]]:
    """Judge each candidate against target_claim with one LLM call per candidate.

    One candidate's LLM call or JSON-parsing failure never affects the others —
    it becomes an error entry instead of raising and losing the whole batch,
    mirroring ChecklistTool's failure-isolation philosophy. Order is preserved;
    no dedup/filtering happens here (already done in distill_and_retrieve).
    """
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            result = await _judge_candidate(target_claim, candidate, client)
        except Exception as exc:
            result = {
                "corpus_id": candidate.get("corpus_id"),
                "title": candidate.get("title"),
                "error": str(exc),
            }
        results.append(result)
    return results
