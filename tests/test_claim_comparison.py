"""Tests for claim_comparison (Replicability module — cross-paper claim
comparator, stage 2: extract each candidate's directional claim and judge it
against the target claim). Consumes distill_and_retrieve's candidate list
(stage 1, tested in test_query_distillation.py) but does not call it here.
"""
from __future__ import annotations

import json

from satyarepro.client.mock import MockClient
from satyarepro.tools.layer2.replicability.claim_comparison import extract_and_compare
from satyarepro.types import CompletionResponse, Usage


def _ok_response(payload: dict) -> CompletionResponse:
    text = json.dumps(payload)
    return CompletionResponse(
        content=text,
        raw_content=[{"type": "text", "text": text}],
        usage=Usage(10, 5),
    )


def _candidate(corpus_id: str, title: str, snippet: str) -> dict:
    return {
        "corpus_id": corpus_id,
        "title": title,
        "doi": f"10.0000/{corpus_id}",
        "snippet": snippet,
        "score": 0.5,
        "raw": {},
    }


_FULL_JUDGMENT = {
    "population_modality": "adult ICU patients, tabular EHR data",
    "extracted_claim": "Oversampling improves recall on the minority class.",
    "comparison_type": "direct",
    "agreement": "agree",
    "evidence": "authors report improved recall after SMOTE",
    "confidence": "high",
}


# ══════════════════════════════════════════════════════════════════════════════
# extract_and_compare — happy path, single candidate
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareHappyPath:
    async def test_single_candidate_all_fields_passed_through(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [_candidate("1", "Paper One", "some snippet text")]

        results = await extract_and_compare(
            target_claim="Oversampling improves recall", candidates=candidates, client=mock
        )

        assert len(results) == 1
        result = results[0]
        assert result["corpus_id"] == "1"
        assert result["title"] == "Paper One"
        assert result["population_modality"] == _FULL_JUDGMENT["population_modality"]
        assert result["extracted_claim"] == _FULL_JUDGMENT["extracted_claim"]
        assert result["comparison_type"] == "direct"
        assert result["agreement"] == "agree"
        assert result["evidence"] == _FULL_JUDGMENT["evidence"]
        assert result["confidence"] == "high"
        assert "error" not in result


# ══════════════════════════════════════════════════════════════════════════════
# extract_and_compare — one LLM call per candidate, isolated prompts
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareMultipleCandidates:
    async def test_one_llm_call_per_candidate(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [
            _candidate("1", "Paper One", "SNIPPET_MARKER_ONE"),
            _candidate("2", "Paper Two", "SNIPPET_MARKER_TWO"),
            _candidate("3", "Paper Three", "SNIPPET_MARKER_THREE"),
        ]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert len(mock.calls) == len(candidates)
        assert len(results) == 3

    async def test_each_call_prompt_isolated_to_its_own_candidate(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [
            _candidate("1", "Paper One Title", "SNIPPET_MARKER_ONE"),
            _candidate("2", "Paper Two Title", "SNIPPET_MARKER_TWO"),
        ]

        await extract_and_compare(target_claim="some claim", candidates=candidates, client=mock)

        first_prompt = mock.calls[0]["messages"][0]["content"]
        second_prompt = mock.calls[1]["messages"][0]["content"]
        assert "SNIPPET_MARKER_ONE" in first_prompt
        assert "Paper One Title" in first_prompt
        assert "SNIPPET_MARKER_TWO" not in first_prompt
        assert "SNIPPET_MARKER_TWO" in second_prompt
        assert "Paper Two Title" in second_prompt
        assert "SNIPPET_MARKER_ONE" not in second_prompt

    async def test_order_preserved(self):
        mock = MockClient()
        mock.enqueue(_ok_response({**_FULL_JUDGMENT, "extracted_claim": "claim A"}))
        mock.enqueue(_ok_response({**_FULL_JUDGMENT, "extracted_claim": "claim B"}))
        mock.enqueue(_ok_response({**_FULL_JUDGMENT, "extracted_claim": "claim C"}))
        candidates = [
            _candidate("1", "A", "a"),
            _candidate("2", "B", "b"),
            _candidate("3", "C", "c"),
        ]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert [r["corpus_id"] for r in results] == ["1", "2", "3"]
        assert [r["extracted_claim"] for r in results] == ["claim A", "claim B", "claim C"]


# ══════════════════════════════════════════════════════════════════════════════
# extract_and_compare — empty extracted_claim forces agreement in code
#
# Design decision: the extracted_claim-empty -> agreement="no_comparable_evidence"
# rule is enforced in Python (not left to the prompt alone), matching the task
# spec's explicit instruction that the code must not "just trust the LLM to
# follow the prompt instruction." The mock response below deliberately violates
# the rule (empty extracted_claim but agreement="agree") to prove the override
# happens in code.
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareEmptyClaimEnforcement:
    async def test_empty_extracted_claim_forces_no_comparable_evidence(self):
        mock = MockClient()
        mock.enqueue(
            _ok_response(
                {
                    "population_modality": "adult ICU patients, tabular EHR data",
                    "extracted_claim": "",
                    "comparison_type": "direct",
                    "agreement": "agree",  # deliberately violates the rule
                    "evidence": "no direction stated",
                    "confidence": "medium",
                }
            )
        )
        candidates = [_candidate("1", "Paper One", "snippet with no stated direction")]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert results[0]["extracted_claim"] == ""
        assert results[0]["agreement"] == "no_comparable_evidence"


# ══════════════════════════════════════════════════════════════════════════════
# extract_and_compare — not_comparable passthrough (NOT enforced in code,
# unlike the extracted_claim-empty rule above)
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareNotComparablePassthrough:
    async def test_not_comparable_agreement_passed_through_as_is(self):
        mock = MockClient()
        mock.enqueue(
            _ok_response(
                {
                    "population_modality": "adult ICU patients, tabular EHR data",
                    "extracted_claim": "Some unrelated directional claim.",
                    "comparison_type": "not_comparable",
                    "agreement": "conflict",  # whatever the LLM says, passed through
                    "evidence": "different research question entirely",
                    "confidence": "medium",
                }
            )
        )
        candidates = [_candidate("1", "Paper One", "snippet about a different question")]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert results[0]["comparison_type"] == "not_comparable"
        assert results[0]["agreement"] == "conflict"


# ══════════════════════════════════════════════════════════════════════════════
# extract_and_compare — failure isolation: one candidate's parse failure
# does not affect the others
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareFailureIsolation:
    async def test_unparseable_json_becomes_error_entry_others_unaffected(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        mock.enqueue(
            CompletionResponse(
                content="not valid json at all",
                raw_content=[{"type": "text", "text": "not valid json at all"}],
                usage=Usage(10, 5),
            )
        )
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [
            _candidate("1", "Paper One", "good snippet one"),
            _candidate("2", "Paper Two", "bad snippet"),
            _candidate("3", "Paper Three", "good snippet three"),
        ]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert len(results) == 3

        assert "error" not in results[0]
        assert results[0]["comparison_type"] == "direct"

        assert results[1]["corpus_id"] == "2"
        assert results[1]["title"] == "Paper Two"
        assert "error" in results[1]
        assert "population_modality" not in results[1]
        assert "extracted_claim" not in results[1]
        assert "comparison_type" not in results[1]
        assert "agreement" not in results[1]

        assert "error" not in results[2]
        assert results[2]["comparison_type"] == "direct"


# ══════════════════════════════════════════════════════════════════════════════
# extract_and_compare — empty candidates list
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareEmptyCandidates:
    async def test_empty_candidates_returns_empty_list(self):
        mock = MockClient()
        results = await extract_and_compare(target_claim="some claim", candidates=[], client=mock)
        assert results == []
        assert len(mock.calls) == 0


# ══════════════════════════════════════════════════════════════════════════════
# _judge_candidate — has_body_text=False hard floor
#
# Design decision: title-only Asta snippets (snippetKind == "title") give the
# LLM nothing to judge from but a bare title, which produced inconsistent
# comparison_type labels in a real run. The floor is enforced in Python
# (no LLM call at all) rather than left to a prompt instruction, matching the
# empty-extracted_claim rule above.
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareTitleOnlyFloor:
    async def test_has_body_text_false_skips_llm_call(self):
        mock = MockClient()
        candidates = [{**_candidate("1", "Title Only Paper", "Title Only Paper"), "has_body_text": False}]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert len(mock.calls) == 0
        assert len(results) == 1
        result = results[0]
        assert result["corpus_id"] == "1"
        assert result["title"] == "Title Only Paper"
        assert result["extracted_claim"] == ""
        assert result["comparison_type"] == "not_comparable"
        assert result["agreement"] == "no_comparable_evidence"
        assert result["confidence"] == "low"
        assert result["population_modality"]
        assert result["evidence"]
        assert "error" not in result

    async def test_has_body_text_false_mixed_with_normal_candidates(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [
            _candidate("1", "Paper One", "good snippet one"),
            {**_candidate("2", "Title Only", "Title Only"), "has_body_text": False},
            _candidate("3", "Paper Three", "good snippet three"),
        ]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        # Only the two has_body_text-true/absent candidates hit the LLM.
        assert len(mock.calls) == 2
        assert len(results) == 3
        assert results[0]["comparison_type"] == "direct"
        assert results[1]["comparison_type"] == "not_comparable"
        assert results[1]["agreement"] == "no_comparable_evidence"
        assert results[2]["comparison_type"] == "direct"


# ══════════════════════════════════════════════════════════════════════════════
# _judge_candidate — has_body_text=True or absent: unaffected regression
# check, still goes through the normal LLM-call path
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractAndCompareHasBodyTextTrueOrAbsentUnaffected:
    async def test_has_body_text_true_goes_through_llm(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [{**_candidate("1", "Paper One", "some snippet text"), "has_body_text": True}]

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert len(mock.calls) == 1
        assert results[0]["comparison_type"] == "direct"
        assert results[0]["agreement"] == "agree"

    async def test_has_body_text_absent_defaults_to_llm_path(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_JUDGMENT))
        candidates = [_candidate("1", "Paper One", "some snippet text")]  # no has_body_text key

        results = await extract_and_compare(
            target_claim="some claim", candidates=candidates, client=mock
        )

        assert len(mock.calls) == 1
        assert results[0]["comparison_type"] == "direct"
        assert results[0]["agreement"] == "agree"
