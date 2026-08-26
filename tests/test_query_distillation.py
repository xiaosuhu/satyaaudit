"""Tests for query_distillation (Replicability module — cross-paper claim
comparator, stage 1 only: distill task_description -> query -> retrieval
candidates). No claim extraction/comparison logic here yet.
"""
from __future__ import annotations

from satyarepro.client.mock import MockClient
from satyarepro.tools.layer2.replicability.query_distillation import (
    distill_and_retrieve,
    distill_query,
)
from satyarepro.types import CompletionResponse, Usage


def _ok_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        content=text,
        raw_content=[{"type": "text", "text": text}],
        usage=Usage(10, 5),
    )


class FakeAstaClient:
    """Duck-typed stand-in for AstaClient — no MCP/network involved."""

    def __init__(self, snippet_results: list[dict] | None = None) -> None:
        self._snippet_results = snippet_results or []
        self.search_calls: list[dict] = []

    async def snippet_search(self, query: str, limit: int = 10, **extra):
        self.search_calls.append({"query": query, "limit": limit, **extra})
        return self._snippet_results

    async def get_citations(self, doi: str, limit: int = 20, **extra):
        return []


# ══════════════════════════════════════════════════════════════════════════════
# distill_query — MockClient, prompt routing + return-value normalization
# ══════════════════════════════════════════════════════════════════════════════

class TestDistillQuery:
    async def test_task_description_forwarded_to_prompt(self):
        mock = MockClient()
        mock.enqueue(_ok_response("preterm birth prediction from EHG signals"))
        await distill_query("UNIQUE_TASK_MARKER using SMOTE and XGBoost, AUC 0.9", client=mock)
        assert "UNIQUE_TASK_MARKER" in mock.calls[0]["messages"][0]["content"]

    async def test_returns_plain_string(self):
        mock = MockClient()
        mock.enqueue(_ok_response("preterm birth prediction from EHG signals"))
        result = await distill_query("some task", client=mock)
        assert isinstance(result, str)
        assert result == "preterm birth prediction from EHG signals"

    async def test_strips_surrounding_whitespace_and_newlines(self):
        mock = MockClient()
        mock.enqueue(_ok_response("\n  preterm birth prediction   \n  from EHG signals  \n"))
        result = await distill_query("some task", client=mock)
        assert result == "preterm birth prediction from EHG signals"
        assert "\n" not in result

    async def test_collapses_internal_multiple_spaces(self):
        mock = MockClient()
        mock.enqueue(_ok_response("preterm   birth    prediction"))
        result = await distill_query("some task", client=mock)
        assert result == "preterm birth prediction"

    async def test_uses_complete_not_complete_with_tools(self):
        mock = MockClient()
        mock.enqueue(_ok_response("distilled"))
        await distill_query("some task", client=mock)
        assert mock.calls[0]["type"] == "complete"


# ══════════════════════════════════════════════════════════════════════════════
# distill_and_retrieve — MockClient + FakeAstaClient, two-step call chaining
# ══════════════════════════════════════════════════════════════════════════════

class TestDistillAndRetrieve:
    async def test_calls_distill_then_search_with_distilled_query(self):
        mock = MockClient()
        mock.enqueue(_ok_response("preterm birth prediction from EHG signals"))
        asta = FakeAstaClient(snippet_results=[])

        await distill_and_retrieve(
            task_description="predict preterm birth using SMOTE-balanced XGBoost, AUC 0.9",
            client=mock,
            asta_client=asta,
        )

        assert mock.calls[0]["type"] == "complete"
        assert len(asta.search_calls) == 1
        assert asta.search_calls[0]["query"] == "preterm birth prediction from EHG signals"

    async def test_limit_forwarded_to_snippet_search(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(snippet_results=[])

        await distill_and_retrieve(task_description="task", client=mock, asta_client=asta, limit=7)
        # Internally over-fetches (3x limit) to leave headroom for corpus_id
        # dedup and the has_body_text/exclude_corpus_id preference — the raw
        # Asta call limit is not the same as the returned count.
        assert asta.search_calls[0]["limit"] == 21

    async def test_default_limit_is_five(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(snippet_results=[])

        await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert asta.search_calls[0]["limit"] == 15

    async def test_normalizes_real_nested_asta_shape(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {
                    "score": 0.9,
                    "paper": {"corpusId": "123", "title": "Some Paper", "openAccessInfo": {}},
                    "snippet": {"text": "some snippet text"},
                }
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert len(results) == 1
        assert results[0]["corpus_id"] == "123"
        assert results[0]["title"] == "Some Paper"
        assert results[0]["snippet"] == "some snippet text"

    async def test_normalizes_flat_shape(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[{"corpus_id": "999", "title": "Flat Paper", "snippet": "flat text"}]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results[0]["corpus_id"] == "999"
        assert results[0]["title"] == "Flat Paper"
        assert results[0]["snippet"] == "flat text"

    async def test_empty_search_results(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(snippet_results=[])

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results == []

    async def test_raw_item_preserved(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        raw_item = {"paper": {"corpusId": "1", "title": "T"}, "snippet": {"text": "S"}, "score": 0.5}
        asta = FakeAstaClient(snippet_results=[raw_item])

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results[0]["raw"] == raw_item

    async def test_multiple_candidates_all_normalized(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {"paper": {"corpusId": "1", "title": "A"}, "snippet": {"text": "a"}},
                {"paper": {"corpusId": "2", "title": "B"}, "snippet": {"text": "b"}},
                {"paper": {"corpusId": "3", "title": "C"}, "snippet": {"text": "c"}},
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert [r["corpus_id"] for r in results] == ["1", "2", "3"]


# ══════════════════════════════════════════════════════════════════════════════
# distill_and_retrieve — corpus_id dedup (Asta can return multiple snippets
# from the same paper; dedup must not shrink results below limit unnecessarily)
# ══════════════════════════════════════════════════════════════════════════════

class TestDistillAndRetrieveDedup:
    async def test_dedupes_by_corpus_id_keeping_highest_score(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {
                    "score": 0.5,
                    "paper": {"corpusId": "1", "title": "Paper One"},
                    "snippet": {"text": "low score snippet"},
                },
                {
                    "score": 0.9,
                    "paper": {"corpusId": "1", "title": "Paper One"},
                    "snippet": {"text": "high score snippet"},
                },
                {
                    "score": 0.3,
                    "paper": {"corpusId": "1", "title": "Paper One"},
                    "snippet": {"text": "another low score snippet"},
                },
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta, limit=5)

        assert len(results) == 1
        assert results[0]["corpus_id"] == "1"
        assert results[0]["snippet"] == "high score snippet"
        assert results[0]["score"] == 0.9

    async def test_dedupe_keeps_first_occurrence_when_no_score(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {"paper": {"corpusId": "1", "title": "Paper One"}, "snippet": {"text": "first snippet"}},
                {"paper": {"corpusId": "1", "title": "Paper One"}, "snippet": {"text": "second snippet"}},
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta, limit=5)

        assert len(results) == 1
        assert results[0]["snippet"] == "first snippet"

    async def test_dedupe_keeps_first_occurrence_when_scores_tied(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {"score": 0.7, "paper": {"corpusId": "1", "title": "Paper One"}, "snippet": {"text": "first"}},
                {"score": 0.7, "paper": {"corpusId": "1", "title": "Paper One"}, "snippet": {"text": "second"}},
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta, limit=5)

        assert len(results) == 1
        assert results[0]["snippet"] == "first"

    async def test_dedup_backfills_to_limit_when_enough_distinct_papers(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        # 8 raw results, 6 distinct corpus_ids ("1" appears 3 times). With
        # limit=5, dedup must return 5 distinct papers, not be short-changed
        # down to fewer just because 2 of the "1" duplicates ate into the
        # raw result list.
        snippet_results = [
            {"score": 0.9, "paper": {"corpusId": "1", "title": "P1"}, "snippet": {"text": "s1a"}},
            {"score": 0.8, "paper": {"corpusId": "2", "title": "P2"}, "snippet": {"text": "s2"}},
            {"score": 0.7, "paper": {"corpusId": "1", "title": "P1"}, "snippet": {"text": "s1b"}},
            {"score": 0.6, "paper": {"corpusId": "3", "title": "P3"}, "snippet": {"text": "s3"}},
            {"score": 0.5, "paper": {"corpusId": "1", "title": "P1"}, "snippet": {"text": "s1c"}},
            {"score": 0.4, "paper": {"corpusId": "4", "title": "P4"}, "snippet": {"text": "s4"}},
            {"score": 0.3, "paper": {"corpusId": "5", "title": "P5"}, "snippet": {"text": "s5"}},
            {"score": 0.2, "paper": {"corpusId": "6", "title": "P6"}, "snippet": {"text": "s6"}},
        ]
        asta = FakeAstaClient(snippet_results=snippet_results)

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta, limit=5)

        assert len(results) == 5
        corpus_ids = [r["corpus_id"] for r in results]
        assert len(set(corpus_ids)) == 5
        assert corpus_ids.count("1") == 1
        assert results[0]["corpus_id"] == "1"
        assert results[0]["snippet"] == "s1a"  # highest score among the "1" duplicates


# ══════════════════════════════════════════════════════════════════════════════
# _extract_candidate (via distill_and_retrieve) — has_body_text derivation
# from Asta's snippetKind field (title-only snippets get inconsistent LLM
# judgments downstream, so this is used as a hard floor in claim_comparison)
# ══════════════════════════════════════════════════════════════════════════════

class TestHasBodyText:
    async def test_title_snippet_kind_gets_has_body_text_false(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {
                    "paper": {"corpusId": "1", "title": "Title Only Paper"},
                    "snippet": {"text": "Title Only Paper", "snippetKind": "title"},
                }
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results[0]["has_body_text"] is False

    async def test_body_snippet_kind_gets_has_body_text_true(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {
                    "paper": {"corpusId": "1", "title": "Body Paper"},
                    "snippet": {"text": "actual body text here", "snippetKind": "body"},
                }
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results[0]["has_body_text"] is True

    async def test_abstract_snippet_kind_gets_has_body_text_true(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {
                    "paper": {"corpusId": "1", "title": "Abstract Paper"},
                    "snippet": {"text": "abstract text here", "snippetKind": "abstract"},
                }
            ]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results[0]["has_body_text"] is True

    async def test_missing_snippet_kind_defaults_has_body_text_true(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        # Old-style flat fixture, predating snippetKind — must not be assumed
        # title-only just because the field is absent.
        asta = FakeAstaClient(
            snippet_results=[{"corpus_id": "999", "title": "Flat Paper", "snippet": "flat text"}]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert results[0]["has_body_text"] is True


# ══════════════════════════════════════════════════════════════════════════════
# distill_and_retrieve — exclude_corpus_id (mirrors outcome_distribution_
# checker.py's pattern: keep the audited paper's own corpus_id out of its
# own comparator set)
# ══════════════════════════════════════════════════════════════════════════════

class TestExcludeCorpusId:
    async def test_exclude_corpus_id_filters_matching_result(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[
                {"paper": {"corpusId": "1", "title": "Self Paper"}, "snippet": {"text": "self citation"}},
                {"paper": {"corpusId": "2", "title": "Other Paper"}, "snippet": {"text": "other"}},
            ]
        )

        results = await distill_and_retrieve(
            task_description="task", client=mock, asta_client=asta, exclude_corpus_id="1"
        )
        assert [r["corpus_id"] for r in results] == ["2"]

    async def test_exclude_corpus_id_none_excludes_nothing(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        asta = FakeAstaClient(
            snippet_results=[{"paper": {"corpusId": "1", "title": "Paper"}, "snippet": {"text": "text"}}]
        )

        results = await distill_and_retrieve(task_description="task", client=mock, asta_client=asta)
        assert [r["corpus_id"] for r in results] == ["1"]


# ══════════════════════════════════════════════════════════════════════════════
# distill_and_retrieve — combined: exclude_corpus_id + has_body_text
# preference both apply before truncation to `limit`
# ══════════════════════════════════════════════════════════════════════════════

class TestBodyTextPreferenceAndExclusionCombined:
    async def test_prefers_body_text_and_excludes_self_citation(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        snippet_results = [
            {
                "paper": {"corpusId": "SELF", "title": "Self"},
                "snippet": {"text": "self citation body", "snippetKind": "body"},
            },
            {
                "paper": {"corpusId": "T1", "title": "Title Only 1"},
                "snippet": {"text": "Title Only 1", "snippetKind": "title"},
            },
            {
                "paper": {"corpusId": "T2", "title": "Title Only 2"},
                "snippet": {"text": "Title Only 2", "snippetKind": "title"},
            },
            {
                "paper": {"corpusId": "B1", "title": "Body 1"},
                "snippet": {"text": "body text 1", "snippetKind": "body"},
            },
            {
                "paper": {"corpusId": "B2", "title": "Body 2"},
                "snippet": {"text": "body text 2", "snippetKind": "abstract"},
            },
        ]
        asta = FakeAstaClient(snippet_results=snippet_results)

        results = await distill_and_retrieve(
            task_description="task",
            client=mock,
            asta_client=asta,
            limit=2,
            exclude_corpus_id="SELF",
        )

        corpus_ids = [r["corpus_id"] for r in results]
        assert len(results) == 2
        assert "SELF" not in corpus_ids
        assert all(r["has_body_text"] for r in results)
        assert set(corpus_ids) == {"B1", "B2"}

    async def test_title_only_fills_remaining_slots_when_not_enough_body_text(self):
        mock = MockClient()
        mock.enqueue(_ok_response("some distilled query"))
        snippet_results = [
            {
                "paper": {"corpusId": "B1", "title": "Body 1"},
                "snippet": {"text": "body text 1", "snippetKind": "body"},
            },
            {
                "paper": {"corpusId": "T1", "title": "Title Only 1"},
                "snippet": {"text": "Title Only 1", "snippetKind": "title"},
            },
            {
                "paper": {"corpusId": "T2", "title": "Title Only 2"},
                "snippet": {"text": "Title Only 2", "snippetKind": "title"},
            },
        ]
        asta = FakeAstaClient(snippet_results=snippet_results)

        results = await distill_and_retrieve(
            task_description="task", client=mock, asta_client=asta, limit=3
        )

        corpus_ids = [r["corpus_id"] for r in results]
        assert len(results) == 3
        assert set(corpus_ids) == {"B1", "T1", "T2"}
