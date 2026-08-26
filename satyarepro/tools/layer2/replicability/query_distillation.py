from __future__ import annotations

from typing import Any

from satyarepro.client.asta import AstaClient
from satyarepro.client.base import ModelClient

_DISTILL_SYSTEM = (
    "You distill research task descriptions into short literature-search queries. "
    "You keep only the core research topic — the predicted outcome and the input "
    "modality/data type — and strip out method names, evaluation metric names, and "
    "other technical/implementation detail, because task-oriented queries retrieve "
    "more relevant results from the search backend than queries mixed with "
    "method/metric terms."
)

_DISTILL_PROMPT = """\
Distill the following research task description into a short literature-search query.

Task description:
\"\"\"
{task_description}
\"\"\"

Keep ONLY the core research topic: the predicted outcome/label and the input modality \
or data type. Remove method names (e.g. SMOTE, XGBoost, random forest, ResNet), \
evaluation metric names (e.g. AUC, F1, accuracy, sensitivity), and any other \
technical/implementation detail.

Respond with ONLY the distilled query text — a short phrase, no quotes, no explanation, \
no markdown, no trailing punctuation."""


async def distill_query(task_description: str, client: ModelClient) -> str:
    """Distill a task description down to a short, task-oriented Asta search query."""
    response = await client.complete(
        messages=[
            {
                "role": "user",
                "content": _DISTILL_PROMPT.format(task_description=task_description),
            }
        ],
        system=_DISTILL_SYSTEM,
        max_tokens=256,
    )
    return " ".join(response.content.split())


def _extract_candidate(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one Asta snippet_search item to a flat candidate dict.

    Adapted from outcome_distribution_checker._format_snippets' field-extraction
    logic: handles both the real nested Asta shape ({"paper": {...}, "snippet": {...}})
    and flat dicts (e.g. test fixtures). The original item is kept under "raw" for
    downstream stages (claim extraction) that may need fields not surfaced here.

    "has_body_text" is derived from the snippet's snippetKind: False only when
    Asta explicitly reports "title" (the snippet is just the paper title
    repeated, with no body/abstract content to judge a claim against).
    Defaults to True whenever snippetKind can't be determined (e.g. flat test
    fixtures predating this field) — a candidate is only assumed title-only
    when we have positive evidence of that, never by absence of the field.
    """
    paper = item["paper"] if isinstance(item.get("paper"), dict) else item
    corpus_id = paper.get("corpusId", paper.get("corpus_id"))
    title = paper.get("title")
    doi = paper.get("doi")
    snippet_obj = item.get("snippet")
    snippet_kind = None
    if isinstance(snippet_obj, dict):
        text = snippet_obj.get("text")
        snippet_kind = snippet_obj.get("snippetKind")
    elif isinstance(snippet_obj, str):
        text = snippet_obj
    else:
        text = item.get("text") or item.get("abstract")
    has_body_text = snippet_kind != "title" if snippet_kind is not None else True
    return {
        "corpus_id": corpus_id,
        "title": title,
        "doi": doi,
        "snippet": text,
        "score": item.get("score"),
        "has_body_text": has_body_text,
        "raw": item,
    }


def _dedupe_by_corpus_id(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep at most one candidate per corpus_id — the highest-scoring one.

    Asta's snippet_search can return multiple snippets from the same paper
    (different passages matched the query), which would otherwise eat into
    the candidate budget available for cross-paper comparison. If score is
    missing or tied, the first-seen candidate for that corpus_id is kept.
    A candidate with no corpus_id at all is passed through unmerged, since
    there's nothing reliable to dedupe it on.
    """
    best_by_id: dict[Any, dict[str, Any]] = {}
    for c in candidates:
        cid = c.get("corpus_id")
        if not cid:
            continue
        existing = best_by_id.get(cid)
        if existing is None:
            best_by_id[cid] = c
            continue
        existing_score = existing.get("score")
        new_score = c.get("score")
        if new_score is not None and (existing_score is None or new_score > existing_score):
            best_by_id[cid] = c

    seen: set[Any] = set()
    deduped: list[dict[str, Any]] = []
    for c in candidates:
        cid = c.get("corpus_id")
        if not cid:
            deduped.append(c)
            continue
        if cid in seen:
            continue
        seen.add(cid)
        deduped.append(best_by_id[cid])
    return deduped


# Over-fetch generously enough that excluding the audited paper's own
# corpus_id (self-citation) plus filtering out title-only candidates still
# leaves good odds of reaching `limit` substantive, non-excluded candidates.
_OVER_FETCH_MULTIPLIER = 3


def _normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _prefer_body_text(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Truncate to `limit`, preferring has_body_text=True candidates.

    Title-only candidates (has_body_text=False) only fill remaining slots
    if there aren't enough substantive candidates to fill `limit` on their
    own. Relative order within each group is preserved.
    """
    if len(candidates) <= limit:
        return candidates
    with_body = [c for c in candidates if c.get("has_body_text", True)]
    title_only = [c for c in candidates if not c.get("has_body_text", True)]
    return (with_body + title_only)[:limit]


async def distill_and_retrieve(
    task_description: str,
    client: ModelClient,
    asta_client: AstaClient,
    limit: int = 5,
    exclude_corpus_id: str | None = None,
) -> list[dict[str, Any]]:
    """Distill task_description into a query, then run it through Asta snippet_search.

    No same_task/claim judgment happens here — this just gets retrieval candidates
    with a reliable corpus_id/title/snippet on top, for a later stage to reason over.

    Over-fetches from Asta (_OVER_FETCH_MULTIPLIER x limit) and dedupes by
    corpus_id before truncating to `limit`, so that duplicate snippets from
    the same paper don't shrink the final candidate count below `limit` when
    enough distinct papers exist.

    `exclude_corpus_id` filters out a known self-citation (e.g. the corpus_id
    of the paper under audit) before dedup/truncation, mirroring the
    exclude_corpus_id pattern in outcome_distribution_checker.py. Truncation
    then prefers has_body_text=True candidates over title-only ones (see
    _prefer_body_text) — both the exclusion and this preference happen before
    the over-fetch headroom is spent.
    """
    query = await distill_query(task_description, client)
    over_fetch_limit = limit * _OVER_FETCH_MULTIPLIER
    results = await asta_client.snippet_search(query, limit=over_fetch_limit)
    candidates = [_extract_candidate(r) for r in results]

    exclude_norm = _normalize_id(exclude_corpus_id)
    if exclude_norm:
        candidates = [c for c in candidates if _normalize_id(c.get("corpus_id")) != exclude_norm]

    deduped = _dedupe_by_corpus_id(candidates)
    return _prefer_body_text(deduped, limit)
