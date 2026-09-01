from __future__ import annotations

import json
import statistics
from typing import Any

from satyarepro.client.asta import AstaClient
from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ...base import Tool
from ..._llm_utils import _extract_json_array

_SYSTEM = (
    "You are a biomedical ML analyst specialising in cross-study outcome distribution checks. "
    "You extract structured comparator data from literature-search snippets to assess whether "
    "a reported model performance metric is consistent with prior published work on the exact "
    "same prediction task. "
    "You never fabricate a value: if a number is not explicitly stated in the provided text, "
    "you report it as null. You always respond with a single valid JSON array and nothing else "
    "— no prose, no markdown code fences."
)

_PROMPT_TEMPLATE = """\
We are running a cross-study outcome distribution check on a reported result.

Target metric under audit: {target_metric_name} = {target_metric_value}
Task under audit: {task_description}

Below are literature-search results retrieved for comparable studies. Each result's \
corpus_id is its ONLY reliable identifier. Any DOI shown is extracted from free text and is \
NOT reliable — never use it to identify, deduplicate, or exclude a paper.

{snippets_block}

For EACH candidate study above, extract one structured record with these fields:
- "title": paper title as given in the result.
- "corpus_id": the corpus_id shown for this result. This is the primary identifier — always \
copy it exactly as given, never invent or guess one.
- "doi": DOI string if present in the result text, else null. This field is unreliable and for \
reference only — it must never be used to identify, deduplicate, or exclude a paper.
- "extracted_value": the reported value for the SAME metric ({target_metric_name}) if it is \
explicitly stated in the snippet text, else null. Never estimate, infer, or guess a value.
- "sample_size": the study's sample size (n) if explicitly stated in the snippet text, else null. \
Never estimate, infer, or guess a value.
- "same_task": true only if the study addresses the EXACT SAME prediction task described above \
— meaning both the same input modality/data type AND the same predicted outcome/label. \
A study on the same disease with a different predicted outcome does NOT count as same_task. \
A study using a similar method on a different modality does NOT count as same_task either. \
Set to false for anything that is merely related (same disease area, similar method, adjacent task).
- "justification": one sentence explaining the same_task determination.

Respond with ONLY a JSON array of these records, one entry per candidate study above. \
If there are no candidate studies, respond with an empty JSON array []."""

_CAVEAT = (
    "This is an automated feasibility-stage cross-study outcome distribution check based on "
    "LLM-extracted values from search snippets, not a substitute for expert literature review "
    "or formal statistical meta-analysis — verify every comparator's value, task match, and "
    "study design manually before relying on it."
)


class OutcomeDistributionChecker(Tool):
    def __init__(
        self,
        client: ModelClient | None = None,
        asta_client: AstaClient | None = None,
    ) -> None:
        self._client = client
        self._asta_client = asta_client

    async def _get_client(self) -> ModelClient:
        if self._client is not None:
            return self._client
        from satyarepro.client.claude import ClaudeClient
        return ClaudeClient()

    async def _get_asta_client(self) -> AstaClient:
        if self._asta_client is not None:
            return self._asta_client
        return AstaClient()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="outcome_distribution_checker",
            description=(
                "Search the literature via Asta for studies on the same prediction task and "
                "check whether a reported metric value falls within the IQR-based range of "
                "comparable published results (a cross-study outcome distribution check). "
                "Feasibility-stage check, not a formal meta-analysis."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "target_metric_name": {
                        "type": "string",
                        "description": "Name of the metric being audited, e.g. 'AUC-ROC'.",
                    },
                    "target_metric_value": {
                        "type": "number",
                        "description": "Reported value of the target metric.",
                    },
                    "task_description": {
                        "type": "string",
                        "description": (
                            "Specific prediction task, including input modality and predicted "
                            "outcome, e.g. '12-lead ECG waveform to predict 1-year mortality'."
                        ),
                    },
                    "exclude_corpus_id": {
                        "type": "string",
                        "description": (
                            "Semantic Scholar corpus_id of the paper under audit, to exclude it "
                            "from its own comparator set. corpus_id is used instead of DOI "
                            "because Asta's snippet_search does not reliably return a DOI."
                        ),
                    },
                    "min_comparators": {
                        "type": "integer",
                        "description": "Minimum number of same-task comparators required to "
                        "compute a range (default 4).",
                    },
                },
                "required": ["target_metric_name", "target_metric_value", "task_description"],
            },
        )

    async def execute(
        self,
        target_metric_name: str,
        target_metric_value: float,
        task_description: str,
        exclude_corpus_id: str | None = None,
        min_comparators: int = 4,
    ) -> str:
        query = f"{task_description} {target_metric_name} performance"

        asta = await self._get_asta_client()
        snippets = await asta.snippet_search(query)

        candidates = await self._extract_candidates(
            query=query,
            snippets=snippets,
            target_metric_name=target_metric_name,
            target_metric_value=target_metric_value,
            task_description=task_description,
        )

        result = _judge(
            query=query,
            target_metric_name=target_metric_name,
            target_metric_value=target_metric_value,
            candidates=candidates,
            exclude_corpus_id=exclude_corpus_id,
            min_comparators=min_comparators,
        )
        return json.dumps(result, indent=2)

    async def _extract_candidates(
        self,
        query: str,
        snippets: list[dict[str, Any]],
        target_metric_name: str,
        target_metric_value: float,
        task_description: str,
    ) -> list[dict[str, Any]]:
        client = await self._get_client()
        prompt = _PROMPT_TEMPLATE.format(
            target_metric_name=target_metric_name,
            target_metric_value=target_metric_value,
            task_description=task_description,
            snippets_block=_format_snippets(snippets),
        )
        response = await client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=_SYSTEM,
            max_tokens=4096,
        )
        return _extract_json_array(response.content)


def _format_snippets(snippets: list[dict[str, Any]]) -> str:
    """Render Asta snippet results for the prompt.

    Handles both the real Asta shape (nested {"paper": {...}, "snippet": {...}})
    and flat dicts (used by tests) — corpusId/corpus_id is the identifier we
    surface as reliable; doi (if present at all) is labeled unreliable.
    """
    if not snippets:
        return "(no search results returned)"
    blocks = []
    for i, s in enumerate(snippets, start=1):
        paper = s["paper"] if isinstance(s.get("paper"), dict) else s
        corpus_id = paper.get("corpusId", paper.get("corpus_id", "(no corpus_id)"))
        title = paper.get("title", "(untitled)")
        doi = paper.get("doi", "(no DOI)")
        snippet_obj = s.get("snippet")
        if isinstance(snippet_obj, dict):
            text = snippet_obj.get("text", "(no snippet text)")
        elif isinstance(snippet_obj, str):
            text = snippet_obj
        else:
            text = s.get("text") or s.get("abstract") or "(no snippet text)"
        blocks.append(
            f"[{i}] corpus_id: {corpus_id}\n"
            f"    Title: {title}\n"
            f"    DOI (unreliable, reference only): {doi}\n"
            f"    Text: {text}"
        )
    return "\n\n".join(blocks)


def _normalize_id(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().lower()


def _judge(
    query: str,
    target_metric_name: str,
    target_metric_value: float,
    candidates: list[dict[str, Any]],
    exclude_corpus_id: str | None,
    min_comparators: int,
) -> dict[str, Any]:
    exclude_norm = _normalize_id(exclude_corpus_id)

    comparators: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for c in candidates:
        corpus_id_norm = _normalize_id(c.get("corpus_id"))
        if exclude_norm and corpus_id_norm == exclude_norm:
            excluded.append({**c, "exclusion_reason": "excluded_corpus_id"})
        elif c.get("same_task") is True:
            comparators.append(c)
        else:
            excluded.append({**c, "exclusion_reason": "different_task"})

    result: dict[str, Any] = {
        "query": query,
        "target_metric": {"name": target_metric_name, "value": target_metric_value},
        "comparators": comparators,
        "excluded_candidates": excluded,
        "computed_range": None,
        "judgment": "insufficient_data",
        "caveats": _CAVEAT,
    }

    if len(comparators) < min_comparators:
        return result

    values = [c["extracted_value"] for c in comparators if isinstance(c.get("extracted_value"), (int, float))]
    if len(values) < 2:
        return result

    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outside = target_metric_value < lower or target_metric_value > upper

    result["computed_range"] = {
        "n": len(values),
        "values": values,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
    }
    result["judgment"] = "outside_range" if outside else "within_range"
    return result


def _format_summary(result_json: str) -> str:
    """Turn an outcome_distribution_checker JSON result into a quick human-readable summary.

    Not a report generator — a scratch summary for manual review before
    trusting the automated judgment.
    """
    data = json.loads(result_json)
    metric = data.get("target_metric", {})
    n_comparators = len(data.get("comparators", []))
    n_excluded = len(data.get("excluded_candidates", []))
    judgment = data.get("judgment")

    lines = [
        f"Query: {data.get('query', '')}",
        f"Target: {metric.get('name')} = {metric.get('value')}",
        f"Comparators: {n_comparators} same-task (excluded {n_excluded})",
    ]
    if judgment == "insufficient_data":
        lines.append("Judgment: insufficient_data — not enough same-task comparators with a value.")
    else:
        rng = data.get("computed_range") or {}
        lines.append(
            f"Judgment: {judgment} — IQR [{rng.get('q1')}, {rng.get('q3')}], "
            f"fence [{rng.get('lower_fence')}, {rng.get('upper_fence')}], n={rng.get('n')}"
        )
    lines.append(f"Caveat: {data.get('caveats', '')}")
    return "\n".join(lines)
