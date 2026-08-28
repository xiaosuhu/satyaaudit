from __future__ import annotations

import json
from typing import Any

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ...base import Tool

_SYSTEM = (
    "You are a biomedical AI reporting-standards auditor specialising in TRIPOD-AI and "
    "CONSORT-AI applicability and external validity reporting. You judge whether a study "
    "adequately reports the boundaries of where its findings apply — target population, "
    "eligibility criteria, external validation, and limitations. "
    "You never fabricate evidence: if a checklist item is not explicitly addressed in the "
    "provided text, you report it as absent (or not_determinable_from_code when only code "
    "is available) rather than inferring intent. You always respond with a single valid "
    "JSON array and nothing else — no prose, no markdown code fences."
)

_CHECKLIST_ITEMS = [
    {
        "id": "target_population_scope",
        "name": "Target population / clinical scope stated",
        "guidance": (
            "The target population and clinical scope (disease/condition, care setting, "
            "intended patient group) for which the model is applicable should be explicitly "
            "stated (TRIPOD-AI Discussion; CONSORT-AI)."
        ),
    },
    {
        "id": "inclusion_exclusion_criteria",
        "name": "Inclusion/exclusion criteria described",
        "guidance": (
            "The inclusion and exclusion criteria used to select the study population or "
            "dataset should be described (TRIPOD-AI Methods/Discussion)."
        ),
    },
    {
        "id": "external_validation",
        "name": "External validation reported",
        "guidance": (
            "Whether the model was evaluated on an external, independent dataset or site "
            "distinct from the development data should be reported (TRIPOD-AI Discussion)."
        ),
    },
    {
        "id": "limitations_discussed",
        "name": "Limitations discussed",
        "guidance": (
            "Study limitations, including generalizability and applicability constraints, "
            "should be discussed (TRIPOD-AI Discussion; CONSORT-AI)."
        ),
    },
]

_CHECKLIST_BLOCK = "\n".join(
    f"- {item['id']} — {item['name']}: {item['guidance']}" for item in _CHECKLIST_ITEMS
)

_PROMPT_HEADER = """\
We are auditing a biomedical AI/ML study for applicability boundary reporting, per \
TRIPOD-AI Discussion items and the CONSORT-AI extension.

Checklist items to judge:
{checklist_block}"""

_MANUSCRIPT_ONLY_INSTRUCTIONS = (
    "Judge each checklist item strictly against the manuscript text below. For each item, "
    "cite the specific sentence or section (e.g. \"Discussion, para 2\") as evidence. If an "
    "item is not addressed anywhere in the text, set status to \"absent\"."
)

_CODE_ONLY_INSTRUCTIONS = (
    "Only source code is available — no manuscript text. Code is a weak proxy for these "
    "narrative reporting items: at most it can show indirect signals (e.g. a held-out "
    "external dataset file/path, a train/test split by site). It CANNOT show whether the "
    "target population, inclusion/exclusion criteria, or limitations were actually "
    "*reported* in a manuscript, since narrative reporting isn't visible in code. Unless "
    "the code gives a clear, explicit signal for an item, set status to "
    "\"not_determinable_from_code\" rather than \"absent\" — \"absent\" should only be used "
    "if the code positively contradicts the item (rare). Set evidence_source to \"code\" "
    "when you cite a code signal, or \"none\" when there is no signal at all."
)

_BOTH_INSTRUCTIONS = (
    "Judge primarily from the manuscript text below. If the code also contains "
    "supplementary evidence for an item (e.g. an external validation split), note it in "
    "the evidence field in addition to the manuscript evidence, and set evidence_source to "
    "\"manuscript\" (or \"code\" only if that item has no manuscript evidence at all)."
)

_OUTPUT_INSTRUCTIONS = """\
Respond with ONLY a JSON array of exactly {n} objects, one per checklist item above, in the \
same order, each with these fields:
- "item_id": one of {ids}.
- "status": "present", "partial", "absent", or "not_determinable_from_code".
- "evidence_source": "manuscript", "code", or "none".
- "evidence": a short quote or paraphrase supporting your judgment, or "" if none.
- "reasoning": one sentence explaining the status determination."""


class ReportingCompletenessChecker(Tool):
    def __init__(self, client: ModelClient | None = None) -> None:
        self._client = client

    async def _get_client(self) -> ModelClient:
        if self._client is not None:
            return self._client
        from satyarepro.client.claude import ClaudeClient
        return ClaudeClient()

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="reporting_completeness_checker",
            description=(
                "Check whether a study's applicability boundaries — target population, "
                "inclusion/exclusion criteria, external validation, and limitations — are "
                "adequately reported, per TRIPOD-AI Discussion items and CONSORT-AI. Prefers "
                "manuscript text; falls back to code as weaker proxy evidence when "
                "manuscript text is unavailable."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "manuscript_text": {
                        "type": "string",
                        "description": (
                            "Full or partial manuscript text, Methods/Discussion sections "
                            "preferred."
                        ),
                    },
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source code, used as fallback evidence when manuscript "
                            "text is unavailable."
                        ),
                    },
                },
                "required": [],
            },
        )

    async def execute(self, manuscript_text: str = "", code: str = "") -> str:
        manuscript_text = manuscript_text or ""
        code = code or ""
        if not manuscript_text.strip() and not code.strip():
            raise ValueError("At least one of manuscript_text or code must be provided.")

        if manuscript_text.strip() and code.strip():
            mode_instructions = _BOTH_INSTRUCTIONS
            evidence_block = (
                f'MANUSCRIPT TEXT:\n"""\n{manuscript_text}\n"""\n\n'
                f"CODE (supplementary evidence only):\n```python\n{code}\n```"
            )
        elif manuscript_text.strip():
            mode_instructions = _MANUSCRIPT_ONLY_INSTRUCTIONS
            evidence_block = f'MANUSCRIPT TEXT:\n"""\n{manuscript_text}\n"""'
        else:
            mode_instructions = _CODE_ONLY_INSTRUCTIONS
            evidence_block = (
                "CODE (weak proxy evidence only, no manuscript text provided):\n"
                f"```python\n{code}\n```"
            )

        item_ids = ", ".join(f'"{item["id"]}"' for item in _CHECKLIST_ITEMS)
        prompt = "\n\n".join(
            [
                _PROMPT_HEADER.format(checklist_block=_CHECKLIST_BLOCK),
                mode_instructions,
                evidence_block,
                _OUTPUT_INSTRUCTIONS.format(n=len(_CHECKLIST_ITEMS), ids=item_ids),
            ]
        )

        client = await self._get_client()
        response = await client.complete(
            messages=[{"role": "user", "content": prompt}],
            system=_SYSTEM,
            max_tokens=2048,
        )
        parsed = _extract_json_array(response.content)
        return json.dumps(parsed, indent=2)


def _extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Could not find a JSON array in LLM response: {text!r}")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
    return parsed
