from __future__ import annotations

import json

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ...base import Tool
from ..._llm_utils import _extract_json_array

_SYSTEM = (
    "You are a biomedical AI reporting-standards auditor specialising in Datasheets for "
    "Datasets (Gebru et al.) and Model Cards (Mitchell et al.) documentation completeness. "
    "You judge whether a study adequately documents its dataset and model — what the data "
    "represents and how it was collected, and the model's intended use and training data. "
    "You never fabricate evidence: if a checklist item is not explicitly addressed in the "
    "provided text, you report it as absent (or not_determinable_from_code when only code "
    "is available) rather than inferring intent. You always respond with a single valid "
    "JSON array and nothing else — no prose, no markdown code fences."
)

_CHECKLIST_ITEMS = [
    {
        "id": "dataset_composition_documented",
        "name": "Dataset composition documented",
        "source": "Datasheets for Datasets — Composition",
        "guidance": (
            "Does the study describe what the dataset instances represent, how many there "
            "are, and whether it's a sample of a larger population?"
        ),
    },
    {
        "id": "dataset_collection_process_documented",
        "name": "Dataset collection process documented",
        "source": "Datasheets for Datasets — Collection Process",
        "guidance": (
            "Is it stated how the data was acquired (e.g. direct measurement, EHR "
            "extraction, survey) and over what time period?"
        ),
    },
    {
        "id": "model_intended_use_documented",
        "name": "Model intended use documented",
        "source": "Model Cards — Intended Use",
        "guidance": (
            "Are the primary intended uses and users of the model stated, along with "
            "out-of-scope uses if applicable?"
        ),
    },
    {
        "id": "model_training_data_documented",
        "name": "Model training data documented",
        "source": "Model Cards — Training Data",
        "guidance": (
            "Is the training data source and any preprocessing/feature engineering applied "
            "to it described?"
        ),
    },
]

_CHECKLIST_BLOCK = "\n".join(
    f"- {item['id']} — {item['name']} ({item['source']}): {item['guidance']}"
    for item in _CHECKLIST_ITEMS
)

_PROMPT_HEADER = """\
We are auditing a biomedical AI/ML study for dataset and model documentation completeness, \
per Datasheets for Datasets (Gebru et al.) and Model Cards (Mitchell et al.).

Checklist items to judge:
{checklist_block}"""

_MANUSCRIPT_ONLY_INSTRUCTIONS = (
    "Judge each checklist item strictly against the manuscript text below. For each item, "
    "cite the specific sentence or section (e.g. \"Methods, para 1\") as evidence. If an "
    "item is not addressed anywhere in the text, set status to \"absent\"."
)

_CODE_ONLY_INSTRUCTIONS = (
    "Only source code is available — no manuscript text. Code is a weak proxy for these "
    "narrative documentation items: at most it can show indirect signals (e.g. a data "
    "loading path suggesting a source, a docstring naming intended users). It CANNOT show "
    "whether the dataset composition, collection process, intended use, or training data "
    "were actually *documented* in a datasheet or model card, since narrative documentation "
    "isn't visible in code. Unless the code gives a clear, explicit signal for an item, set "
    "status to \"not_determinable_from_code\" rather than \"absent\" — \"absent\" should only "
    "be used if the code positively contradicts the item (rare). Set evidence_source to "
    "\"code\" when you cite a code signal, or \"none\" when there is no signal at all."
)

_BOTH_INSTRUCTIONS = (
    "Judge primarily from the manuscript text below. If the code also contains "
    "supplementary evidence for an item (e.g. a data loading path or preprocessing step), "
    "note it in the evidence field in addition to the manuscript evidence, and set "
    "evidence_source to \"manuscript\" (or \"code\" only if that item has no manuscript "
    "evidence at all)."
)

_OUTPUT_INSTRUCTIONS = """\
Respond with ONLY a JSON array of exactly {n} objects, one per checklist item above, in the \
same order, each with these fields:
- "item_id": one of {ids}.
- "status": "present", "partial", "absent", or "not_determinable_from_code".
- "evidence_source": "manuscript", "code", or "none".
- "evidence": a short quote or paraphrase supporting your judgment, or "" if none.
- "reasoning": one sentence explaining the status determination."""


class DocumentationCompletenessChecker(Tool):
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
            name="documentation_completeness_checker",
            description=(
                "Check whether a study's dataset and model documentation is adequately "
                "reported, per Datasheets for Datasets (Gebru et al.) and Model Cards "
                "(Mitchell et al.). Prefers manuscript text; falls back to code as weaker "
                "proxy evidence when manuscript text is unavailable."
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
