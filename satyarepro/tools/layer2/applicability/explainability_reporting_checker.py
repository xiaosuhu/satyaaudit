from __future__ import annotations

import json

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ...base import Tool
from ..._llm_utils import _extract_json_array

_SYSTEM = (
    "You are a biomedical AI reporting-standards auditor specialising in SHAP (Lundberg & "
    "Lee) and LIME (Ribeiro et al.) explainability/interpretability reporting. You judge "
    "whether a study reports the application of an explainability method, if one was "
    "applied — this is descriptive, not prescriptive: you do not assume every study must "
    "use SHAP/LIME, and you do not penalize studies using inherently interpretable models "
    "(e.g. linear/logistic regression, single decision trees) for the absence of SHAP/LIME. "
    "You never fabricate evidence: if a checklist item is not explicitly addressed in the "
    "provided text, you report it as absent (or not_determinable_from_code when only code "
    "is available) rather than inferring intent. You always respond with a single valid "
    "JSON array and nothing else — no prose, no markdown code fences."
)

_CHECKLIST_ITEMS = [
    {
        "id": "explainability_method_applied",
        "name": "Explainability method applied",
        "source": "SHAP (Lundberg & Lee) / LIME (Ribeiro et al.)",
        "guidance": (
            "Is an explainability/interpretability method (e.g. SHAP, LIME, or a "
            "comparable feature-attribution technique) explicitly applied to the model's "
            "predictions?"
        ),
    },
    {
        "id": "feature_attribution_reported",
        "name": "Feature attribution reported",
        "source": "SHAP (Lundberg & Lee) / LIME (Ribeiro et al.)",
        "guidance": (
            "Are the resulting feature importance rankings or per-prediction "
            "attributions (e.g. SHAP values, LIME local weights) actually reported or "
            "visualized, not just mentioned as applied?"
        ),
    },
    {
        "id": "explanation_method_limitations_discussed",
        "name": "Explanation method limitations discussed",
        "source": "SHAP (Lundberg & Lee) / LIME (Ribeiro et al.)",
        "guidance": (
            "Are known limitations or approximation caveats of the interpretability "
            "method discussed (e.g. SHAP's background/reference dataset choice, LIME's "
            "local linear approximation validity)?"
        ),
    },
]

_CHECKLIST_BLOCK = "\n".join(
    f"- {item['id']} — {item['name']} ({item['source']}): {item['guidance']}"
    for item in _CHECKLIST_ITEMS
)

_PROMPT_HEADER = """\
We are auditing a biomedical AI/ML study for explainability method application and \
reporting, per SHAP (Lundberg & Lee) and LIME (Ribeiro et al.).

Checklist items to judge:
{checklist_block}"""

_MANUSCRIPT_ONLY_INSTRUCTIONS = (
    "Judge each checklist item strictly against the manuscript text below. For each item, "
    "cite the specific sentence or section (e.g. \"Results, para 3\") as evidence. If an "
    "item is not addressed anywhere in the text, set status to \"absent\"."
)

_CODE_ONLY_INSTRUCTIONS = (
    "Only source code is available — no manuscript text. Code is a weak proxy for these "
    "narrative reporting items: at most it can show indirect signals (e.g. an import of "
    "the shap or lime library, a call to a SHAP/LIME explainer). It CANNOT show whether "
    "the resulting feature attributions were actually *reported* or visualized, or whether "
    "the method's limitations were *discussed* in a manuscript, since narrative reporting "
    "isn't visible in code. Unless the code gives a clear, explicit signal for an item, set "
    "status to \"not_determinable_from_code\" rather than \"absent\" — \"absent\" should only "
    "be used if the code positively contradicts the item (rare). Set evidence_source to "
    "\"code\" when you cite a code signal, or \"none\" when there is no signal at all."
)

_BOTH_INSTRUCTIONS = (
    "Judge primarily from the manuscript text below. If the code also contains "
    "supplementary evidence for an item (e.g. a shap/lime import or explainer call), note "
    "it in the evidence field in addition to the manuscript evidence, and set "
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


class ExplainabilityReportingChecker(Tool):
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
            name="explainability_reporting_checker",
            description=(
                "Check whether a study reports the application of explainability/"
                "interpretability methods, per SHAP (Lundberg & Lee) and LIME (Ribeiro et "
                "al.). This tool is descriptive, not prescriptive: it reports whether "
                "explainability was applied and documented, not whether the study was "
                "required to use one — explainability necessity depends heavily on model "
                "type (e.g. a simple logistic regression may not need SHAP/LIME the way a "
                "black-box deep learning model would). Prefers manuscript text; falls back "
                "to code as weaker proxy evidence when manuscript text is unavailable."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "manuscript_text": {
                        "type": "string",
                        "description": (
                            "Full or partial manuscript text, Methods/Results sections "
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
