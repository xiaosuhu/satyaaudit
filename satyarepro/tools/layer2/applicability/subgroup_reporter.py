from __future__ import annotations

from satyarepro.client.base import ModelClient
from satyarepro.types import ToolSchema

from ...base import Tool

_SYSTEM = (
    "You are a biomedical AI fairness auditor specialising in demographic subgroup analysis "
    "and equitable ML reporting."
)

_PROMPT_TEMPLATE = """\
Analyse the following Python biomedical ML code for demographic subgroup performance reporting.

Check whether the code evaluates and reports performance separately for:
1. Age groups (e.g. paediatric, adult, elderly)
2. Sex / gender
3. Race / ethnicity
4. Disease severity subgroups
5. Any statistical tests for subgroup differences (e.g. interaction tests)

For each dimension report: present (yes/partial/no), evidence or location in the code, and what is missing.
Note any fairness or equity concerns.
Be concise: report only actionable findings, skip boilerplate.

```python
{code}
```"""


class SubgroupReporter(Tool):
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
            name="subgroup_reporter",
            description=(
                "Check whether the code reports model performance across demographic subgroups "
                "(age, sex, race) as required by biomedical AI reporting standards."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code to analyse."},
                },
                "required": ["code"],
            },
        )

    async def execute(self, code: str) -> str:
        client = await self._get_client()
        response = await client.complete(
            messages=[{"role": "user", "content": _PROMPT_TEMPLATE.format(code=code)}],
            system=_SYSTEM,
            max_tokens=6144,
        )
        return response.content
