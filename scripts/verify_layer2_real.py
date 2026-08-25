"""Integration test: run all three Layer 2 tools against a real notebook
using the live Claude API.

Usage:
    python scripts/verify_layer2_real.py
    python scripts/verify_layer2_real.py --notebook fixtures/notebooks/test_notebook.ipynb
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Make the repo root importable when running the script directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

# pydantic-settings loads .env into settings but not os.environ;
# AsyncAnthropic() reads os.environ directly, so bridge the gap here.
from satyarepro.config import settings
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

from satyarepro.client.claude import ClaudeClient
from satyarepro.types import CompletionResponse, ToolSchema
from satyarepro.tools.layer2.hyperparameter_reporter import HyperparameterReporter
from satyarepro.tools.layer2.leakage_detector import LeakageDetector
from satyarepro.tools.layer2.metrics_completeness_checker import MetricsCompletenessChecker
from satyarepro.tools.layer2.provenance_checker import ProvenanceChecker
from satyarepro.tools.layer2.subgroup_reporter import SubgroupReporter
from satyarepro.tools.parsers import parse_input


# ── Usage-tracking wrapper ────────────────────────────────────────────────────

class TrackingClient(ClaudeClient):
    """ClaudeClient that accumulates token usage across all complete() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def _record(self, response: CompletionResponse) -> None:
        self.calls.append({
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens,
            "cache_read": response.usage.cache_read_tokens,
            "cache_creation": response.usage.cache_creation_tokens,
            "stop_reason": response.stop_reason,
        })

    async def complete(self, *args, **kwargs) -> CompletionResponse:
        response = await super().complete(*args, **kwargs)
        self._record(response)
        return response

    async def complete_with_tools(self, *args, **kwargs) -> CompletionResponse:
        response = await super().complete_with_tools(*args, **kwargs)
        self._record(response)
        return response

    @property
    def total(self) -> dict:
        return {
            "input": sum(c["input"] for c in self.calls),
            "output": sum(c["output"] for c in self.calls),
            "cache_read": sum(c["cache_read"] for c in self.calls),
            "cache_creation": sum(c["cache_creation"] for c in self.calls),
            "api_calls": len(self.calls),
        }


# ── Formatting helpers ────────────────────────────────────────────────────────

_WIDTH = 60

def _header(title: str) -> None:
    print(f"\n{'═' * _WIDTH}")
    print(f"  {title}")
    print(f"{'═' * _WIDTH}\n")

def _divider() -> None:
    print(f"\n{'─' * _WIDTH}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main(notebook_path: str) -> None:
    _header(f"Layer 2 Integration Test")
    print(f"  Notebook : {notebook_path}")
    print(f"  Model    : claude-sonnet-4-6")

    # Step 1: parse the notebook
    print("\n[1/6] Parsing notebook…")
    code = await parse_input(notebook_path)
    print(f"      {len(code)} chars extracted.\n")

    # Shared client — tracks usage across all tool calls
    client = TrackingClient()

    tools = [
        ("leakage_detector",             LeakageDetector(client=client)),
        ("subgroup_reporter",            SubgroupReporter(client=client)),
        ("provenance_checker",           ProvenanceChecker(client=client)),
        ("hyperparameter_reporter",      HyperparameterReporter(client=client)),
        ("metrics_completeness_checker", MetricsCompletenessChecker(client=client)),
    ]

    n_total = 1 + len(tools)  # parse step + tool steps

    for step, (name, tool) in enumerate(tools, start=2):
        print(f"[{step}/{n_total}] Running {name}…")
        result = await tool.execute(code=code)

        _header(f"Result: {name}")
        print(result.strip())
        _divider()

        usage = client.calls[-1]
        truncated = usage["stop_reason"] == "max_tokens"
        print(f"  tokens this call — input: {usage['input']:,}  "
              f"output: {usage['output']:,}  "
              f"cache_read: {usage['cache_read']:,}  "
              f"cache_creation: {usage['cache_creation']:,}")
        print(f"  stop_reason: {usage['stop_reason']}"
              + ("  ⚠ TRUNCATED" if truncated else ""))

    # Step 5: totals
    _header("Token Usage Summary")
    t = client.total
    rows = [
        ("API calls",          str(t["api_calls"])),
        ("Total input tokens", f"{t['input']:,}"),
        ("Total output tokens",f"{t['output']:,}"),
        ("Cache read tokens",  f"{t['cache_read']:,}"),
        ("Cache creation",     f"{t['cache_creation']:,}"),
    ]
    col = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"  {label:<{col}} : {value}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Layer 2 tools with real Claude API.")
    parser.add_argument(
        "--notebook",
        default="fixtures/notebooks/test_notebook.ipynb",
        help="Path to the .ipynb or .py file to audit (default: fixtures/notebooks/test_notebook.ipynb).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.notebook))
