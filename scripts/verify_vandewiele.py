"""Targeted audit of GillesVandewiele/EHG-oversampling oversampling_analysis.py.

Runs two Layer 2 tools against a real-world EHR oversampling script:
  - leakage_detector          (expected: StandardScaler fit before split)
  - metrics_completeness_checker

Source:
  https://raw.githubusercontent.com/GillesVandewiele/EHG-oversampling/
  master/experiments/oversampling_analysis.py

Local copy: fixtures/notebooks/oversampling_analysis.py

Usage:
    python scripts/verify_vandewiele.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from satyarepro.config import settings
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

from satyarepro.client.claude import ClaudeClient
from satyarepro.types import CompletionResponse
from satyarepro.tools.layer2.leakage_detector import LeakageDetector
from satyarepro.tools.layer2.metrics_completeness_checker import MetricsCompletenessChecker
from satyarepro.tools.parsers import parse_input

_TARGET = "fixtures/notebooks/oversampling_analysis.py"

# ── Usage-tracking wrapper ────────────────────────────────────────────────────

class TrackingClient(ClaudeClient):
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

async def main() -> None:
    _header("Vandewiele EHG Oversampling — Layer 2 Audit")
    print(f"  File  : {_TARGET}")
    print(f"  Model : claude-sonnet-4-6")
    print(f"  Tools : leakage_detector, metrics_completeness_checker")

    print("\n[1/3] Parsing script…")
    code = await parse_input(_TARGET)
    print(f"      {len(code)} chars extracted.\n")

    client = TrackingClient()
    n_total = 3  # parse + 2 tools

    tools = [
        ("leakage_detector",             LeakageDetector(client=client)),
        ("metrics_completeness_checker", MetricsCompletenessChecker(client=client)),
    ]

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

    _header("Token Usage Summary")
    t = client.total
    rows = [
        ("API calls",           str(t["api_calls"])),
        ("Total input tokens",  f"{t['input']:,}"),
        ("Total output tokens", f"{t['output']:,}"),
        ("Cache read tokens",   f"{t['cache_read']:,}"),
        ("Cache creation",      f"{t['cache_creation']:,}"),
    ]
    col = max(len(r[0]) for r in rows)
    for label, value in rows:
        print(f"  {label:<{col}} : {value}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
