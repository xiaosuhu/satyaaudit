"""Generate a structured NIH DMSP compliance report for a biomedical ML notebook.

Pipeline:
  1. parse_input()          — extract code from the notebook
  2. provenance_checker     — audit NIH DMSP data provenance
  3. dmsp_compliance_report — format findings into a structured markdown report
  4. Save to outputs/dmsp_report_<stem>.md

Usage:
    python scripts/generate_dmsp_report.py
    python scripts/generate_dmsp_report.py \
        --input  fixtures/notebooks/heart-disease-prediction-notebook.ipynb \
        --output outputs/dmsp_report_heart_disease.md
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from satyarepro.config import settings
if settings.anthropic_api_key:
    os.environ.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)

from satyarepro.client.claude import ClaudeClient
from satyarepro.types import CompletionResponse
from satyarepro.tools.layer2.provenance_checker import ProvenanceChecker
from satyarepro.tools.reports.dmsp_compliance_report import DMSPComplianceReport
from satyarepro.tools.parsers import parse_input

_WIDTH = 60

# ── Usage-tracking wrapper (same pattern as test_layer2_real.py) ──────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'═' * _WIDTH}")
    print(f"  {title}")
    print(f"{'═' * _WIDTH}\n")

def _divider() -> None:
    print(f"\n{'─' * _WIDTH}\n")

def _print_usage(call: dict) -> None:
    truncated = call["stop_reason"] == "max_tokens"
    print(f"  tokens — input: {call['input']:,}  output: {call['output']:,}  "
          f"cache_read: {call['cache_read']:,}  cache_creation: {call['cache_creation']:,}")
    print(f"  stop_reason: {call['stop_reason']}"
          + ("  ⚠ TRUNCATED" if truncated else ""))


# ── Main ─────────────────────────────────────────────────────────────────────

async def main(input_path: str, output_path: str) -> None:
    filename = Path(input_path).name
    report_date = date.today().isoformat()

    _header("DMSP Compliance Report Generator")
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_path}")
    print(f"  Date   : {report_date}")
    print(f"  Model  : claude-sonnet-4-6")

    client = TrackingClient()
    n_total = 3  # parse + provenance_checker + dmsp_compliance_report

    # Step 1: parse
    print(f"\n[1/{n_total}] Parsing {filename}…")
    code = await parse_input(input_path)
    print(f"      {len(code)} chars extracted.")

    # Step 2: provenance_checker
    print(f"\n[2/{n_total}] Running provenance_checker…")
    provenance_findings = await ProvenanceChecker(client=client).execute(code=code)

    _header("Provenance Checker Output")
    print(provenance_findings.strip())
    _divider()
    _print_usage(client.calls[-1])

    # Step 3: dmsp_compliance_report
    print(f"\n[3/{n_total}] Generating DMSP compliance report…")
    report_md = await DMSPComplianceReport(client=client).execute(
        provenance_findings=provenance_findings,
        filename=filename,
        date=report_date,
    )

    _header("DMSP Compliance Report")
    print(report_md.strip())
    _divider()
    _print_usage(client.calls[-1])

    # Save report
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_md, encoding="utf-8")
    print(f"\n  Report saved to: {out.resolve()}")

    # Token summary
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
    parser = argparse.ArgumentParser(description="Generate NIH DMSP compliance report.")
    parser.add_argument(
        "--input",
        default="fixtures/notebooks/heart-disease-prediction-notebook.ipynb",
        help="Path to .ipynb or .py file to audit.",
    )
    parser.add_argument(
        "--output",
        default="outputs/dmsp_report_heart_disease.md",
        help="Output path for the markdown report.",
    )
    args = parser.parse_args()
    asyncio.run(main(args.input, args.output))
