from __future__ import annotations

from satyarepro.client.asta import AstaClient
from satyarepro.client.base import ModelClient

from .base import Tool, ToolRegistry
from .layer1 import CheckpointCheck, DependencyCheck, SeedCheck, SplitCheck
from .layer2 import (
    DocumentationCompletenessChecker,
    LeakageDetector,
    OutcomeDistributionChecker,
    ProvenanceChecker,
    ReportingCompletenessChecker,
    SubgroupReporter,
    WithinPaperRobustnessChecker,
)
from .parsers import NotebookParser, RepoFetcher, ScriptParser
from .reports import DMSPGenerator, TripodAIGenerator


def create_default_registry(
    client: ModelClient | None = None,
    asta_client: AstaClient | None = None,
) -> ToolRegistry:
    """Build a ToolRegistry with all tools.

    Pass a ModelClient to share it across Layer 2 and report tools
    (useful in tests with MockClient). If None, each tool lazily
    creates its own ClaudeClient on first use. Same for asta_client,
    used by outcome_distribution_checker.
    """
    registry = ToolRegistry()
    registry.register(
        # Layer 1 — static analysis, no LLM
        SeedCheck(),
        DependencyCheck(),
        SplitCheck(),
        CheckpointCheck(),
        # Layer 2 — semantic analysis, LLM-powered
        LeakageDetector(client),
        ProvenanceChecker(client),
        # Applicability — cross-study outcome distribution and subgroup reporting checks
        SubgroupReporter(client),
        OutcomeDistributionChecker(client, asta_client),
        ReportingCompletenessChecker(client),
        DocumentationCompletenessChecker(client),
        # Replicability — within-paper statistical robustness signals
        WithinPaperRobustnessChecker(client),
        # Report generators
        TripodAIGenerator(client),
        DMSPGenerator(client),
        # Input parsers
        NotebookParser(),
        ScriptParser(),
        RepoFetcher(),
    )
    return registry


__all__ = ["Tool", "ToolRegistry", "create_default_registry"]
