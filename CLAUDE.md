# SatyaRepro — Developer Guide for Claude Code

## What is SatyaRepro
An agentic AI tool that audits biomedical AI/ML workflows for 
reproducibility issues and generates standards-compliant reports.
Connected to DAIR3 (NIH grant 5R25GM151182-03) at University of 
Michigan MIDAS.

## Input / Output
Input: Jupyter notebook (.ipynb), Python script (.py), 
       Git repository (URL)
Output 1: Reproducibility Audit Report (specific issues + fixes)
Output 2: Standards-Compliant Report (TRIPOD-AI, NIH DMSP)

## Architecture
Central LLM Agent orchestrates all analysis via tool calls.
The agent decides which tools to call, in what order, and 
how many times — not a fixed pipeline.

Three categories of tools:
- Layer 1: Static analysis tools (Python AST) — deterministic,
  no LLM inference needed, called by agent
- Layer 2: Semantic analysis tools — require LLM reasoning,
  agent uses these for complex contextual checks
- Data: Local data analysis tools — deterministic, pandas-based,
  no LLM or API calls; run against CSV datasets directly

ModelClient abstraction layer: agent's "brain" is swappable.
Backends: ClaudeClient, MockClient (testing), 
OllamaClient (local, to be added in Year 2).

## Tool Library
These are the tools available to the LLM agent via function 
calling. The agent decides which tools to invoke and in what 
order based on the input code.

### Layer 1 — Static Analysis (no LLM)
- seed_check: detect missing random seed fixation 
  (numpy, torch, random, tensorflow)
- dependency_check: detect missing version pins 
  (requirements.txt or environment.yml passed as content string)
- split_check: detect potential train/test data leakage patterns
  (handles chained calls e.g. StandardScaler().fit_transform(X))
- checkpoint_check: detect missing model checkpoint saving

### Layer 2 — Semantic Analysis (LLM-powered)
- leakage_detector: cross-function patient-level data leakage
- subgroup_reporter: check age/sex/race subgroup performance reporting
- provenance_checker: data source description vs NIH DMSP requirements
- hyperparameter_reporter: document model hyperparameters and tuning
- metrics_completeness_checker: verify reporting of all relevant metrics

### Data — Local Analysis (no LLM, no API)
- data_profiler: profile a CSV dataset — basic stats, missing values,
  class imbalance, demographic distributions (age/sex), IQR outliers,
  and consistency checks for known clinical columns (ca: 0–3, thal: 1–3)

### Report Generators
- tripod_ai_generator: generate TRIPOD-AI checklist from audit results
- dmsp_generator: generate NIH Data Management & Sharing Plan draft
- dmsp_compliance_report: structured NIH DMSP compliance report with
  per-section Status (COMPLIANT/PARTIAL/NON-COMPLIANT), evidence
  citations from provenance_checker output, and Summary Scorecard

### Input Parsers (Year 1: code only)
- notebook_parser: parse .ipynb files — strips Jupyter magic commands
  (%matplotlib, !pip, etc.) and skips cells with syntax errors; 
  remaining cells are always valid Python for downstream AST tools
- script_parser: parse .py files — strips magic lines (%/! prefixes)
  and handles syntax errors with placeholder comments; prepends
  # ── script ── header for consistency with notebook output
- unified_parser: parse_input(path) entry point — routes .ipynb to
  NotebookParser, .py to ScriptParser, raises ValueError otherwise
- repo_fetcher: clone GitHub repo (shallow), extract relevant files
# dataset_parser: to be added in Year 2

## File Structure
satyarepro/
├── types.py          — shared dataclasses (ToolCall, ToolSchema, 
│                        CompletionResponse, Usage)
├── config.py         — pydantic-settings (.env, ANTHROPIC_API_KEY,
│                        CLAUDE_MODEL, NCBI_EMAIL, MAX_AUDIT_ITERATIONS)
├── cli.py            — CLI entry point (satyarepro command)
├── client/           — ModelClient abstraction
│   ├── base.py       — abstract ModelClient
│   ├── claude.py     — ClaudeClient (prompt caching on system prompt)
│   └── mock.py       — MockClient (FIFO queue, for tests)
├── agent/
│   └── orchestrator.py — AuditOrchestrator (tool-use loop)
├── api/              — FastAPI app
│   ├── app.py        — create_app() factory
│   ├── schemas.py    — Pydantic request/response models
│   └── routers/
│       ├── audit.py  — POST /audit (202 + background task),
│       │               GET /audit/{id} (poll status)
│       └── health.py — GET /health
└── tools/
    ├── base.py       — Tool ABC, ToolRegistry
    ├── _utils.py     — shared AST helpers (dotted_name,
    │                    collect_imports, collect_calls)
    ├── layer1/       — seed_check, dependency_check,
    │                    split_check, checkpoint_check
    ├── layer2/       — leakage_detector, subgroup_reporter,
    │                    provenance_checker, hyperparameter_reporter,
    │                    metrics_completeness_checker
    ├── data/         — data_profiler (local CSV analysis, no LLM)
    ├── reports/      — tripod_ai_generator, dmsp_generator,
    │                    dmsp_compliance_report
    └── parsers/      — notebook_parser, script_parser, repo_fetcher,
                       unified_parser (parse_input routing fn)

## CLI Usage
# Install (editable):
pip install -e .

# Static audit — no API key required:
satyarepro notebook.ipynb --static-only

# Full agentic audit — requires ANTHROPIC_API_KEY:
satyarepro notebook.ipynb
satyarepro script.py --model claude-opus-4-7 --max-iter 15

# Also runnable as a module:
python -m satyarepro.cli notebook.ipynb --static-only

## Phasing
Year 1 (current): Python only, local deployment, 
                  TRIPOD-AI + NIH DMSP reports
Year 2: R, MATLAB support; cloud deployment; dataset_parser;
        OllamaClient
Year 3: DOME, CONSORT-AI standards

## Implementation Status (Year 1)
Built and tested (88 tests passing):
- 13 tools total: Layer 1 (4) + Layer 2 (5) + Data (1) + Reports (3)
  registered in create_default_registry() (Data tools registered
  separately — not yet wired into the agent's default registry)
- Layer 2 tools and report generators accept optional ModelClient;
  fall back to lazy ClaudeClient() if none passed
- ClaudeClient caches the system prompt (cache_control: ephemeral)
  to reduce token cost across agent iterations
- CLI supports --static-only (no API key) and full agentic mode
- FastAPI server (main.py / uvicorn) for HTTP access
- .py input support end-to-end:
  - script_parser upgraded with magic strip + syntax error handling
  - unified_parser.parse_input() routes by file extension
  - cli.py, app.py, repo_fetcher all use parse_input() — no manual
    if/else routing in call sites
- DataProfiler runs fully locally (pandas only, no API key required);
  piloted on UCI Heart Disease Cleveland dataset (303 rows × 14 cols)

## Test Inputs and Integration Scripts
Three categories of test inputs, each with a corresponding script:

### Jupyter notebooks (fixtures/notebooks/)
- test_notebook.ipynb — generic ML notebook for Layer 2 integration tests
- heart-disease-prediction-notebook.ipynb — Kaggle heart disease notebook;
  used for DMSP compliance report pipeline
  Script: scripts/verify_layer2_real.py (all 5 Layer 2 tools)
          scripts/generate_dmsp_report.py (provenance → DMSP report)

### Python scripts (fixtures/notebooks/)
- oversampling_analysis.py — Vandewiele EHG oversampling (real-world EHR);
  known leakage: StandardScaler fit before CV split
  Script: scripts/verify_vandewiele.py (leakage_detector + metrics_checker)

### CSV datasets (fixtures/data/)
- heart_disease.csv — UCI Cleveland dataset prepared by
  fixtures/data/prepare_heart_disease_data.py (303 rows, 6 missing values)
  Script: scripts/verify_data_profiler.py (DataProfiler, no API key needed)

Known gaps for Year 1 completion:
- dependency_check requires requirements.txt content passed separately;
  CLI --static-only auto-discovers requirements.txt in the same dir
- Benchmark dataset (annotated notebooks with known issues) not yet built
- OllamaClient not yet implemented (Year 2)

Backlog:
- 支持本地 Ollama 作为可选 ModelClient backend：
  新建 satyarepro/client/ollama.py，实现和 ClaudeClient 相同的接口
  （complete / complete_with_tools），通过环境变量 MODEL_BACKEND=ollama /
  OLLAMA_MODEL=llama3 等切换。
  用途：本地测试 Layer 2 prompt 质量、对比小模型 vs Claude 的检测准确度、
  离线开发。

## Testing Strategy
- Layer 1 tools: unit tested deterministically — no LLM or MockClient needed
- Layer 2 tools + report generators: tested with MockClient for structure
  and prompt routing; ClaudeClient for quality (manual / integration)
- Data tools: unit tested with tmp_path CSV fixtures — no LLM or network
- Parsers: tested with tmp_path fixtures (no network)
- Each tool is independently testable via ToolRegistry.dispatch()
- Run tests: python -m pytest tests/ -q

## Deployment
- Local: Ollama + open-weight model (Gemma 4) for 
  privacy-sensitive data (Year 2)
- Cloud: Claude API / Gemini API for broader accessibility
- Config controls which backend is active

## Do Not Change
- client/ directory (ModelClient abstraction is correct)
- agent/orchestrator.py (agent loop structure is correct)
- api/ directory (FastAPI structure is correct)
- tests/ structure (keep existing test patterns)
- tools/base.py interface (Tool ABC and ToolRegistry contract)
