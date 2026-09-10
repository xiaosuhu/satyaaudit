---
title: SatyaAudit
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.41.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# SatyaAudit

An agentic AI tool that audits biomedical AI/ML workflows for reproducibility
issues and generates standards-compliant reports (TRIPOD-AI, NIH DMSP).

Connected to **DAIR3** (NIH grant 5R25GM151182-03) at the University of
Michigan [MIDAS](https://midas.umich.edu/) center.

---

## What it checks

| Check | What it detects |
|---|---|
| **Seed fixation** | Missing `numpy.random.seed`, `torch.manual_seed`, etc. |
| **Train/test leakage** | Preprocessing fitted on the full dataset |
| **Model checkpoint** | Training loops that never save the trained model |
| **Dependency pins** | Packages without version pins in `requirements.txt` |

Layer 1 (static analysis) runs entirely offline with no API key.
Layer 2 adds LLM-powered semantic checks (patient-level leakage, subgroup
reporting, data provenance) and requires an Anthropic API key.

---

## HuggingFace Space demo

> **Try it in your browser — no install needed:**  
> 🔗 **Live Demo:** [https://huggingface.co/spaces/xiaosuhu1986/satyarepro](https://huggingface.co/spaces/xiaosuhu1986/satyarepro)


---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/your-org/satyarepro.git
cd satyarepro
pip install -e .
```

To run Layer 2 or the full agentic audit, set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## CLI usage

### Static audit — no API key required

Runs seed, leakage, and checkpoint checks using Python AST only.

```bash
satyarepro notebook.ipynb --static-only
satyarepro experiment.py  --static-only
```

Dependency check is included automatically when a `requirements.txt` is found
in the same directory as the input file.

### Full agentic audit — requires `ANTHROPIC_API_KEY`

The LLM agent calls all tools, generates a TRIPOD-AI checklist, and drafts a
NIH Data Management & Sharing Plan.

```bash
satyarepro notebook.ipynb
satyarepro train.ipynb --model claude-opus-4-7 --max-iter 15
```

Also runnable as a module:

```bash
python -m satyarepro.cli notebook.ipynb --static-only
```

### All CLI options

```
usage: satyarepro [-h] [--static-only] [--model MODEL] [--max-iter N] path

positional arguments:
  path             Path to a .ipynb or .py file.

options:
  --static-only    Layer 1 static analysis only — no API key required.
  --model MODEL    Claude model ID (default: claude-sonnet-4-6).
  --max-iter N     Maximum agent iterations for full audit (default: 10).
```

---

## Architecture

```
Central LLM Agent
│
├── Layer 1 — Static Analysis (Python AST, no LLM)
│   ├── seed_check          detect missing random seed fixation
│   ├── split_check         detect train/test data leakage patterns
│   ├── checkpoint_check    detect missing model checkpoint saving
│   └── dependency_check    detect unpinned package versions
│
├── Layer 2 — Semantic Analysis (LLM-powered)
│   ├── leakage_detector    cross-function patient-level data leakage
│   ├── subgroup_reporter   age/sex/race subgroup performance reporting
│   └── provenance_checker  data source vs NIH DMSP requirements
│
└── Report Generators
    ├── tripod_ai_generator  TRIPOD-AI checklist
    └── dmsp_generator       NIH Data Management & Sharing Plan draft
```

Input parsers support `.ipynb` notebooks and `.py` scripts (Year 1). R and
MATLAB support are planned for Year 2.

---

## Running the FastAPI server

```bash
uvicorn main:app --reload
```

Endpoints:
- `POST /audit` — submit a file path for auditing (returns job ID)
- `GET  /audit/{id}` — poll job status and retrieve results
- `GET  /health` — health check

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q      # 68 tests, all passing
```

---

## Citation / Acknowledgment

This work is supported by the National Institutes of Health under award
**5R25GM151182-03** (DAIR3), University of Michigan MIDAS.
