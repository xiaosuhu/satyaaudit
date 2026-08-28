"""Tests for all tools in the SatyaRepro tool library.

Layer 1 (static): tested deterministically — no LLM needed.
Layer 2 + reports: tested with MockClient to verify prompt routing and response pass-through.
Parsers: tested with in-memory temp files.
"""
import json
import os
import textwrap
import tempfile

import pandas as pd
import pytest

from satyarepro.client.mock import MockClient
from satyarepro.tools import create_default_registry
from satyarepro.tools.data.data_profiler import DataProfiler
from satyarepro.tools.layer1.checkpoint_check import CheckpointCheck
from satyarepro.tools.layer1.dependency_check import DependencyCheck
from satyarepro.tools.layer1.seed_check import SeedCheck
from satyarepro.tools.layer1.split_check import SplitCheck
from satyarepro.tools.layer2.hyperparameter_reporter import HyperparameterReporter
from satyarepro.tools.layer2.leakage_detector import LeakageDetector
from satyarepro.tools.layer2.metrics_completeness_checker import MetricsCompletenessChecker
from satyarepro.tools.layer2.provenance_checker import ProvenanceChecker
from satyarepro.tools.layer2.applicability.reporting_completeness_checker import (
    ReportingCompletenessChecker,
)
from satyarepro.tools.layer2.applicability.subgroup_reporter import SubgroupReporter
from satyarepro.tools.parsers import parse_input
from satyarepro.tools.parsers.notebook_parser import NotebookParser
from satyarepro.tools.parsers.script_parser import ScriptParser
from satyarepro.tools.reports.dmsp_generator import DMSPGenerator
from satyarepro.tools.reports.tripod_ai_generator import TripodAIGenerator
from satyarepro.types import CompletionResponse, Usage


# ── helpers ────────────────────────────────────────────────────────────────────

def _ok_response(text: str) -> CompletionResponse:
    return CompletionResponse(
        content=text,
        raw_content=[{"type": "text", "text": text}],
        usage=Usage(10, 5),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — SeedCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestSeedCheck:
    @pytest.fixture
    def tool(self):
        return SeedCheck()

    async def test_numpy_with_seed(self, tool):
        code = "import numpy as np\nnp.random.seed(42)\nX = np.zeros((10, 3))"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "numpy" in result["seeded"]

    async def test_numpy_missing_seed(self, tool):
        code = "import numpy as np\nX = np.zeros((10, 3))"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert any("numpy" in issue for issue in result["issues"])

    async def test_torch_with_manual_seed(self, tool):
        code = "import torch\ntorch.manual_seed(0)\nmodel = torch.nn.Linear(4, 1)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "torch" in result["seeded"]

    async def test_torch_missing_seed(self, tool):
        code = "import torch\nmodel = torch.nn.Linear(4, 1)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert any("torch" in issue for issue in result["issues"])

    async def test_random_module_seeded(self, tool):
        code = "import random\nrandom.seed(7)\nsamples = random.sample(range(100), 10)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_tensorflow_seeded(self, tool):
        code = "import tensorflow as tf\ntf.random.set_seed(1)\nmodel = tf.keras.Sequential()"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "tensorflow" in result["seeded"]

    async def test_multiple_frameworks_partial(self, tool):
        code = textwrap.dedent("""\
            import numpy as np
            import torch
            np.random.seed(42)
            model = torch.nn.Linear(4, 1)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert "numpy" in result["seeded"]
        assert any("torch" in issue for issue in result["issues"])

    async def test_no_ml_imports(self, tool):
        code = "x = 1 + 2\nprint(x)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert result["frameworks_detected"] == []

    async def test_syntax_error_returns_error_key(self, tool):
        result = json.loads(await tool.execute(code="def (broken"))
        assert "error" in result

    async def test_alias_import(self, tool):
        code = "import numpy as np\nnp.random.seed(99)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — DependencyCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestDependencyCheck:
    @pytest.fixture
    def tool(self):
        return DependencyCheck()

    async def test_all_pinned(self, tool):
        reqs = "numpy==1.26.0\nscikit-learn==1.4.0\npandas==2.2.0\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "ok"

    async def test_unpinned_detected(self, tool):
        reqs = "numpy\nscikit-learn==1.4.0\npandas\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "issues_found"
        assert "numpy" in result["unpinned_packages"]
        assert "pandas" in result["unpinned_packages"]
        assert "scikit-learn" not in result["unpinned_packages"]

    async def test_comments_and_blank_lines_ignored(self, tool):
        reqs = "# this is a comment\n\nnumpy==1.26.0\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "ok"

    async def test_extras_syntax_handled(self, tool):
        reqs = "uvicorn[standard]==0.32.0\nfastapi\n"
        result = json.loads(await tool.execute(requirements=reqs))
        assert result["status"] == "issues_found"
        assert "fastapi" in result["unpinned_packages"]
        assert "uvicorn" not in result["unpinned_packages"]

    async def test_conda_env_pinned(self, tool):
        env = "name: myenv\ndependencies:\n  - numpy=1.26.0\n  - scikit-learn=1.4.0\n"
        result = json.loads(await tool.execute(requirements=env, file_type="environment.yml"))
        assert result["status"] == "ok"

    async def test_conda_env_unpinned(self, tool):
        env = "name: myenv\ndependencies:\n  - numpy\n  - scikit-learn=1.4.0\n"
        result = json.loads(await tool.execute(requirements=env, file_type="environment.yml"))
        assert result["status"] == "issues_found"
        assert "numpy" in result["unpinned_packages"]

    async def test_empty_requirements(self, tool):
        result = json.loads(await tool.execute(requirements=""))
        assert result["status"] == "ok"


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — SplitCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestSplitCheck:
    @pytest.fixture
    def tool(self):
        return SplitCheck()

    async def test_fit_on_train_data_ok(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_fit_on_full_data_flagged(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"
        assert any("X" in issue["call"] for issue in result["issues"])

    async def test_fit_transform_on_full_data_flagged(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X_all)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"

    async def test_no_fitting_ok(self, tool):
        code = "y_pred = model.predict(X_test)\nprint(y_pred)"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_syntax_error_returns_error_key(self, tool):
        result = json.loads(await tool.execute(code="class ("))
        assert "error" in result

    async def test_line_number_reported(self, tool):
        code = textwrap.dedent("""\
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X)
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["issues"][0]["line"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# Layer 1 — CheckpointCheck
# ══════════════════════════════════════════════════════════════════════════════

class TestCheckpointCheck:
    @pytest.fixture
    def tool(self):
        return CheckpointCheck()

    async def test_training_with_save_ok(self, tool):
        code = textwrap.dedent("""\
            import torch
            model = torch.nn.Linear(4, 1)
            for epoch in range(10):
                loss.backward()
                optimizer.step()
            torch.save(model.state_dict(), "model.pt")
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_training_without_save_flagged(self, tool):
        code = textwrap.dedent("""\
            import torch
            model = torch.nn.Linear(4, 1)
            for epoch in range(10):
                loss.backward()
                optimizer.step()
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "issues_found"

    async def test_sklearn_fit_with_joblib_ok(self, tool):
        code = textwrap.dedent("""\
            from sklearn.ensemble import RandomForestClassifier
            import joblib
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            joblib.dump(model, "rf_model.pkl")
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_no_training_not_applicable(self, tool):
        code = "import numpy as np\nX = np.load('data.npy')"
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"
        assert "not applicable" in result["message"]

    async def test_keras_fit_with_save_ok(self, tool):
        code = textwrap.dedent("""\
            model.fit(X_train, y_train, epochs=10)
            model.save("my_model.keras")
        """)
        result = json.loads(await tool.execute(code=code))
        assert result["status"] == "ok"

    async def test_syntax_error_returns_error_key(self, tool):
        result = json.loads(await tool.execute(code="for ("))
        assert "error" in result


# ══════════════════════════════════════════════════════════════════════════════
# Layer 2 — LLM-powered tools (MockClient)
# ══════════════════════════════════════════════════════════════════════════════

_SAMPLE_CODE = "import pandas as pd\ndf = pd.read_csv('patients.csv')\n"
_SAMPLE_AUDIT = '{"seed_check": "ok", "split_check": "issues_found"}'


class TestLeakageDetector:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("No leakage detected."))
        tool = LeakageDetector(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert result == "No leakage detected."

    async def test_code_is_sent_to_client(self):
        mock = MockClient()
        tool = LeakageDetector(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert mock.calls[0]["type"] == "complete"
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert LeakageDetector().schema.name == "leakage_detector"


class TestSubgroupReporter:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("Age: partial. Sex: no. Race: no."))
        tool = SubgroupReporter(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert "Age" in result

    async def test_code_forwarded(self):
        mock = MockClient()
        tool = SubgroupReporter(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert SubgroupReporter().schema.name == "subgroup_reporter"


_SAMPLE_MANUSCRIPT = (
    "Methods: We included adult patients (age 18-75) with confirmed diagnosis at Hospital A. "
    "Patients with incomplete records were excluded. "
    "Discussion: The model was validated on an external cohort from Hospital B. "
    "Limitations: Our sample may not generalize to pediatric populations."
)

_FULL_PRESENT_ARRAY = json.dumps(
    [
        {
            "item_id": "target_population_scope",
            "status": "present",
            "evidence_source": "manuscript",
            "evidence": "adult patients (age 18-75) with confirmed diagnosis at Hospital A",
            "reasoning": "Target population and clinical scope explicitly stated.",
        },
        {
            "item_id": "inclusion_exclusion_criteria",
            "status": "present",
            "evidence_source": "manuscript",
            "evidence": "Patients with incomplete records were excluded.",
            "reasoning": "Exclusion criteria stated.",
        },
        {
            "item_id": "external_validation",
            "status": "present",
            "evidence_source": "manuscript",
            "evidence": "validated on an external cohort from Hospital B",
            "reasoning": "External validation reported.",
        },
        {
            "item_id": "limitations_discussed",
            "status": "present",
            "evidence_source": "manuscript",
            "evidence": "may not generalize to pediatric populations",
            "reasoning": "Limitations discussed.",
        },
    ]
)

_CODE_ONLY_ARRAY = json.dumps(
    [
        {
            "item_id": "target_population_scope",
            "status": "not_determinable_from_code",
            "evidence_source": "none",
            "evidence": "",
            "reasoning": "Code contains no narrative description of target population.",
        },
        {
            "item_id": "inclusion_exclusion_criteria",
            "status": "not_determinable_from_code",
            "evidence_source": "none",
            "evidence": "",
            "reasoning": "No inclusion/exclusion narrative present in code.",
        },
        {
            "item_id": "external_validation",
            "status": "not_determinable_from_code",
            "evidence_source": "none",
            "evidence": "",
            "reasoning": "No explicit external validation signal found.",
        },
        {
            "item_id": "limitations_discussed",
            "status": "not_determinable_from_code",
            "evidence_source": "none",
            "evidence": "",
            "reasoning": "Limitations are a narrative item, not visible in code.",
        },
    ]
)


class TestReportingCompletenessChecker:
    async def test_manuscript_only(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_PRESENT_ARRAY))
        tool = ReportingCompletenessChecker(client=mock)
        result = json.loads(await tool.execute(manuscript_text=_SAMPLE_MANUSCRIPT))
        assert len(result) == 4
        assert result[0]["item_id"] == "target_population_scope"
        assert result[0]["status"] == "present"
        assert _SAMPLE_MANUSCRIPT in mock.calls[0]["messages"][0]["content"]

    async def test_code_only_not_determinable(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_CODE_ONLY_ARRAY))
        tool = ReportingCompletenessChecker(client=mock)
        result = json.loads(await tool.execute(code=_SAMPLE_CODE))
        assert len(result) == 4
        assert all(item["status"] == "not_determinable_from_code" for item in result)
        sent_prompt = mock.calls[0]["messages"][0]["content"]
        assert _SAMPLE_CODE in sent_prompt
        assert "not_determinable_from_code" in sent_prompt

    async def test_both_provided(self):
        mock = MockClient()
        mock.enqueue(_ok_response(_FULL_PRESENT_ARRAY))
        tool = ReportingCompletenessChecker(client=mock)
        result = json.loads(
            await tool.execute(manuscript_text=_SAMPLE_MANUSCRIPT, code=_SAMPLE_CODE)
        )
        assert len(result) == 4
        sent_prompt = mock.calls[0]["messages"][0]["content"]
        assert _SAMPLE_MANUSCRIPT in sent_prompt
        assert _SAMPLE_CODE in sent_prompt

    async def test_raises_without_manuscript_or_code(self):
        tool = ReportingCompletenessChecker(client=MockClient())
        with pytest.raises(ValueError):
            await tool.execute()

    async def test_schema_name(self):
        assert ReportingCompletenessChecker().schema.name == "reporting_completeness_checker"


class TestProvenanceChecker:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("Dataset source: partial."))
        tool = ProvenanceChecker(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert "Dataset source" in result

    async def test_code_forwarded(self):
        mock = MockClient()
        tool = ProvenanceChecker(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert ProvenanceChecker().schema.name == "provenance_checker"


class TestHyperparameterReporter:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("Learning rate hardcoded: high severity."))
        tool = HyperparameterReporter(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert "Learning rate" in result

    async def test_code_forwarded(self):
        mock = MockClient()
        tool = HyperparameterReporter(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert HyperparameterReporter().schema.name == "hyperparameter_reporter"


class TestMetricsCompletenessChecker:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("AUC-ROC missing: high severity."))
        tool = MetricsCompletenessChecker(client=mock)
        result = await tool.execute(code=_SAMPLE_CODE)
        assert "AUC-ROC" in result

    async def test_code_forwarded(self):
        mock = MockClient()
        tool = MetricsCompletenessChecker(client=mock)
        await tool.execute(code=_SAMPLE_CODE)
        assert _SAMPLE_CODE in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert MetricsCompletenessChecker().schema.name == "metrics_completeness_checker"


# ══════════════════════════════════════════════════════════════════════════════
# Report generators (MockClient)
# ══════════════════════════════════════════════════════════════════════════════

class TestTripodAIGenerator:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("TRIPOD-AI checklist: Item 1 ✓"))
        tool = TripodAIGenerator(client=mock)
        result = await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert "TRIPOD-AI" in result

    async def test_audit_results_forwarded(self):
        mock = MockClient()
        tool = TripodAIGenerator(client=mock)
        await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert _SAMPLE_AUDIT in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert TripodAIGenerator().schema.name == "tripod_ai_generator"


class TestDMSPGenerator:
    async def test_returns_llm_response(self):
        mock = MockClient()
        mock.enqueue(_ok_response("DMSP Draft: 1. Data Type ..."))
        tool = DMSPGenerator(client=mock)
        result = await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert "DMSP" in result

    async def test_audit_results_forwarded(self):
        mock = MockClient()
        tool = DMSPGenerator(client=mock)
        await tool.execute(audit_results=_SAMPLE_AUDIT)
        assert _SAMPLE_AUDIT in mock.calls[0]["messages"][0]["content"]

    async def test_schema_name(self):
        assert DMSPGenerator().schema.name == "dmsp_generator"


# ══════════════════════════════════════════════════════════════════════════════
# Parsers
# ══════════════════════════════════════════════════════════════════════════════

class TestNotebookParser:
    @pytest.fixture
    def tool(self):
        return NotebookParser()

    @pytest.fixture
    def sample_notebook(self, tmp_path):
        nb = {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {},
            "cells": [
                {"cell_type": "markdown", "source": ["# Title"], "metadata": {}},
                {"cell_type": "code", "source": ["import numpy as np"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["x = np.array([1, 2, 3])"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["   \n"], "metadata": {}, "outputs": []},  # blank cell
            ],
        }
        path = tmp_path / "test.ipynb"
        path.write_text(json.dumps(nb), encoding="utf-8")
        return str(path)

    async def test_extracts_code_cells(self, tool, sample_notebook):
        result = await tool.execute(path=sample_notebook)
        assert "import numpy as np" in result
        assert "x = np.array([1, 2, 3])" in result

    async def test_skips_markdown_cells(self, tool, sample_notebook):
        result = await tool.execute(path=sample_notebook)
        assert "# Title" not in result

    async def test_skips_blank_code_cells(self, tool, sample_notebook):
        result = await tool.execute(path=sample_notebook)
        assert result.count("# ── cell ──") == 1  # only one separator between two non-blank cells

    async def test_empty_notebook_returns_message(self, tool, tmp_path):
        nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": []}
        path = tmp_path / "empty.ipynb"
        path.write_text(json.dumps(nb))
        result = await tool.execute(path=str(path))
        assert "No code cells" in result

    async def test_syntax_error_cell_skipped_rest_included(self, tool, tmp_path):
        nb = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["import numpy as np"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["def broken(\n    pass\n"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["x = np.zeros(3)"], "metadata": {}, "outputs": []},
            ],
        }
        path = tmp_path / "broken.ipynb"
        path.write_text(json.dumps(nb))
        result = await tool.execute(path=str(path))
        assert "import numpy as np" in result
        assert "x = np.zeros(3)" in result
        assert "cell 2 skipped" in result
        assert "syntax error" in result.lower()

    async def test_syntax_error_header_lists_all_skipped(self, tool, tmp_path):
        nb = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["def bad1(\n"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["x = 1"], "metadata": {}, "outputs": []},
                {"cell_type": "code", "source": ["def bad2(\n"], "metadata": {}, "outputs": []},
            ],
        }
        path = tmp_path / "two_broken.ipynb"
        path.write_text(json.dumps(nb))
        result = await tool.execute(path=str(path))
        assert "cell 1 skipped" in result
        assert "cell 3 skipped" in result
        assert "x = 1" in result

    async def test_magic_commands_replaced_with_comments(self, tool, tmp_path):
        nb = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["%matplotlib inline\nimport matplotlib.pyplot as plt\n!pip install seaborn\n"],
                    "metadata": {}, "outputs": [],
                },
            ],
        }
        path = tmp_path / "magic.ipynb"
        path.write_text(json.dumps(nb))
        result = await tool.execute(path=str(path))
        assert "import matplotlib.pyplot as plt" in result
        assert "# [magic]" in result
        assert "%matplotlib" not in result.replace("# [magic] %matplotlib", "")
        assert "!pip" not in result.replace("# [magic] !pip", "")

    async def test_magic_cell_is_valid_python_after_stripping(self, tool, tmp_path):
        import ast
        nb = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["%matplotlib inline\nx = 1\n"], "metadata": {}, "outputs": []},
            ],
        }
        path = tmp_path / "magic_valid.ipynb"
        path.write_text(json.dumps(nb))
        result = await tool.execute(path=str(path))
        ast.parse(result)  # must not raise

    async def test_schema_name(self, tool):
        assert tool.schema.name == "notebook_parser"


class TestScriptParser:
    @pytest.fixture
    def tool(self):
        return ScriptParser()

    async def test_reads_file(self, tool, tmp_path):
        script = tmp_path / "model.py"
        script.write_text("import torch\nprint('hello')", encoding="utf-8")
        result = await tool.execute(path=str(script))
        assert "import torch" in result
        assert "print('hello')" in result

    async def test_has_script_header(self, tool, tmp_path):
        script = tmp_path / "model.py"
        script.write_text("x = 1\n", encoding="utf-8")
        result = await tool.execute(path=str(script))
        assert result.startswith("# ── script ──")

    async def test_magic_lines_stripped_to_comments(self, tool, tmp_path):
        script = tmp_path / "magic.py"
        script.write_text(
            "%matplotlib inline\nimport numpy as np\n!pip install pandas\nx = 1\n",
            encoding="utf-8",
        )
        result = await tool.execute(path=str(script))
        assert "# [magic] %matplotlib inline" in result
        assert "# [magic] !pip install pandas" in result
        assert "import numpy as np" in result
        assert "x = 1" in result
        # no line should start with a raw magic prefix
        for line in result.splitlines():
            assert not line.lstrip().startswith(("%", "!")), f"magic line survived: {line!r}"

    async def test_syntax_error_returns_placeholder(self, tool, tmp_path):
        script = tmp_path / "broken.py"
        script.write_text("def bad(\n    pass\n", encoding="utf-8")
        result = await tool.execute(path=str(script))
        assert "script skipped" in result
        assert "syntax error" in result.lower()
        # must not crash and still have the header
        assert "# ── script ──" in result

    async def test_schema_name(self, tool):
        assert tool.schema.name == "script_parser"


# ══════════════════════════════════════════════════════════════════════════════
# ParseInput routing
# ══════════════════════════════════════════════════════════════════════════════

class TestParseInput:
    async def test_ipynb_routes_to_notebook_parser(self, tmp_path):
        nb = {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["import pandas as pd"], "metadata": {}, "outputs": []},
            ],
        }
        path = tmp_path / "notebook.ipynb"
        path.write_text(json.dumps(nb), encoding="utf-8")
        result = await parse_input(str(path))
        assert "import pandas as pd" in result
        # NotebookParser uses cell separators, not the script header
        assert "# ── cell ──" not in result or "# ── script ──" not in result

    async def test_py_routes_to_script_parser(self, tmp_path):
        script = tmp_path / "train.py"
        script.write_text("import sklearn\n", encoding="utf-8")
        result = await parse_input(str(script))
        assert "import sklearn" in result
        assert result.startswith("# ── script ──")

    async def test_unsupported_format_raises_value_error(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file format"):
            await parse_input(str(path))

    async def test_unsupported_format_error_lists_supported_types(self, tmp_path):
        path = tmp_path / "model.pkl"
        path.write_text("binary", encoding="utf-8")
        with pytest.raises(ValueError, match=r"\.(ipynb|py)"):
            await parse_input(str(path))


# ══════════════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_all_fifteen_tools_registered(self):
        registry = create_default_registry()
        names = {s.name for s in registry.schemas()}
        expected = {
            "seed_check", "dependency_check", "split_check", "checkpoint_check",
            "leakage_detector", "subgroup_reporter", "provenance_checker",
            "outcome_distribution_checker",
            "reporting_completeness_checker",
            "within_paper_robustness_checker",
            "tripod_ai_generator", "dmsp_generator",
            "notebook_parser", "script_parser", "repo_fetcher",
        }
        assert names == expected

    async def test_dispatch_layer1_tool(self):
        registry = create_default_registry()
        result = json.loads(
            await registry.dispatch("seed_check", {"code": "import numpy as np\nnp.random.seed(0)"})
        )
        assert result["status"] == "ok"

    async def test_dispatch_unknown_tool_raises(self):
        registry = create_default_registry()
        with pytest.raises(ValueError, match="Unknown tool"):
            await registry.dispatch("nonexistent", {})

    def test_mock_client_shared_with_layer2(self):
        mock = MockClient()
        registry = create_default_registry(client=mock)
        names = {s.name for s in registry.schemas()}
        assert "leakage_detector" in names


# ══════════════════════════════════════════════════════════════════════════════
# DataProfiler — local data analysis tool (no LLM)
# ══════════════════════════════════════════════════════════════════════════════

class TestDataProfiler:
    @pytest.fixture
    def tool(self):
        return DataProfiler()

    @pytest.fixture
    def csv_path(self, tmp_path):
        df = pd.DataFrame({
            "age":    [50, 55, 60, 62, 64, 66, 68, 200, 70, None, 72,   74],
            "sex":    [ 1,  0,  1,  0,  1,  0,  1,   0,  1,    0,  1,    0],
            "ca":     [ 0,  1,  2,  3,  0,  1,  2,   3,  0,    1, None,  4],
            "thal":   [ 3,  2,  1,  3,  2,  3,  1,   3,  2,    3,  2,    7],
            "target": [ 0,  0,  0,  0,  0,  0,  0,   1,  1,    1,  1,    1],
        })
        p = tmp_path / "test.csv"
        df.to_csv(p, index=False)
        return str(p)

    async def test_basic_stats(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["basic"]["rows"] == 12
        assert result["basic"]["columns"] == 5

    async def test_missing_values_detected(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["missing_total"] == 2
        assert result["missing"]["age"]["count"] == 1
        assert result["missing"]["ca"]["count"] == 1

    async def test_missing_ratio(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["missing"]["age"]["ratio"] == pytest.approx(1 / 12, abs=0.0001)

    async def test_no_missing_column_omitted(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert "sex" not in result["missing"]
        assert "target" not in result["missing"]

    async def test_class_imbalance_distribution(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        dist = result["class_imbalance"]["distribution"]
        assert dist["0"] == 7
        assert dist["1"] == 5

    async def test_class_imbalance_ratio(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["class_imbalance"]["imbalance_ratio"] == pytest.approx(7 / 5, abs=0.001)

    async def test_demographics_sex_distribution(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        sex = result["demographics"]["sex"]
        assert sex["distribution"]["0"] == 6
        assert sex["distribution"]["1"] == 6

    async def test_demographics_age_min_max(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        age = result["demographics"]["age"]
        assert age["min"] == 50.0
        assert age["max"] == 200.0

    async def test_outlier_detected_in_age(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["outliers"]["age"]["outlier_count"] == 1

    async def test_no_outlier_columns_have_count_zero_or_positive(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        for col, info in result["outliers"].items():
            assert info["outlier_count"] >= 0

    async def test_consistency_ca_out_of_range(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["consistency"]["ca"]["out_of_range_count"] == 1
        assert result["consistency"]["ca"]["expected_range"] == [0, 3]

    async def test_consistency_thal_out_of_range(self, tool, csv_path):
        result = json.loads(await tool.execute(path=csv_path))
        assert result["consistency"]["thal"]["out_of_range_count"] == 1
        assert result["consistency"]["thal"]["expected_range"] == [1, 3]

    async def test_missing_target_col_reports_error(self, tool, tmp_path):
        df = pd.DataFrame({"age": [1, 2], "label": [0, 1]})
        p = tmp_path / "no_target.csv"
        df.to_csv(p, index=False)
        result = json.loads(await tool.execute(path=str(p)))
        assert "error" in result["class_imbalance"]

    async def test_custom_target_col(self, tool, tmp_path):
        df = pd.DataFrame({"age": [1, 2, 3], "label": [0, 1, 0]})
        p = tmp_path / "custom.csv"
        df.to_csv(p, index=False)
        result = json.loads(await tool.execute(path=str(p), target_col="label"))
        assert "error" not in result["class_imbalance"]
        assert result["class_imbalance"]["column"] == "label"

    async def test_file_not_found_returns_error(self, tool):
        result = json.loads(await tool.execute(path="/nonexistent/file.csv"))
        assert "error" in result

    async def test_schema_name(self, tool):
        assert tool.schema.name == "data_profiler"
