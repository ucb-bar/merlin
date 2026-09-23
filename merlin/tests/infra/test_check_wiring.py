"""The wiring gate: an import in production code wires a module; a test, a comment or itself does not."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from merlin.common.paths import repo_root


def _gate(root: Path):
    spec = importlib.util.spec_from_file_location(
        "check_wiring_under_test", repo_root() / "build_tools/scripts/check_wiring.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT, module.PACKAGE_ROOT = root, root / "merlin" / "python"
    module.LEDGER = root / "build_tools" / "scripts" / "unwired_ratchet.txt"
    return module


def _tree(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


_PERF = "merlin/python/merlin/perf"


def test_shared_extension_evaluators_remain_instrumented(tmp_path):
    source = "packages/merlin-experiments/src/merlin/targetgen"
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{source}/grader.py": "X = 1\n",
                f"{source}/orphan.py": "X = 1\n",
                "src/merlin/driver.py": "from merlin.targetgen import grader\n",
                "packages/merlin-experiments/tests/test_grader.py": "from merlin.targetgen import orphan\n",
            },
        )
    )
    # Debt keeps its original module identity across the move; tests are not production callers.
    assert gate.unwired() == [f"{source}/orphan.py"]
    gate.LEDGER.parent.mkdir(parents=True)
    gate.LEDGER.write_text("merlin/python/merlin/targetgen/orphan.py\n")
    assert gate.main([]) == 0


def test_only_a_production_import_wires_a_module(tmp_path: Path) -> None:
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/__init__.py": "",
                f"{_PERF}/called.py": "X = 1\n",
                f"{_PERF}/lazy.py": "X = 1\n",
                f"{_PERF}/relative.py": "X = 1\n",
                f"{_PERF}/only_tested.py": "X = 1\n",
                f"{_PERF}/only_named.py": "X = 1\n",
                f"{_PERF}/cli.py": "def main(): ...\n",
                f"{_PERF}/user.py": (
                    "from merlin.perf import called\nfrom . import relative\n"
                    "# merlin.perf.only_named is mentioned, never imported\n"
                    "def f():\n    from merlin.perf.lazy import X\n    return X\n"
                ),
                "merlin/tests/infra/test_x.py": "from merlin.perf import only_tested\n",
                "merlin/experiments/run.py": "import merlin.perf.user\n",
                "pyproject.toml": '[project.scripts]\ntool = "merlin.perf.cli:main"\n[tool.x]\ny = "z"\n',
            },
        )
    )
    assert gate.unwired() == [f"{_PERF}/only_named.py", f"{_PERF}/only_tested.py"]


def test_new_debt_fails_and_a_stale_entry_fails(tmp_path: Path, capsys) -> None:
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/__init__.py": "",
                f"{_PERF}/orphan.py": "X = 1\n",
                f"{_PERF}/wired.py": "X = 1\n",
                "build_tools/use.py": "from merlin.perf import wired\n",
            },
        )
    )
    assert gate.main([]) == 1 and "unwired" in capsys.readouterr().out
    gate.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    gate.LEDGER.write_text(f"{_PERF}/orphan.py\n{_PERF}/wired.py  # was debt once\n", encoding="utf-8")
    assert gate.main([]) == 1 and "stale ledger entry" in capsys.readouterr().out
    gate.LEDGER.write_text(f"# comment\n{_PERF}/orphan.py\n", encoding="utf-8")
    assert gate.main([]) == 0


def test_a_worker_importing_its_sibling_by_bare_name_counts_as_a_caller(tmp_path, monkeypatch):
    import importlib.util

    from merlin.common.paths import repo_root

    spec = importlib.util.spec_from_file_location(
        "check_wiring_under_test", repo_root() / "build_tools/scripts/check_wiring.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    package = tmp_path / "merlin/python/merlin/targetgen"
    package.mkdir(parents=True)
    (package / "_helper.py").write_text("X = 1\n", encoding="utf-8")
    worker = package / "_worker.py"
    worker.write_text("import sys\nimport _helper as H\n", encoding="utf-8")
    monkeypatch.setattr(gate, "PACKAGE_ROOT", tmp_path / "merlin/python")
    assert "merlin.targetgen._helper" in gate._imports(worker)
    assert "merlin.targetgen.sys" not in gate._imports(worker)  # no such sibling file
