"""The wiring gate: a production reference wires a module -- and a symbol; a test does not.

The second half of this file is the FUNCTION granularity. An imported module is "wired" whatever is
inside it, so ``llvmlower/device_build.py`` could carry ``routing_for_placement`` -- zero callers, ten
test references -- and report clean forever. Each symbol test below names the gate line it would fail
without; the ``__all__`` case is the exact line that hid the real defect.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

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
    module.SYMBOL_LEDGER = root / "build_tools" / "scripts" / "unwired_symbols_ratchet.txt"
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
    (package / "_reference.py").write_text("def freeze(): ...\n", encoding="utf-8")
    worker.write_text("import sys\nfrom _reference import freeze\nfrom json import dumps\n", encoding="utf-8")
    assert "merlin.targetgen._reference" in gate._imports(worker)
    assert "merlin.targetgen.json" not in gate._imports(worker)  # no such sibling file


@pytest.mark.parametrize(
    "selection",
    [
        'from merlin.common.paths import module_source_path\nreader = module_source_path("merlin.perf.reader")\n',
        'from merlin.common.paths import module_source_path as select\nreader = select(module="merlin.perf.reader")\n',
        'import merlin.common.paths\nreader = merlin.common.paths.module_source_path("merlin.perf.reader")\n',
        'import merlin.common.paths as P\nreader = P.module_source_path("merlin.perf.reader")\n',
        'from merlin.common import paths as P\nreader = P.module_source_path("merlin.perf.reader")\n',
    ],
)
def test_canonical_literal_source_selection_wires_an_isolated_reader(tmp_path, selection):
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/reader.py": "def main(): ...\n",
                "build_tools/worker.py": selection,
            },
        )
    )
    assert gate.unwired() == []


@pytest.mark.parametrize(
    "selection",
    [
        'from other.paths import module_source_path\nmodule_source_path("merlin.perf.reader")\n',
        'module_source_path("merlin.perf.reader")\n',
        "from merlin.common.paths import module_source_path\n"
        'request = "merlin.perf.reader"\nmodule_source_path(request)\n',
        'from merlin.common.paths import module_source_path\nmodule_source_path("merlin.perf." + "reader")\n',
        'from merlin.common.paths import module_source_path\nmodule_source_path("merlin.perf.reader", extra=True)\n',
        'from merlin.common.paths import module_source_path\nmodule_source_path(other="merlin.perf.reader")\n',
        'from merlin.common.paths import module_source_path\n"module_source_path(merlin.perf.reader)"\n',
    ],
)
def test_computed_or_unbound_source_requests_do_not_wire_a_module(tmp_path, selection):
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/reader.py": "def main(): ...\n",
                "build_tools/worker.py": selection,
            },
        )
    )
    assert gate.unwired() == [f"{_PERF}/reader.py"]


def test_a_test_only_source_selection_is_not_a_production_caller(tmp_path):
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/reader.py": "def main(): ...\n",
                "merlin/tests/infra/test_reader.py": (
                    "from merlin.common.paths import module_source_path\n"
                    'reader = module_source_path("merlin.perf.reader")\n'
                ),
            },
        )
    )
    assert gate.unwired() == [f"{_PERF}/reader.py"]


def test_a_target_workflow_under_examples_is_a_production_caller(tmp_path: Path) -> None:
    """Target workflows moved from ``merlin/targets`` to ``examples/<name>``; their documented
    operator commands still wire what they import. A sample with no ``target/`` and no
    ``experiment.yaml`` does not, and neither does a workflow's own test suite."""
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{_PERF}/__init__.py": "",
                f"{_PERF}/bound.py": "X = 1\n",
                f"{_PERF}/probed.py": "X = 1\n",
                f"{_PERF}/sampled.py": "X = 1\n",
                f"{_PERF}/tested.py": "X = 1\n",
                "examples/acc/target/descriptor.yaml": "target: acc\n",
                "examples/acc/verification/bind.py": "from merlin.perf.bound import X\n",
                "examples/acc/tests/test_bind.py": "from merlin.perf import tested\n",
                "examples/npu/experiment.yaml": "id: npu\n",
                "examples/npu/phase0/probe.py": "import merlin.perf.probed\n",
                "examples/samples/demo.py": "from merlin.perf import sampled\n",
            },
        )
    )
    assert gate._target_workflow_roots() == ("examples/acc", "examples/npu")
    assert gate.unwired() == [f"{_PERF}/sampled.py", f"{_PERF}/tested.py"]


def test_only_executable_commands_in_the_rendered_shared_prompt_are_wired(tmp_path: Path) -> None:
    base = "src/merlin/targetgen"
    prompt = f"{base}/generate_prompt.py"
    command = "merlin.targetgen.oot_starterkit.plan"
    plan = f"{base}/oot_starterkit/plan.py"
    files = {
        prompt: (
            '"""Docs mention python -m merlin.targetgen.docs_only."""\n'
            '_TEMPLATE = "Use python -m merlin.targetgen.oot_starterkit.plan inventory"\n'
            '_DORMANT = "python -m merlin.targetgen.dormant"\n'
            "def render_prompt():\n    return _TEMPLATE.format()\n"
        ),
        plan: 'def main():\n    return 0\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
        f"{base}/docs_only.py": 'def main():\n    return 0\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
        f"{base}/dormant.py": 'def main():\n    return 0\nif __name__ == "__main__":\n    raise SystemExit(main())\n',
        f"{base}/not_executable.py": "def main():\n    return 0\n",
        "src/merlin/driver.py": "from merlin.targetgen.generate_prompt import render_prompt\n",
        "docs/guide.md": "python -m merlin.targetgen.docs_only\n",
    }
    gate = _gate(_tree(tmp_path, files))
    assert gate._generated_prompt_module_commands(
        {"merlin.targetgen.generate_prompt": {tmp_path / "src/merlin/driver.py"}},
        {command: tmp_path / plan},
    ) == {command}
    assert gate.unwired() == [f"{base}/docs_only.py", f"{base}/dormant.py", f"{base}/not_executable.py"]
    files[prompt] = files[prompt].replace(
        "Use python -m merlin.targetgen.oot_starterkit.plan inventory",
        "Use python -m merlin.targetgen.not_executable inventory",
    )
    gate = _gate(_tree(tmp_path, files))
    assert gate.unwired() == [f"{base}/docs_only.py", f"{base}/dormant.py", f"{base}/not_executable.py", plan]
    files["src/merlin/driver.py"] = "# A dormant generator is not a task command.\n"
    gate = _gate(_tree(tmp_path, files))
    assert gate.unwired() == [f"{base}/docs_only.py", f"{base}/dormant.py", prompt, f"{base}/not_executable.py", plan]


@pytest.mark.parametrize("source", ["src/merlin", "packages/extra/src/merlin"])
def test_from_bare_sibling_resolves_only_existing_local_modules(tmp_path, source):
    base = f"{source}/targetgen"
    gate = _gate(
        _tree(
            tmp_path,
            {
                f"{base}/_helper.py": "X = 1\n",
                f"{base}/_worker.py": (
                    "from _helper import X as value\nfrom absent import X\nfrom external.helper import X\n"
                ),
                "src/merlin/driver.py": "import merlin.targetgen._worker\n",
            },
        )
    )
    imports = gate._imports(tmp_path / base / "_worker.py")
    assert "merlin.targetgen._helper" in imports
    assert "merlin.targetgen.absent" not in imports
    assert "merlin.targetgen.external.helper" not in imports
    assert gate.unwired() == []


# --------------------------------------------------------------------------------------------
# Function granularity. Each test names the gate line it would fail without.
# --------------------------------------------------------------------------------------------

#: A package whose public definitions differ only in HOW they are reached. `orphan` is the shape the
#: gate exists for: exported, tested, called by nothing.
_SYMBOL_TREE = {
    f"{_PERF}/__init__.py": "",
    f"{_PERF}/widget.py": (
        '"""A wired module with one orphan inside it."""\n'
        "\n"
        '__all__ = ["called_one", "orphan", "registered_one", "dispatched", "Untested"]\n'
        "\n"
        "def _register(fn):\n"
        "    return fn\n"
        "\n"
        "def called_one():\n"
        "    return 1\n"
        "\n"
        "def orphan(placement):\n"
        "    return placement\n"
        "\n"
        "@_register\n"
        "def registered_one():\n"
        "    return 3\n"
        "\n"
        "def dispatched():\n"
        "    return 5\n"
        "\n"
        "class Untested:\n"
        "    pass\n"
        "\n"
        "def _private_orphan():\n"
        "    return 4\n"
    ),
    f"{_PERF}/caller.py": (
        "from merlin.perf.widget import called_one\n"
        "import merlin.perf.widget as W\n"
        "\n"
        "def go():\n"
        '    return called_one() + getattr(W, "dispatched")()\n'
    ),
    "merlin/experiments/run.py": "import merlin.perf.caller\nimport merlin.perf.widget\n",
    "merlin/tests/infra/test_widget.py": (
        "from merlin.perf.widget import dispatched, orphan, registered_one\n"
        "\n"
        "def test_orphan():\n"
        "    assert orphan(1) == 1 and registered_one() == 3 and dispatched() == 5\n"
    ),
}


def test_a_public_definition_with_tests_and_no_caller_is_named(tmp_path: Path) -> None:
    """THE REGRESSION. `orphan` is exported, imported by a test, and called by no production code.

    `called_one` has a production caller; `dispatched` is reached by name through `getattr`;
    `registered_one` is decorated, so the gate declines to judge it; `Untested` has no test behind it
    and belongs to ordinary dead-code review; `_private_orphan` is private.
    """
    gate = _gate(_tree(tmp_path, _SYMBOL_TREE))
    assert gate.unwired_symbols(set()) == [f"{_PERF}/widget.py::orphan"]


def test_an_export_list_is_not_a_caller(tmp_path: Path) -> None:
    """`__all__` NAMES a symbol, it does not USE one. Counting it made a function with ten test
    references and no caller read as wired. Drop `_export_list_nodes` and this goes red."""
    gate = _gate(_tree(tmp_path, _SYMBOL_TREE))
    assert "orphan" not in gate._referenced_names(tmp_path / f"{_PERF}/widget.py")
    assert "called_one" in gate._referenced_names(tmp_path / f"{_PERF}/caller.py")


def test_a_test_reference_does_not_wire_a_symbol_and_a_package_test_counts_as_a_test(tmp_path: Path) -> None:
    """Counting a test suite as production is the other way to make this gate unable to fail; and a
    distribution's own tests are tests, so a symbol only they exercise is still debt."""
    files = dict(_SYMBOL_TREE)
    files["packages/merlin-extra/tests/test_more.py"] = "from merlin.perf.widget import Untested\n"
    gate = _gate(_tree(tmp_path, files))
    assert not any(part.endswith("tests") for part in gate.PRODUCTION)
    assert gate.unwired_symbols(set()) == [f"{_PERF}/widget.py::Untested", f"{_PERF}/widget.py::orphan"]


def test_a_symbol_whose_module_is_already_ledgered_is_not_recorded_twice(tmp_path: Path) -> None:
    """One debt, one entry. The coarser ledger owns a module nothing imports at all."""
    gate = _gate(_tree(tmp_path, _SYMBOL_TREE))
    assert gate.unwired_symbols({f"{_PERF}/widget.py"}) == []


def test_a_console_script_entry_point_is_wired_by_its_packaging(tmp_path: Path) -> None:
    """An entry point has no in-tree caller by construction; the metadata is its caller."""
    files = dict(_SYMBOL_TREE)
    files[f"{_PERF}/widget.py"] = files[f"{_PERF}/widget.py"].replace("def orphan(", "def main(")
    files["merlin/tests/infra/test_widget.py"] = "from merlin.perf.widget import main\n"
    files["pyproject.toml"] = '[project.scripts]\nwidget = "merlin.perf.widget:main"\n'
    gate = _gate(_tree(tmp_path, files))
    assert gate.unwired_symbols(set()) == []


def test_a_new_unwired_symbol_fails_and_a_stale_symbol_entry_fails(tmp_path: Path, capsys) -> None:
    """The verdict path, not just the scan: the symbol ledger must be able to turn the exit code."""
    gate = _gate(_tree(tmp_path, _SYMBOL_TREE))
    gate.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    gate.LEDGER.write_text("", encoding="utf-8")
    assert gate.main([]) == 1
    assert "unwired symbol" in capsys.readouterr().out
    gate.SYMBOL_LEDGER.write_text(f"{_PERF}/widget.py::orphan\n", encoding="utf-8")
    assert gate.main([]) == 0
    gate.SYMBOL_LEDGER.write_text(
        f"{_PERF}/widget.py::orphan\n{_PERF}/widget.py::called_one  # was debt once\n", encoding="utf-8"
    )
    assert gate.main([]) == 1
    assert "stale ledger entry" in capsys.readouterr().out


def test_a_relocated_module_keeps_its_symbol_identity(tmp_path: Path) -> None:
    """Symbol debt is keyed by the stable policy path, so moving a module into `src/` neither forgives
    its entries nor makes them look new."""
    files = {k.replace("merlin/python/merlin", "src/merlin"): v for k, v in _SYMBOL_TREE.items()}
    gate = _gate(_tree(tmp_path, files))
    assert gate.unwired_symbols(set()) == [f"{_PERF}/widget.py::orphan"]
    assert gate.unwired_symbols({"src/merlin/perf/widget.py"}) == []


# --------------------------------------------------------------------------- prefix dispatch

_DISPATCHER = "packages/merlin-experiments/src/merlin_experiments/phase2/claims/dispatch.py"
_PREFIX_TREE = {
    f"{_PERF}/reached_claim.py": "def preflight_reached(x):\n    return x\n\ndef analyze_reached(x):\n    return x\n",
    f"{_PERF}/other_claim.py": "def preflight_other(x):\n    return x\n",
    _DISPATCHER: (
        'PREFLIGHT_PREFIX = "preflight_"\n'
        "\n"
        "def _registry():\n"
        "    from merlin.perf import reached_claim as R\n"
        "    return {'reached': R.analyze_reached}\n"
    ),
    "merlin/tests/infra/test_claims.py": (
        "from merlin.perf.reached_claim import preflight_reached\nfrom merlin.perf.other_claim import preflight_other\n"
    ),
}


def test_a_prefix_dispatched_entry_point_is_wired_only_in_a_module_the_dispatcher_imports(tmp_path: Path) -> None:
    """``dispatch.resolve`` runs the one ``preflight_*`` of each analyzer module it loads, and no file
    spells that name. The same definition in a module the dispatcher does not import is still debt."""
    gate = _gate(_tree(tmp_path, _PREFIX_TREE))
    assert gate.unwired_symbols(set()) == [f"{_PERF}/other_claim.py::preflight_other"]


def test_a_dispatcher_that_stops_publishing_its_prefix_fails_rather_than_reaching_nothing(tmp_path: Path) -> None:
    files = dict(_PREFIX_TREE)
    files[_DISPATCHER] = files[_DISPATCHER].replace("PREFLIGHT_PREFIX", "ENTRY_PREFIX")
    gate = _gate(_tree(tmp_path, files))
    with pytest.raises(SystemExit, match="no longer publishes PREFLIGHT_PREFIX"):
        gate.unwired_symbols(set())


def test_every_declared_prefix_dispatcher_exists_and_publishes_its_prefix() -> None:
    """On the real tree: a moved or renamed dispatcher must move its declaration with it."""
    gate = _gate(repo_root())
    for relative, constant in gate.PREFIX_DISPATCHERS:
        path = repo_root() / relative
        assert path.is_file(), relative
        assert gate._published_constant(gate._tree(path), constant), (relative, constant)
