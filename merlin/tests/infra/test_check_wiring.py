"""The wiring gate: a production reference wires a module -- and a symbol; a test does not.

The second half of this file is the FUNCTION granularity. An imported module is "wired" whatever is
inside it, so ``llvmlower/device_build.py`` could carry ``routing_for_placement`` -- zero callers, ten
test references -- and report clean forever. Each symbol test below names the gate line it would fail
without; the ``__all__`` case is the exact line that hid the real defect.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
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


def test_exact_compatibility_alias_preserves_owner_debt_and_real_caller(tmp_path):
    alias = (
        '"""Compatibility identity."""\nimport sys\n'
        "from merlin.verification import probe as _owner\nsys.modules[__name__] = _owner\n"
    )
    gate = _gate(
        _tree(
            tmp_path,
            {
                "src/merlin/verify/probe.py": alias,
                "src/merlin/verification/probe.py": "def grade(): return 1\n",
                "merlin/tests/infra/test_probe.py": "from merlin.verification.probe import grade\n",
            },
        )
    )
    # A forwarding import alone cannot fabricate a production consumer of the checker.
    assert gate.unwired() == ["src/merlin/verification/probe.py"]
    assert gate.unwired_symbols() == []
    gate.LEDGER.parent.mkdir(parents=True)
    gate.LEDGER.write_text("merlin/python/merlin/verify/probe.py\n")
    assert gate.main([]) == 0
    _tree(tmp_path, {"src/merlin/consumer.py": "from merlin.verify import probe\n"})
    assert gate.unwired() == []
    # Stale debt still fails when the owner gains a real importer.
    assert gate.main([]) == 1
    gate.LEDGER.write_text("")
    _tree(tmp_path, {"src/merlin/verify/orphan.py": "SIGNAL = 1\n"})
    assert gate.unwired() == ["src/merlin/verify/orphan.py"]
    assert gate.main([]) == 1
    former = tmp_path / "src/merlin/verify/probe.py"
    assert gate._source_layout.module_alias(former, tmp_path) is not None
    former.write_text(alias + "EXTRA = True\n")
    assert gate._source_layout.module_alias(former, tmp_path) is None


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


_PORTABLE_LOADER = """DEPENDENCIES = ("first", "second")
def helper(name):
    if name not in DEPENDENCIES:
        raise ImportError(name)
    path = _TARGETGEN / (name + '.py')
    canonical = 'merlin.targetgen.' + name
    spec = importlib.util.spec_from_file_location(canonical, path)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    return selected

def prepare_helpers():
    for name in DEPENDENCIES:
        helper(name)
"""


@pytest.mark.parametrize("case", ["active", "constant_only", "no_execution", "uninvoked_initializer"])
def test_closed_portable_sources_require_invoked_loader(tmp_path, case):
    relative = "src/merlin/compiler/quantization/torchao/portable.py"
    loader = _PORTABLE_LOADER
    initializer = "from .portable import prepare_helpers\nprepare_helpers()\n"
    if case == "constant_only":
        loader = 'DEPENDENCIES = ("first", "second")\n'
    elif case == "no_execution":
        loader = loader.replace("spec.loader.exec_module(selected)", "selected = None")
    elif case == "uninvoked_initializer":
        initializer = "from .portable import prepare_helpers\n"
    _tree(tmp_path, {relative: loader, str(Path(relative).parent / "__init__.py"): initializer})
    gate = _gate(tmp_path)
    observed = gate.wiring_sources.closed_source_imports(tmp_path / relative, tmp_path, ast.parse(loader))
    assert observed == ({"merlin.targetgen.first", "merlin.targetgen.second"} if case == "active" else set())


def _published_compiler_tree(root):
    source = "def existing_api(): return 1\n"
    relative = "src/merlin/perf/reader.py"
    digest = hashlib.sha256(source.encode()).hexdigest()
    document = {
        "schema": "merlin.fresh10_tooling_projection.v1",
        "profiles": {
            "fixture": {
                "full_members": [
                    {
                        "module": "merlin.perf.reader",
                        "path": "merlin/perf/reader.py",
                        "source_relative": relative,
                        "sha256": digest,
                        "baseline_sha256": digest,
                    }
                ]
            }
        },
    }
    raw = json.dumps(document, sort_keys=True)
    producer = (
        f"DATA_SHA256 = {hashlib.sha256(raw.encode()).hexdigest()!r}\n"
        + """
def prepare_sources(repo, destination, profile_name):
    rows = _rows(profile_name, "full")
    payloads = [(row, _payload(repo, row)) for row in rows]
    selected = F.select(destination, payloads)
    verify_sources(profile_name, selected)
    return selected
"""
    )
    stage = """def stage(selection, submission):
    for relative, digest, module in selection.members:
        path = submission / relative
        payload = _read(selection.root, relative)
        with path.open("xb") as stream:
            stream.write(payload)
"""
    _tree(
        root,
        {
            relative: source,
            "src/merlin_experiments/phase1/fresh10_profile_data.json": raw,
            "src/merlin_experiments/phase1/fresh10_sources.py": producer,
            "src/merlin_experiments/phase1/full_compiler_workspace.py": stage,
        },
    )
    return root / relative


@pytest.mark.parametrize("case", ["active", "changed_owner", "new_api", "stale_inventory", "disconnected_copy"])
def test_published_source_credit_binds_actual_current_bodies(tmp_path, case):
    source = _published_compiler_tree(tmp_path)
    base = tmp_path / "src/merlin_experiments/phase1"
    if case == "changed_owner":
        source.write_text("def existing_api(): return 2\n")
    elif case == "new_api":
        source.write_text(source.read_text() + "def new_api(): return 3\n")
        _tree(tmp_path, {"merlin/tests/infra/test_reader.py": "from merlin.perf.reader import new_api\n"})
    elif case == "stale_inventory":
        path = base / "fresh10_profile_data.json"
        path.write_text(path.read_text() + "\n")
    elif case == "disconnected_copy":
        path = base / "full_compiler_workspace.py"
        path.write_text(path.read_text().replace("stream.write(payload)", "stream.write(b'unrelated')"))
    gate = _gate(tmp_path)
    if case == "stale_inventory":
        with pytest.raises(ValueError, match="producer pin"):
            gate.wiring_sources.published_source_modules(tmp_path)
    else:
        assert gate.wiring_sources.published_source_modules(tmp_path) == (
            {"merlin.perf.reader"} if case == "active" else set()
        )
        if case == "new_api":
            assert any(name.endswith("::new_api") for name in gate.unwired_symbols(set()))


@pytest.fixture
def pinned_backend(tmp_path):
    source = tmp_path / "provider"
    _tree(
        source,
        {
            "pyproject.toml": '[tool.setuptools]\npackages = ["fixture"]\npackage-dir = {fixture = "python/fixture"}\n',
            ".gitignore": "generated.py\n",
            "python/fixture/__init__.py": "",
            "python/fixture/consumer.py": "from merlin.perf import tracked_owner\n",
            "tools/inspect.py": "from merlin.perf import tracked_tool_owner\n",
        },
    )

    def git(*arguments):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "-c",
                "core.autocrlf=false",
                "-c",
                "user.name=Wiring fixture",
                "-c",
                "user.email=wiring-fixture@example.invalid",
                "-c",
                "commit.gpgsign=false",
                *arguments,
            ],
            text=True,
        ).strip()

    git("init", "-q")
    git("add", "--", ".")
    git("commit", "-q", "--no-verify", "-m", "Create bounded wiring fixture")
    revision = git("rev-parse", "HEAD")
    pins = tmp_path / "build_tools/upstreams/backend-boundaries.json"
    pins.parent.mkdir(parents=True)
    pins.write_text(json.dumps({"backends": [{"id": "fixture", "revision": revision}]}))
    return source, pins, git


@pytest.mark.parametrize(
    "case", ["tracked_only", "dirty", "wrong_revision", "missing_id", "skip_worktree_edit", "packaging_edit"]
)
def test_backend_consumers_are_exact_tracked_head_payloads(tmp_path, pinned_backend, case):
    source, pins, git = pinned_backend
    selected = {"fixture": str(source)}
    consumer = source / "python/fixture/consumer.py"
    # An ignored generated file is not part of the reviewed consumer source.
    generated = source / "python/fixture/generated.py"
    generated.write_text("from merlin.perf import fabricated_consumer\n")
    if case == "dirty":
        consumer.write_text("from merlin.perf import changed_consumer\n")
    elif case == "wrong_revision":
        pins.write_text(json.dumps({"backends": [{"id": "fixture", "revision": "0" * 40}]}))
    elif case == "missing_id":
        selected = {}
    elif case == "skip_worktree_edit":
        git("update-index", "--skip-worktree", "python/fixture/consumer.py")
        consumer.write_text("from merlin.perf import hidden_changed_consumer\n")
        assert git("status", "--porcelain=v1", "--untracked-files=all") == ""
    elif case == "packaging_edit":
        git("update-index", "--skip-worktree", "pyproject.toml")
        (source / "pyproject.toml").write_text("[tool.setuptools]\npackages = []\n")
        assert git("status", "--porcelain=v1", "--untracked-files=all") == ""
    gate = _gate(tmp_path)
    if case == "tracked_only":
        files = gate.wiring_sources.selected_backend_sources(selected, tmp_path)
        assert set(files) == {consumer, source / "python/fixture/__init__.py", source / "tools/inspect.py"}
        assert generated not in files
        gate.EXTERNAL_PRODUCTION = files
        assert "merlin.perf.fabricated_consumer" not in set().union(*(gate._imports(path) for path in files))
        assert "merlin.perf.tracked_owner" in gate._imports(consumer)
    else:
        with pytest.raises(ValueError, match="changed|revision|every exact|bytes differ"):
            gate.wiring_sources.selected_backend_sources(selected, tmp_path)


def test_saved_backend_consumer_report_requires_current_pin_identity(tmp_path, pinned_backend):
    source, pins, _ = pinned_backend
    report = tmp_path / "report.json"
    body = {
        "schema": "merlin.backend-boundaries.v1",
        "passed": True,
        "pins_sha256": hashlib.sha256(pins.read_bytes()).hexdigest(),
        "sources": {"fixture": str(source)},
    }
    report.write_text(json.dumps(body))
    gate = _gate(tmp_path)
    assert source / "python/fixture/consumer.py" in gate.wiring_sources.backend_sources(report, tmp_path)
    body["pins_sha256"] = "0" * 64
    report.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="current reviewed pins"):
        gate.wiring_sources.backend_sources(report, tmp_path)


@pytest.mark.parametrize("case", ["valid", "missing", "wrong_revision", "changed", "skip_worktree", "escape"])
def test_source_only_consumers_bind_exact_original_call_chain(tmp_path, pinned_backend, case):
    source, pins, git = pinned_backend
    identity = "support-source"
    revision = git("rev-parse", "HEAD")
    body = json.loads(pins.read_text())
    body["source_consumers"] = [
        {
            "id": identity,
            "revision": revision,
            "repository": "https://example.invalid/support.git",
            "files": ["python/fixture/consumer.py"],
        }
    ]
    pins.write_text(json.dumps(body))
    selected = {identity: str(source)}
    consumer = source / "python/fixture/consumer.py"
    if case == "missing":
        selected = {}
    elif case == "wrong_revision":
        body["source_consumers"][0]["revision"] = "0" * 40
        pins.write_text(json.dumps(body))
    elif case in {"changed", "skip_worktree"}:
        if case == "skip_worktree":
            git("update-index", "--skip-worktree", "python/fixture/consumer.py")
        consumer.write_text("from merlin.perf import fabricated_consumer\n")
    elif case == "escape":
        body["source_consumers"][0]["files"] = ["../outside.py"]
        pins.write_text(json.dumps(body))
    gate = _gate(tmp_path)
    if case == "valid":
        files = gate.wiring_sources.selected_static_sources(selected, tmp_path)
        assert files == (consumer,)
        assert "merlin.perf.tracked_owner" in gate._imports(consumer)
    else:
        with pytest.raises(ValueError, match="every exact|revision|changed|bytes differ|escapes"):
            gate.wiring_sources.selected_static_sources(selected, tmp_path)


@pytest.mark.parametrize("case", ["valid", "wrong_schema", "failed", "missing_static"])
def test_report_requires_selected_static_sources_without_granting_execution(tmp_path, pinned_backend, case):
    source, pins, git = pinned_backend
    manifest = json.loads(pins.read_text())
    manifest["source_consumers"] = [
        {
            "id": "support-source",
            "revision": git("rev-parse", "HEAD"),
            "repository": "https://example.invalid/support.git",
            "files": ["python/fixture/consumer.py"],
        }
    ]
    pins.write_text(json.dumps(manifest))
    body = {
        "schema": "merlin.backend-boundaries.v1",
        "passed": True,
        "pins_sha256": hashlib.sha256(pins.read_bytes()).hexdigest(),
        "sources": {"fixture": str(source)},
        "static_sources": {"support-source": str(source)},
    }
    if case == "wrong_schema":
        body["schema"] = "unrelated.report"
    elif case == "failed":
        body["passed"] = False
    elif case == "missing_static":
        del body["static_sources"]
    report = tmp_path / "report.json"
    report.write_text(json.dumps(body))
    gate = _gate(tmp_path)
    if case == "valid":
        assert source / "python/fixture/consumer.py" in gate.wiring_sources.backend_sources(report, tmp_path)
    else:
        with pytest.raises(ValueError, match="current reviewed pins|every exact"):
            gate.wiring_sources.backend_sources(report, tmp_path)


def test_static_consumer_sparse_checkout_materializes_only_selected_tracked_files(tmp_path, pinned_backend):
    source, _, git = pinned_backend
    gate = _gate(tmp_path)
    location = Path(gate.__file__).with_name("check_backend_boundaries.py")
    spec = importlib.util.spec_from_file_location("source_only_boundary_fixture", location)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    destination = tmp_path / "retained-support-source"
    commands = []
    row = {
        "id": "support-source",
        "repository": str(source),
        "revision": git("rev-parse", "HEAD"),
        "files": ["python/fixture/consumer.py"],
    }
    runner.source_checkout(row, destination, tmp_path, dict(os.environ), commands, sparse=True)
    assert (destination / "python/fixture/consumer.py").is_file()
    assert not (destination / "tools/inspect.py").exists()
    assert not (destination / "python/fixture/__init__.py").exists()
    assert all(command["argv"][0] == "git" for command in commands)
    fetches = [command["argv"] for command in commands if "fetch" in command["argv"]]
    assert len(fetches) == 1 and "--filter=blob:none" in fetches[0]
    assert runner.git(destination, "config", "--get", "remote.origin.promisor") == "true"
    assert runner.git(destination, "config", "--get", "remote.origin.partialclonefilter") == "blob:none"
    assert runner.git(destination, "rev-parse", "HEAD") == row["revision"]
    assert gate.wiring_sources.tracked_consumer_files(
        destination, row["revision"], {destination / row["files"][0]}
    ) == (destination / row["files"][0],)


@pytest.mark.parametrize(
    "case", ["valid", "duplicate_backend", "execution_fields", "escaping_path", "floating_revision"]
)
def test_static_consumer_roster_does_not_create_an_installed_lane(tmp_path, case):
    gate = _gate(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "source_pin_boundary_fixture", Path(gate.__file__).with_name("check_backend_boundaries.py")
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    manifest = json.loads((repo_root() / "build_tools/upstreams/backend-boundaries.json").read_text())
    assert len(manifest["backends"]) == 6
    assert len(manifest["source_consumers"]) == 1
    row = manifest["source_consumers"][0]
    if case == "duplicate_backend":
        row["id"] = manifest["backends"][0]["id"]
    elif case == "execution_fields":
        row["tests"] = ["test_support.py"]
    elif case == "escaping_path":
        row["files"] = ["../outside.py"]
    elif case == "floating_revision":
        row["revision"] = "main"
    path = tmp_path / "pins.json"
    path.write_text(json.dumps(manifest))
    backends = runner.pins(path)
    assert len(backends) == 6
    if case == "valid":
        assert runner.static_pins(path, backends) == [row]
        assert row["id"] not in {backend["id"] for backend in backends}
    else:
        with pytest.raises(ValueError, match="consumer|escapes"):
            runner.static_pins(path, backends)
