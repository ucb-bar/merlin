"""The ownership gate rejects new upward imports even inside lazy functions."""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root


def _gate():
    path = repo_root() / "build_tools/scripts/check_core_dependencies.py"
    spec = importlib.util.spec_from_file_location("core_dependency_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, relative, text=""):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_absolute_relative_and_lazy_imports_are_checked(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "src/merlin/targetgen/__init__.py")
    _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/evaluator.py")
    _write(
        tmp_path,
        "src/merlin/targetgen/core.py",
        '"""import merlin.targetgen.evaluator is only documentation"""\n'
        "from . import evaluator\n"
        "def execute():\n"
        "    from merlin.targetgen.evaluator import run\n"
        "    import merlin.targetgen.evaluator\n",
    )
    errors = _gate().audit(tmp_path)
    assert len(errors) == 3, errors
    assert all("optional merlin-experiments" in error for error in errors)
    assert any(":2:" in error for error in errors)


def test_shared_core_namespace_and_external_dependencies_are_not_optional(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "src/merlin/runtime/__init__.py")
    _write(tmp_path, "src/merlin/runtime/core.py", "from merlin import runtime\nimport numpy\n")
    _write(tmp_path, "packages/merlin-analysis/src/merlin/research/report.py")
    assert _gate().audit(tmp_path) == []


def test_optional_implicit_namespace_is_detected(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "src/merlin/core.py", "from merlin import research\n")
    _write(tmp_path, "packages/merlin-analysis/src/merlin/research/report.py")
    assert len(_gate().audit(tmp_path)) == 1


def test_compatibility_exemption_is_exactly_function_scoped(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "packages/merlin-mining/src/merlin/mining/fork.py")
    _write(
        tmp_path,
        "src/merlin/mining/from_strategy.py",
        "def mint_fork():\n    from .fork import mint_fork\ndef core_compile():\n    from .fork import mint_fork\n",
    )
    errors = _gate().audit(tmp_path)
    assert len(errors) == 1, errors
    assert ":4:" in errors[0]


def test_unparseable_core_cannot_pass_by_being_unscanned(tmp_path):
    _write(tmp_path, "src/merlin/core.py", "def broken(\n")
    assert "cannot parse" in _gate().audit(tmp_path)[0]


def test_compatibility_from_package_does_not_allow_sibling_imports(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/package_certification.py")
    _write(tmp_path, "packages/merlin-experiments/src/merlin/targetgen/other.py")
    _write(
        tmp_path,
        "src/merlin/targetgen/package_runtime.py",
        "def __getattr__(name):\n    from merlin.targetgen import package_certification, other\n",
    )
    errors = _gate().audit(tmp_path)
    assert len(errors) == 1, errors
    assert "merlin.targetgen.other" in errors[0]


@pytest.mark.parametrize(
    "module,function",
    [("conformance", "__getattr__"), ("contract/materialize", "_legacy_public_capsules_for")],
)
def test_removed_corpus_compatibility_cannot_reintroduce_upward_imports(tmp_path, module, function):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "packages/merlin-experiments/src/merlin_experiments/corpus/admission.py")
    _write(
        tmp_path,
        f"src/merlin/targetgen/{module}.py",
        f"def {function}():\n    from merlin_experiments.corpus import admission as corpus_workflow\n",
    )
    errors = _gate().audit(tmp_path)
    assert len(errors) == 1, errors
    assert "merlin_experiments.corpus.admission" in errors[0]


def test_repository_dependency_direction():
    assert _gate().audit(repo_root()) == []


def test_class_body_cannot_masquerade_as_a_lazy_compatibility_function(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "packages/merlin-mining/src/merlin/mining/fork.py")
    _write(tmp_path, "src/merlin/mining/from_strategy.py", "class mint_fork:\n    from .fork import mint_fork\n")
    assert len(_gate().audit(tmp_path)) == 1


def test_stale_core_copy_cannot_hide_an_optional_module(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "src/merlin/mining/fork.py")
    _write(tmp_path, "packages/merlin-mining/src/merlin/mining/fork.py")
    assert "duplicate source module merlin.mining.fork" in _gate().audit(tmp_path)[0]


def test_optional_initializer_wins_over_implicit_core_namespace(tmp_path):
    _write(tmp_path, "src/merlin/__init__.py")
    _write(tmp_path, "src/merlin/shared/core.py")
    _write(
        tmp_path,
        "src/merlin/consumer.py",
        "import merlin.shared\nfrom merlin.shared import core\nimport merlin.shared.core\n",
    )
    _write(tmp_path, "packages/merlin-analysis/src/merlin/shared/__init__.py")
    errors = _gate().audit(tmp_path)
    assert len(errors) == 3, errors
    assert all("optional merlin-analysis" in error for error in errors)


def test_missing_source_tree_is_not_a_vacuous_pass(tmp_path):
    assert "has no Python sources" in _gate().audit(tmp_path)[0]
