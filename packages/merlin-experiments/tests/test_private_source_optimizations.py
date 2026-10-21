"""Actual ordinary feedback call expression, with pure synthetic build products.

This does not bypass or qualify the private whole-model gate. Its original call
expression is executed under neutral original inputs to check option transport;
all actual pass/report/recipe checks come from the core transport fixture.
"""

from __future__ import annotations

import ast
import inspect
import runpy
import sys
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import private_full_models as full_models
from merlin_experiments.phase1.feedback import private_source_optimizations as feedback

from merlin.common.paths import merlin_dir
from merlin.llvmlower.reviewed_pass_capabilities import (
    PassCapabilityBinding,
    ReviewedPassCapabilities,
    ReviewedPassCapability,
)
from merlin.targetgen.compiler_pass_selection import CompilerPassSelection, scoped_compiler_passes


def core_fixture(tmp_path):
    path = merlin_dir() / "tests/infra/test_compiler_source_optimizations.py"
    namespace = runpy.run_path(str(path))
    case = namespace["case"].__wrapped__(tmp_path)
    return namespace, case


def actual_build_expression(namespace):
    """Execute the repository's actual compile call, without claiming gate admission."""
    source = inspect.getsource(full_models.run)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "source_optimizations"
        and node.func.attr == "compile_model"
    ]
    assert len(calls) == 1
    expression = ast.Expression(calls[0])
    # Keep the real source filename for original input-reader implementation pins.
    return eval(compile(expression, inspect.getsourcefile(full_models.run), "eval"), namespace)


def test_unselected_feedback_does_not_import_feature_read_context_or_add_options(monkeypatch):
    monkeypatch.delitem(sys.modules, "merlin.targetgen.compiler_source_optimizations", raising=False)
    called = []

    def compiler(**options):
        called.append(options)
        return "unchanged"

    assert (
        feedback.compile_model(
            compiler,
            compiler_package="absent",
            original_inputs=lambda: pytest.fail("unselected context read"),
            exact_original="unchanged",
        )
        == "unchanged"
    )
    assert called == [{"exact_original": "unchanged"}]
    assert "merlin.targetgen.compiler_source_optimizations" not in sys.modules


def test_actual_feedback_expression_passes_original_typed_policy_and_current_inputs(tmp_path):
    fixtures, case = core_fixture(tmp_path)
    old_owner, _, package, capture, files = case
    digest = fixtures["digest"]
    original = fixtures["OriginalOwner"]

    class FeedbackOwner(original):
        def source_optimization_binding(self, *, inputs):
            self.calls.append("binding")
            # Runtime/library come from independently held original selections,
            # not from the selected binding or candidate-provided policy fields.
            return PassCapabilityBinding(
                digest(inputs["facts"]["path"]),
                digest(inputs["software"]["path"]),
                inputs["numerical_policy"],
                digest(files["runtime"]),
                digest(files["library"]),
            )

    owner = FeedbackOwner(old_owner.evidence)
    binding = PassCapabilityBinding(*(digest(files[role]) for role in fixtures["ROLES"]))
    selected = CompilerPassSelection(
        ReviewedPassCapabilities((ReviewedPassCapability(fixtures["bridge"].MECHANISM, binding, owner),)),
        binding,
        mechanisms=(fixtures["bridge"].MECHANISM,),
    )
    catalog, dts = tmp_path / "boards.yaml", tmp_path / "host.dts"
    catalog.write_text("neutral original board catalog")
    dts.write_text("neutral original device tree")
    received = []

    def compiler(**options):
        received.append(options)
        return fixtures["prepared_compiler"](**options)

    environment = dict(
        source_optimizations=feedback,
        compile_saved_model=compiler,
        package=package,
        facts_path=files["hardware"],
        software_path=files["software"],
        capability_path=files["runtime"],
        host_path=files["library"],
        recipe_proof={"recipe_semantic_sha256": digest(files["numerical_policy"])},
        stage=capture,
        host_package=package,
        catalog=catalog,
        program_boards={"model": "neutral-board"},
        program="model",
        dts=dts,
        base=tmp_path / "products",
        name="neutral-source",
        target="neutral-target",
        row={"arena_mb": 1},
        device=None,
        linkage_support=type("Linkage", (), {"build_options": staticmethod(lambda *_: {})})(),
        source={},
        selected_math={"policy": "native"},
        program_options={"model": {}},
        document={},
        capability={},
        facts={},
    )
    with scoped_compiler_passes(selected):
        receipt = actual_build_expression(environment)
    assert len(received) == 1
    policy = received[0]["source_optimizations"]
    assert type(policy) is fixtures["SourceOptimizationPolicy"]
    assert policy.permitted_passes == fixtures["PERMISSIONS"]
    assert received[0]["respect_captured_quantization_scope"] is True and received[0]["run"] == "none"
    record = receipt["output"]["source_optimizations"]
    assert record["input"]["sha256"] != digest(capture / "model.mlir")
    assert record["selection"]["request"]["source_sha256"] == record["input"]["sha256"]
    assert owner.calls == ["binding", "policy", "binding"]


def test_missing_original_context_refuses_without_selected_compiler_call(tmp_path):
    fixtures, case = core_fixture(tmp_path)
    owner, selection, package, capture, _ = case
    owner.source_optimization_binding = None
    with scoped_compiler_passes(selection):
        with pytest.raises(ValueError, match="verification context"):
            feedback.compile_model(
                lambda **_kwargs: pytest.fail("selected compiler must not execute"),
                compiler_package=package,
                original_inputs=lambda: {},
                capture=capture,
                package=package,
                output=tmp_path / "out",
            )


@pytest.mark.parametrize(
    "schema", ["SCHEMA", "LIBRARY_SCHEMA", "PASSES_SCHEMA", "CHECKED_SOURCE_SCHEMA", "DELEGATED_PASSES_SCHEMA"]
)
def test_unrelated_child_closure_does_not_read_source_optimization_owners(monkeypatch, schema):
    from merlin_experiments.phase1.feedback import compiler_profile_child as child

    from merlin.targetgen import compiler_source_optimizations as bridge

    monkeypatch.setattr(bridge, "implementation_sources", lambda: pytest.fail("unselected source owners read"))
    document = {"schema": getattr(child, schema), "compiler_passes": {"mechanisms": ["unrelated"]}}
    assert child._selected_closure(document)


@pytest.mark.parametrize("schema", ["PASSES_SCHEMA", "CHECKED_SOURCE_SCHEMA", "DELEGATED_PASSES_SCHEMA"])
def test_selected_child_closure_pins_direct_owners_and_reopens_actual_bytes(tmp_path, monkeypatch, schema):
    from merlin_experiments.phase1.feedback import compiler_profile_child as child

    from merlin.targetgen import compiler_source_optimizations as bridge

    actual_sources = bridge.implementation_sources()
    assert Path(inspect.getsourcefile(bridge)) in actual_sources
    assert any(path.name == "source_optimization_build.py" for path in actual_sources)
    assert any(path.name == "im2col_padding_check.py" for path in actual_sources)
    assert any(path.name == "linalg_mlir.py" for path in actual_sources)
    document = {"schema": getattr(child, schema), "compiler_passes": {"mechanisms": [bridge.MECHANISM]}}
    original = child._selected_closure(document)
    assert {str(path) for path in actual_sources} <= {row["path"] for row in original}
    # This controls the existing child closure checker, not saved-owner restore.
    copied = tmp_path / "copied_owner.py"
    copied.write_bytes(actual_sources[0].read_bytes())
    monkeypatch.setattr(bridge, "implementation_sources", lambda: (copied,))
    before = child._selected_closure(document)
    copied.write_bytes(copied.read_bytes() + b"\n# changed direct owner\n")
    after = child._selected_closure(document)
    assert before != after
    assert before[-1]["path"] == after[-1]["path"] == str(copied)
    assert before[-1]["sha256"] != after[-1]["sha256"]


def test_extracted_exact_source_group_wrapper_preserves_original_call_and_alias(monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import private_source_demand

    from merlin.llvmlower import source_compute_groups

    calls = []
    sentinel = ([], 3)

    def exact(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(source_compute_groups, "form_exact_groups", exact)
    module, form = object(), object()
    assert full_models._exact_source_groups is private_source_demand.exact_source_groups
    assert full_models._exact_source_groups(module, "neutral", tmp_path, form) is sentinel
    assert calls == [(module, "neutral", tmp_path, form)]
