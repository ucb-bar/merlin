"""Original source/cost transport; synthetic terms qualify no actual timer."""

import importlib.util
from hashlib import sha256

import pytest
from test_recomputation import budgets, permissions, source

from merlin.llvmlower.lower import LowerResult
from merlin.llvmlower.source_recomputation import (
    SourceRecomputationPreparation,
    SourceRecomputationSelection,
)
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
)
from merlin.perf.offload_profitability import compare_complete_offload_cost
from merlin.perf.term import Bounds, PerformanceTerm, Provenance, Validity
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext
from merlin.transforms.recomputation import plan_source_recomputation


def digest(value):
    return sha256(value.encode()).hexdigest()


def plan():
    return plan_source_recomputation(
        source(),
        PipelineContext("hardware", "numeric", "resources", "runtime", "library"),
        permissions(),
        budgets(),
    )


def original_owner(**kwargs):
    assert kwargs["stage"] in ("entry", "return")
    assert kwargs["settings"]["index_bits"] == 64
    assert kwargs["settings"]["runtime"] == "original-runtime"


def settings():
    return {"index_bits": 64, "runtime": "original-runtime", "targets": ["host"]}


def synthetic_comparison(proposal, *, bounds=(70, 80), state=None):
    """Exercise the real complete-cost gate, with explicitly synthetic evidence."""
    state = {} if state is None else state
    scope = ComponentCostScope(digest("test-timer"), digest("test-numeric"), digest("test-inputs"))

    def observation(arm):
        executable = digest(arm + "-synthetic-executable")
        return ComponentFeatureObservation(
            digest(arm + "-compiler"),
            digest("member"),
            digest("corpus"),
            digest("target"),
            scope.sha256,
            digest(arm + "-joint"),
            executable,
            digest(arm + "-dependencies"),
            digest("inputs"),
            (digest(arm + "-evidence"),),
            (ComponentCostRegion("whole", COMPLETE_STAGES, ("synthetic",), {"count": 1}),),
            (ComponentCostRegion("whole", COMPLETE_STAGES, ("synthetic",), {"count": 1}),),
            "PASS",
            artifact_files=((arm + ".synthetic", executable),),
            legality_status="PASS",
        )

    def term(low, high):
        return PerformanceTerm(
            "synthetic-whole",
            (low + high) / 2,
            "ns",
            Provenance("measured", (digest("synthetic-only"),)),
            Validity("unit transport test only; no runtime/timer qualification"),
            Bounds(low, high),
        )

    def verifier(**kwargs):
        assert kwargs["stamp"] == proposal.stamp
        assert kwargs["original_selection_sha256"] == proposal.baseline_selection_sha256
        assert kwargs["candidate_selection_sha256"] == proposal.candidate_selection_sha256
        if state.get("changed"):
            raise ValueError("original synthetic owner refuses stale runtime")
        if callback := state.get("callback"):
            callback()

    return compare_complete_offload_cost(
        original=observation("baseline"),
        candidate=observation("candidate"),
        original_costs={name: term(100, 120) for name in ("cold", "warm")},
        candidate_costs={name: term(*bounds) for name in ("cold", "warm")},
        original_selection_sha256=proposal.baseline_selection_sha256,
        candidate_selection_sha256=proposal.candidate_selection_sha256,
        scope=scope,
        regime="warm",
        stamp=proposal.stamp,
        verify_original=verifier,
    )


def test_late_completion_closure_replacement_refuses():
    proposal = plan()
    preparation = SourceRecomputationPreparation(
        ir_text(proposal.source), SourceRecomputationSelection(proposal, verify_current=original_owner), settings()
    )
    preparation._check = lambda **_: None
    with pytest.raises(ValueError, match="original|issued|check"):
        preparation.revalidate()


@pytest.mark.parametrize("mutation", ["identity", "code"])
def test_last_original_callback_cannot_retarget_completion_guard(tmp_path, mutation):
    proposal, holder = plan(), {}

    def owner(**kwargs):
        if kwargs["stage"] != "return":
            return
        preparation = holder["preparation"]
        if mutation == "identity":
            preparation._check = lambda **_: None
        else:
            preparation._check.__code__ = preparation._check.__code__.replace(co_name="changed_completion")

    prepared = SourceRecomputationPreparation(
        ir_text(proposal.source), SourceRecomputationSelection(proposal, verify_current=owner), settings()
    )
    holder["preparation"] = prepared
    path = tmp_path / "actual.ll"
    path.write_text("actual original product")
    with pytest.raises(ValueError, match="guard|changed|original"):
        prepared.revalidate(result=LowerResult(tmp_path, path))


def test_cost_callback_cannot_rewrite_final_product_before_snapshot(tmp_path):
    proposal, state = plan(), {}
    comparison = synthetic_comparison(proposal, state=state)
    preparation = SourceRecomputationPreparation(
        ir_text(proposal.source),
        SourceRecomputationSelection(proposal, verify_current=original_owner, comparison=comparison),
        settings(),
    )
    path = tmp_path / "actual.ll"
    path.write_bytes(b"original compiler output")
    result = LowerResult(tmp_path, path)
    state["callback"] = lambda: path.write_bytes(b"substituted compiler data")
    with pytest.raises(ValueError, match="product|output|changed"):
        preparation.revalidate(result=result)


@pytest.mark.parametrize(
    "bounds,status", [(None, "MISSING_COST"), ((90, 110), "TIE_OR_OVERLAP"), ((130, 150), "REGRESSION")]
)
def test_inconclusive_cost_keeps_exact_original_text(bounds, status):
    proposal = plan()
    original = "\n// retain exact original bytes\n" + ir_text(proposal.source) + "\n"
    comparison = None if bounds is None else synthetic_comparison(proposal, bounds=bounds)
    prepared = SourceRecomputationPreparation(
        original,
        SourceRecomputationSelection(proposal, verify_current=original_owner, comparison=comparison),
        settings(),
    )
    assert prepared.source == prepared.selected_source == original
    assert prepared.report["selected"] is False and prepared.report["status"] == status


def test_complete_opportunity_selects_actual_checked_candidate():
    proposal = plan()
    prepared = SourceRecomputationPreparation(
        ir_text(proposal.source),
        SourceRecomputationSelection(
            proposal, verify_current=original_owner, comparison=synthetic_comparison(proposal)
        ),
        settings(),
    )
    assert prepared.selected_source == ir_text(proposal.candidate)
    assert prepared.selected_source != prepared.source and prepared.report["selected"] is True
    assert (
        prepared.report["plan"]["conformance"]["after_scalar_evaluations"]
        > prepared.report["plan"]["conformance"]["before_scalar_evaluations"]
    )


def test_unknown_context_cost_retains_original_without_invented_timing():
    proposal = plan_source_recomputation(source(), PipelineContext(), permissions(), budgets())
    original = ir_text(proposal.source)
    prepared = SourceRecomputationPreparation(
        original,
        SourceRecomputationSelection(
            proposal, verify_current=original_owner, comparison=synthetic_comparison(proposal)
        ),
        settings(),
    )
    assert prepared.selected_source == original
    assert prepared.report["status"] == "UNRESOLVED"
    assert prepared.report["comparison"]["missing"]


def test_foreign_actual_source_refuses_before_original_verifier():
    proposal, called = plan(), []

    def owner(**kwargs):
        called.append(kwargs)

    with pytest.raises(ValueError, match="actual ordinary input"):
        SourceRecomputationPreparation(
            ir_text(source(width=16)), SourceRecomputationSelection(proposal, verify_current=owner), settings()
        )
    assert not called


def test_foreign_cost_alternative_or_stamp_refuses_before_original_verifier():
    proposal = plan()
    foreign = plan_source_recomputation(
        source(width=16),
        PipelineContext("hardware", "numeric", "resources", "runtime", "library"),
        permissions(),
        budgets(),
    )
    called = []

    def owner(**kwargs):
        called.append(kwargs)

    with pytest.raises(ValueError, match="costs differ"):
        SourceRecomputationPreparation(
            ir_text(proposal.source),
            SourceRecomputationSelection(proposal, verify_current=owner, comparison=synthetic_comparison(foreign)),
            settings(),
        )
    assert not called


@pytest.mark.parametrize("key,value", [("index_bits", 32), ("runtime", "foreign-runtime")])
def test_actual_abi_and_runtime_are_original_owner_obligations(key, value):
    proposal, actual = plan(), settings()
    actual[key] = value
    with pytest.raises(AssertionError):
        SourceRecomputationPreparation(
            ir_text(proposal.source), SourceRecomputationSelection(proposal, verify_current=original_owner), actual
        )


@pytest.mark.parametrize(
    "mutation", ["settings", "source", "candidate", "context", "owner", "plan_record", "comparison_record"]
)
def test_first_original_callback_cannot_change_original_bindings(mutation):
    proposal, actual, holder = plan(), settings(), {}
    comparison = synthetic_comparison(proposal)

    def owner(**kwargs):
        if mutation == "settings":
            actual["runtime"] = "changed"
        elif mutation in ("source", "candidate"):
            module = getattr(proposal, mutation)
            function = module.body.block.first_op
            module.body.block.insert_op_before(function.clone(), function)
            module.body.block.erase_op(function)
        elif mutation == "context":
            object.__setattr__(proposal.stamp.context, "execution", "changed")
        elif mutation == "owner":
            object.__setattr__(holder["selection"], "verify_current", original_owner)
        elif mutation == "plan_record":
            kwargs["plan"]["cost"] = "forged"
        else:
            kwargs["comparison"]["choose_candidate"] = False

    selected = SourceRecomputationSelection(proposal, verify_current=owner, comparison=comparison)
    holder["selection"] = selected
    with pytest.raises(ValueError, match="changed|replan"):
        SourceRecomputationPreparation(ir_text(proposal.source), selected, actual)


def test_stale_complete_cost_owner_refuses_on_later_revalidation():
    proposal, state = plan(), {}
    prepared = SourceRecomputationPreparation(
        ir_text(proposal.source),
        SourceRecomputationSelection(
            proposal, verify_current=original_owner, comparison=synthetic_comparison(proposal, state=state)
        ),
        settings(),
    )
    state["changed"] = True
    with pytest.raises(ValueError, match="stale runtime"):
        prepared.revalidate()


def test_non_none_original_verifier_response_refuses():
    proposal = plan()

    def invalid_owner(**kwargs):
        return {"approved": True}

    with pytest.raises(ValueError, match="return None"):
        SourceRecomputationPreparation(
            ir_text(proposal.source), SourceRecomputationSelection(proposal, verify_current=invalid_owner), settings()
        )


@pytest.mark.parametrize("mutation", ["llvm", "stats", "path", "recipe", "retained", "symlink"])
def test_final_original_owner_cannot_change_products_or_result(tmp_path, mutation):
    proposal, holder = plan(), {}
    output = tmp_path / "actual.ll"
    output.write_bytes(b"original output")
    alternate = tmp_path / "alternate.ll"
    alternate.write_bytes(output.read_bytes())
    recipe = tmp_path / "lowering_recipe.json"
    recipe.write_text('{"original":true}')
    result = LowerResult(tmp_path, output, {"original": True})

    def owner(**kwargs):
        if kwargs["stage"] != "return":
            return
        if mutation == "llvm":
            output.write_bytes(b"changed output")
        elif mutation == "stats":
            result.stats["original"] = False
        elif mutation == "path":
            result.ll_path = alternate
        elif mutation == "recipe":
            recipe.write_text('{"changed":true}')
        elif mutation == "retained":
            (tmp_path / "source-recomputation-input.mlir").write_text("changed")
        else:
            output.unlink()
            output.symlink_to(alternate)

    prepared = SourceRecomputationPreparation(
        ir_text(proposal.source), SourceRecomputationSelection(proposal, verify_current=owner), settings()
    )
    holder["preparation"] = prepared
    prepared.retain(tmp_path)
    with pytest.raises(ValueError, match="changed|linked|symlink"):
        prepared.revalidate(result=result)


@pytest.mark.parametrize("mutation", ["code", "file"])
def test_original_callback_implementation_and_source_remain_pinned(tmp_path, mutation):
    proposal = plan()
    path = tmp_path / "original_owner.py"
    path.write_text("def owner(**kwargs):\n    return None\n")
    spec = importlib.util.spec_from_file_location("original_recomputation_test_owner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prepared = SourceRecomputationPreparation(
        ir_text(proposal.source), SourceRecomputationSelection(proposal, verify_current=module.owner), settings()
    )
    if mutation == "code":
        module.owner.__code__ = original_owner.__code__
    else:
        path.write_text(path.read_text() + "# owner bytes changed\n")
    with pytest.raises(ValueError, match="changed"):
        prepared.revalidate()


@pytest.mark.parametrize("selected", [False, True])
def test_ordinary_file_caller_only_forwards_explicit_selection(tmp_path, monkeypatch, selected):
    from merlin.llvmlower import lower

    proposal, calls, sentinel = plan(), [], object()
    selection = SourceRecomputationSelection(proposal, verify_current=original_owner) if selected else None
    path = tmp_path / "source.mlir"
    path.write_text(ir_text(proposal.source))

    def original_lower(text, work, targets, **kwargs):
        calls.append(kwargs)
        assert text == path.read_text()
        return sentinel

    monkeypatch.setattr(lower, "lower_model", original_lower)
    assert lower.lower_model_file(path, tmp_path / "out", source_recomputation=selection) is sentinel
    assert ("source_recomputation" in calls[0]) is selected
    if selected:
        assert calls[0]["source_recomputation"] is selection


@pytest.mark.parametrize("indirect", [False, True])
def test_reused_unselected_workdir_cannot_retain_a_previous_recomputation_verdict(tmp_path, monkeypatch, indirect):
    from merlin.llvmlower import lower

    work = tmp_path / "reused"
    work.mkdir()
    original = tmp_path / "original-evidence"
    original.write_text("previous selection")
    paths = tuple(
        work / name
        for name in (
            "source-recomputation-input.mlir",
            "source-recomputation-selected.mlir",
            "source-recomputation.json",
        )
    )
    for path in paths:
        if indirect:
            path.symlink_to(original)
        else:
            path.write_text(original.read_text())

    class ReachedPreprocessing(Exception):
        pass

    def observe_current_input(text, *args, **kwargs):
        assert text == "module {}"
        assert all(not path.exists() and not path.is_symlink() for path in paths)
        assert original.read_text() == "previous selection"
        raise ReachedPreprocessing

    monkeypatch.setattr(lower, "preprocess_text", observe_current_input)
    with pytest.raises(ReachedPreprocessing):
        lower.lower_model("module {}", work, targets=(), vectorize=False, parallel=False)


@pytest.mark.parametrize(
    "options",
    [
        {"graph_optimizations": object()},
        {"source_optimizations": object()},
        {"textual": True},
    ],
)
def test_unqualified_mixed_source_routes_refuse_before_lowering(tmp_path, options):
    from merlin.llvmlower.lower import lower_model

    proposal, called = plan(), []

    def owner(**kwargs):
        called.append(kwargs)

    output = tmp_path / "must-not-create"
    with pytest.raises(ValueError, match="source route|ordering"):
        lower_model(
            ir_text(proposal.source),
            output,
            targets=(),
            source_recomputation=SourceRecomputationSelection(proposal, verify_current=owner),
            **options,
        )
    assert not output.exists() and not called


@pytest.mark.parametrize("mutation", ["plan", "reader", "owner_file"])
def test_earlier_original_borrowed_validator_cannot_change_recomputation(tmp_path, monkeypatch, mutation):
    from merlin.llvmlower import lower, source_recomputation
    from merlin.llvmlower.borrowed_buffer_calls import BorrowedBufferCalls

    proposal, called = plan(), []
    text = ir_text(proposal.source)
    path = tmp_path / "original.mlir"
    path.write_text(text)
    owner_path = tmp_path / "current_owner.py"
    owner_path.write_text("def owner(**kwargs):\n    return None\n")
    spec = importlib.util.spec_from_file_location("borrowed_recomputation_original_owner", owner_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_reader = source_recomputation._functions

    def borrowed_owner():
        called.append("borrowed")
        if mutation == "plan":
            candidate = proposal.candidate
            function = candidate.body.block.first_op
            candidate.body.block.insert_op_before(function.clone(), function)
            candidate.body.block.erase_op(function)
        elif mutation == "reader":
            monkeypatch.setattr(source_recomputation, "_functions", lambda: original_reader())
        else:
            owner_path.write_text(owner_path.read_text() + "# changed original callback owner\n")

    # A genuine empty original roster exercises its validation boundary without
    # asserting any opaque call effects or running a native lifetime proof.
    borrowed = BorrowedBufferCalls(path, digest(text), digest(text), (), (), borrowed_owner)
    selected = SourceRecomputationSelection(proposal, verify_current=module.owner)
    with pytest.raises(ValueError, match="changed|SSA|candidate"):
        lower.lower_model(
            text, tmp_path / "must-not-lower", targets=(), borrowed_buffer_calls=borrowed, source_recomputation=selected
        )
    assert called == ["borrowed"]
