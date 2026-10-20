"""Pure observed transport fixtures: no compiler, linker, target or timer qualification."""

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_recomputation import budgets, permissions, source
from test_source_recomputation_selection import synthetic_comparison

from merlin.common.ir_audit import IrAudit
from merlin.compile import baremetal_model as BM
from merlin.llvmlower import compilation_recipe as C
from merlin.llvmlower import lower as L
from merlin.llvmlower import source_recomputation as S
from merlin.llvmlower.source_recomputation_policy import SourceRecomputationPolicy
from merlin.runtime import source_recomputation_build as B
from merlin.transforms.certificates import ir_text
from merlin.transforms.pipeline_state import PipelineContext


def current_owner(**kwargs):
    assert kwargs["settings"]["index_bits"] == 64


def final_owner(**kwargs):
    assert kwargs["stage"] in ("linked", "executed")
    assert kwargs["products"]


def simulated_runner(argv, **kwargs):
    """Write synthetic products without executing argv; test observation only."""
    Path(argv[-1]).write_bytes(b"synthetic product " + Path(argv[-1]).name.encode())
    return subprocess.CompletedProcess(list(map(str, argv)), 0, b"", b"")


def setup(tmp_path, *, comparison=None, verifier=final_owner, text=None):
    capture = tmp_path / "capture"
    capture.mkdir()
    text = ir_text(source()) if text is None else text
    (capture / "model.mlir").write_text(text)
    policy = SourceRecomputationPolicy(
        PipelineContext("hardware", "numeric", "resources", "runtime", "library"),
        permissions(),
        budgets(),
        verify_current=current_owner,
        comparison_for=comparison,
    )
    build = B.SourceRecomputationBuild(policy, verify_final=verifier)
    build.capture(capture, {"run": "none", "scope": "synthetic transport only"})
    work = tmp_path / "work"
    work.mkdir()
    recipe = C.CompilationRecipe(work, producer=Path(__file__).resolve())
    build.bind_sources(recipe)
    prepared = work / "model.prepared.mlir"
    prepared.write_text(text)
    lower = work / "lower"
    lower.mkdir()
    build.prepare(prepared, lower, {"targets": (), "index_bits": 64}, {"compiler": "synthetic"})
    return build, recipe, prepared, lower


def lowered(build, prepared, lower):
    options = build.options()
    text = prepared.read_text()
    selection = options.get("source_recomputation")
    evaluation = None
    # Use actual effective settings identical to prepare(), without executing lowering.
    arguments = inspect.signature(L.lower_model_file).bind(prepared, lower, targets=(), index_bits=64)
    arguments.apply_defaults()
    if selection:
        evaluation = S.SourceRecomputationPreparation(text, selection, S.lowering_settings(arguments.arguments))
        evaluation.retain(lower)
    llvm = lower / "model.ll"
    llvm.write_text("; synthetic LLVM transport bytes\n")
    with IrAudit(lower, enabled=True, producer="synthetic transport fixture", source=Path(__file__)) as audit:
        audit.stage("input", text)
        if evaluation and evaluation.report["selected"]:
            audit.stage("source-recomputed", evaluation.selected_source)
        audit.stage("llvm-final", llvm.read_text(), format="llvm-ir")
    return L.LowerResult(
        lower,
        llvm,
        stats={"source_recomputation": evaluation.report} if evaluation else {},
        audit_index=audit.directory / "index.json",
    )


def observed(build, recipe, result, *, link_inputs=None, before_finish=None, after_consumed=None):
    build.consumed(result, recipe)
    if after_consumed:
        after_consumed()
    tool = Path(sys.executable).resolve()
    obj, elf = recipe.path.parent / "model.o", recipe.path.parent / "model.elf"
    build.observer(recipe, "model_object")(
        [tool, "-c", result.ll_path, "-o", obj],
        runner=simulated_runner,
        inputs=[result.ll_path],
        output=obj,
    )
    inputs = [obj] if link_inputs is None else link_inputs
    build.observer(recipe, "final_link")(
        [tool, *inputs, "-o", elf],
        runner=simulated_runner,
        inputs=inputs,
        output=elf,
    )
    recipe.completed(elf)
    if before_finish:
        before_finish()
    build.finish(recipe, elf)
    return elf


def chosen(*, plan, settings):
    return synthetic_comparison(plan)


@pytest.mark.parametrize("comparison,status,selected", [(None, "MISSING_COST", False), (chosen, "CANDIDATE", True)])
def test_genuine_bound_plan_observed_products_and_final_owner(tmp_path, comparison, status, selected):
    calls = []

    def verify(**kwargs):
        calls.append(kwargs["stage"])
        final_owner(**kwargs)

    build, recipe, prepared, lower = setup(tmp_path, comparison=comparison, verifier=verify)
    result = lowered(build, prepared, lower)
    original = prepared.read_bytes()
    elf = observed(build, recipe, result)
    record = build.record()
    assert calls == ["linked"]
    assert record["selection"]["selected"] is selected
    if not selected:
        assert record["selection"]["status"] == status
        assert (lower / "source-recomputation-selected.mlir").read_bytes() == original
    assert record["executable"] == B._pin(elf)
    assert prepared.read_bytes() == original
    assert len(list(recipe.path.parent.glob("invocations/*/invocation.json"))) == 2
    assert any(row["path"] == str(result.ll_path) for row in record["products"])
    assert len(recipe.record["preparation"]) == len({row["path"] for row in recipe.record["preparation"].values()})
    run = lower / "console.bin"
    run.write_bytes(b"synthetic execution record, no process executed")
    build.executed({"console": B._pin(run)})
    assert calls == ["linked", "executed"]
    assert B._pin(run) in build.record()["products"]
    with pytest.raises(ValueError, match="once"):
        build.executed({})


def test_unsupported_source_records_no_plan_exact_original(tmp_path):
    build, recipe, prepared, lower = setup(tmp_path, text="module {}")
    assert build.options() == {"ir_audit": True}
    result = lowered(build, prepared, lower)
    observed(build, recipe, result)
    assert build.record()["selection"]["status"] == "NO_PLAN"


@pytest.mark.parametrize(
    "mutation", ["input_hash", "input_file", "duplicate", "missing_selected", "ignored", "llvm", "report"]
)
def test_consumption_rejects_forged_or_ignored_exact_source(tmp_path, mutation):
    build, recipe, prepared, lower = setup(tmp_path, comparison=chosen)
    result = lowered(build, prepared, lower)
    document = json.loads(result.audit_index.read_text())
    if mutation == "input_hash":
        document["stages"][0]["sha256"] = "0" * 64
    elif mutation == "input_file":
        (result.audit_index.parent / document["stages"][0]["file"]).write_text("module {}")
    elif mutation == "duplicate":
        document["stages"].append(document["stages"][1])
    elif mutation == "missing_selected":
        document["stages"] = [row for row in document["stages"] if row["name"] != "source-recomputed"]
    elif mutation == "ignored":
        path = result.audit_index.parent / document["stages"][1]["file"]
        path.write_text(prepared.read_text())
        document["stages"][1].update(sha256=B._pin(path)["sha256"], bytes=path.stat().st_size)
    elif mutation == "llvm":
        result.ll_path.write_text("changed")
    else:
        result.stats["source_recomputation"]["selected"] = False
    result.audit_index.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        build.consumed(result, recipe)


@pytest.mark.parametrize("product", ["model.o", "model.elf", "lower/model.ll", "compilation_recipe.json"])
def test_final_owner_cannot_rewrite_linked_products(tmp_path, product):
    def verify(**kwargs):
        (tmp_path / "work" / product).write_bytes(b"changed by original callback")

    build, recipe, prepared, lower = setup(tmp_path, verifier=verify)
    with pytest.raises(ValueError, match="changed"):
        observed(build, recipe, lowered(build, prepared, lower))


def test_original_owner_must_accept_final_image_cost_transfer(tmp_path):
    def reject(**kwargs):
        raise ValueError("third image cost transfer is unqualified")

    build, recipe, prepared, lower = setup(tmp_path, comparison=chosen, verifier=reject)
    with pytest.raises(ValueError, match="third image"):
        observed(build, recipe, lowered(build, prepared, lower))
    with pytest.raises(ValueError, match="previously failed"):
        build.record()


def test_link_must_consume_the_actual_observed_model_object(tmp_path):
    build, recipe, prepared, lower = setup(tmp_path)
    foreign = recipe.path.parent / "foreign.o"
    foreign.write_bytes(b"foreign object")
    with pytest.raises(ValueError, match="model object"):
        observed(build, recipe, lowered(build, prepared, lower), link_inputs=[foreign])


def test_model_object_must_consume_exact_observed_llvm(tmp_path):
    build, recipe, prepared, lower = setup(tmp_path)
    build.consumed(lowered(build, prepared, lower), recipe)
    foreign = lower / "foreign.ll"
    foreign.write_bytes(b"foreign")
    with pytest.raises(ValueError, match="LLVM"):
        build.observer(recipe, "model_object")(
            [sys.executable, "-c", foreign, "-o", lower / "bad.o"],
            runner=simulated_runner,
            inputs=[foreign],
            output=lower / "bad.o",
        )
    assert not recipe.record["commands"]


@pytest.mark.parametrize("which", ["materialized", "prepared", "defaults", "method", "policy"])
def test_original_inputs_methods_and_signature_metadata_do_not_refresh(tmp_path, monkeypatch, which):
    build, recipe, prepared, lower = setup(tmp_path)
    if which == "materialized":
        (tmp_path / "capture/model.mlir").write_text("module {}")
    elif which == "prepared":
        prepared.write_text("module {}")
    elif which == "defaults":
        raw = inspect.unwrap(L.lower_model)
        monkeypatch.setattr(raw, "__defaults__", (("foreign",), *raw.__defaults__[1:]))
    elif which == "method":
        monkeypatch.setattr(B.SourceRecomputationBuild, "finish", lambda *args: None)
    else:
        object.__setattr__(build.policy.context, "execution", "different")
    with pytest.raises(ValueError):
        build.options()


def test_symlink_retargeting_cannot_replace_original_products(tmp_path):
    build, recipe, prepared, lower = setup(tmp_path)
    observed(build, recipe, lowered(build, prepared, lower))
    copy = lower / "original.ll"
    (lower / "model.ll").rename(copy)
    (lower / "model.ll").symlink_to(copy)
    with pytest.raises(ValueError, match="canonical"):
        build.record()


def test_absent_outer_selection_adds_no_kwargs_record_or_leaf_import(tmp_path, monkeypatch):
    from test_baremetal_saved_model import _fake_build, _fixture

    inputs, out = _fixture(tmp_path, monkeypatch)
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs)
        return _fake_build(*args, **kwargs)

    monkeypatch.setattr(BM.spike_model, "build", build)
    # Removing the already-imported leaf also checks that the default does not reload it.
    monkeypatch.delitem(sys.modules, B.__name__)
    receipt = BM.compile_saved_model(**inputs, output=out / "default", run="none")
    assert "source_recomputation" not in calls[0]
    assert "source_recomputation" not in receipt["output"]
    assert B.__name__ not in sys.modules


@pytest.mark.parametrize("field", ["context", "selection", "products"])
def test_final_owner_cannot_mutate_detached_qualification_inputs(tmp_path, field):
    def verify(**kwargs):
        if field == "products":
            kwargs[field].clear()
        else:
            kwargs[field]["foreign"] = True

    build, recipe, prepared, lower = setup(tmp_path, verifier=verify)
    with pytest.raises(ValueError, match="verification changed"):
        observed(build, recipe, lowered(build, prepared, lower))


def test_reentrant_final_owner_poison_is_not_silently_caught(tmp_path):
    holder = {}

    def verify(**kwargs):
        with pytest.raises(ValueError, match="not reentrant"):
            holder["build"].executed({})

    build, recipe, prepared, lower = setup(tmp_path, verifier=verify)
    holder["build"] = build
    with pytest.raises(ValueError, match="previously failed"):
        observed(build, recipe, lowered(build, prepared, lower))


def test_complete_original_recipe_inputs_survive_late_owner_callbacks(tmp_path):
    generated = tmp_path / "generated.c"
    generated.write_text("synthetic runtime source")

    def verify(**kwargs):
        generated.write_text("changed by final verifier")

    build, recipe, prepared, lower = setup(tmp_path, verifier=verify)
    result = lowered(build, prepared, lower)
    # Runtime source is an earlier ordinary recipe command, not a model/link observation input.
    runtime = recipe.path.parent / "runtime.o"

    def runtime_command():
        recipe.run(
            [Path(sys.executable).resolve(), "-c", generated, "-o", runtime],
            runner=simulated_runner,
            inputs=[generated],
            output=runtime,
        )

    with pytest.raises(ValueError, match="changed"):
        observed(build, recipe, result, after_consumed=runtime_command)


def test_execution_products_are_pinned_before_later_verification(tmp_path):
    build, recipe, prepared, lower = setup(tmp_path)
    observed(build, recipe, lowered(build, prepared, lower))
    output = lower / "outputs.npz"
    output.write_bytes(b"synthetic readback")
    build.retain_execution([output])
    output.write_bytes(b"changed after numeric comparison")
    with pytest.raises(ValueError, match="changed"):
        build.executed({"output": str(output)})


@pytest.mark.parametrize("readback_mutation", [False, True])
def test_outer_forwards_the_same_live_custodian_to_the_original_inner_build(tmp_path, monkeypatch, readback_mutation):
    from test_baremetal_saved_model import _console, _fake_build, _fixture

    inputs, out = _fixture(tmp_path, monkeypatch)
    armed = {"value": False}
    if readback_mutation:
        model = inputs["capture"] / "model.mlir"
        model.write_text(ir_text(source()))
        receipt_path = inputs["capture"] / "capture_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["artifacts"]["model.mlir"] = {"bytes": model.stat().st_size, "sha256": B._pin(model)["sha256"]}
        receipt_path.write_text(json.dumps(receipt))
        original_judge = BM._judge_prefix

        def judge(*args, **kwargs):
            original_judge(*args, **kwargs)
            armed["value"] = True

        monkeypatch.setattr(BM, "_judge_prefix", judge)
        monkeypatch.setattr(BM.spike_backend, "available", lambda: True)
        monkeypatch.setattr(BM, "_spike_extension_backend", lambda target: None)
        monkeypatch.setattr(BM.spike_model, "run", lambda *args, **kwargs: {"console": _console()})

    def change():
        if armed["value"]:
            (out / "selected" / "console.log").write_text("changed after judgment before first snapshot")

    def comparison(*, plan, settings):
        return synthetic_comparison(plan, state={"callback": change})

    seen = []

    def build(bundle, work, **kwargs):
        custodian = kwargs["source_recomputation"]
        seen.append(custodian)
        base = _fake_build(bundle, work, **kwargs)
        custodian.capture(bundle)
        recipe = C.CompilationRecipe(work, producer=Path(__file__).resolve())
        custodian.bind_sources(recipe)
        prepared = work / "prepared.mlir"
        prepared.write_text((bundle / "model.mlir").read_text())
        lower = work / "lower"
        lower.mkdir()
        custodian.prepare(prepared, lower, {"targets": (), "index_bits": 64}, {"test": "synthetic"})
        base["elf"] = observed(custodian, recipe, lowered(custodian, prepared, lower))
        base["source_recomputation"] = custodian.record()
        return base

    monkeypatch.setattr(BM.spike_model, "build", build)
    policy = SourceRecomputationPolicy(
        PipelineContext("hardware", "numeric", "resources", "runtime", "library"),
        permissions(),
        budgets(),
        verify_current=current_owner,
        comparison_for=comparison if readback_mutation else None,
    )
    selected = B.SourceRecomputationBuild(policy, verify_final=final_owner)
    options = dict(
        inputs,
        output=out / "selected",
        run="spike" if readback_mutation else "none",
        source_recomputation=selected,
        reference_file="golden.npy",
    )
    if readback_mutation:
        with pytest.raises(ValueError, match="changed"):
            BM.compile_saved_model(**options)
        return
    receipt = BM.compile_saved_model(**options)
    assert seen == [selected]
    assert receipt["status"] == "compiled"
    assert receipt["output"]["source_recomputation"] == selected.record()


def test_returned_lower_result_is_frozen_before_repeated_cost_callbacks(tmp_path):
    holder = {}

    def change():
        if "result" in holder:
            holder["result"].stats["late"] = "changed before first product snapshot"

    def comparison(*, plan, settings):
        return synthetic_comparison(plan, state={"callback": change})

    build, recipe, prepared, lower = setup(tmp_path, comparison=comparison)
    result = lowered(build, prepared, lower)
    holder["result"] = result
    with pytest.raises(ValueError, match="result|changed"):
        build.consumed(result, recipe)


@pytest.mark.parametrize("method", ["run", "_run", "bind_preparation", "completed"])
def test_recipe_instance_cannot_replace_original_class_methods(tmp_path, monkeypatch, method):
    build, recipe, prepared, lower = setup(tmp_path)
    monkeypatch.setattr(recipe, method, lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="callback changed|implementation"):
        build.options()


def test_original_runner_default_metadata_is_rechecked_after_observation(tmp_path, monkeypatch):
    def runner(argv, **kwargs):
        monkeypatch.setattr(runner, "__kwdefaults__", {"late": 1})
        return simulated_runner(argv, **kwargs)

    build, recipe, prepared, lower = setup(tmp_path)
    result = lowered(build, prepared, lower)
    build.consumed(result, recipe)
    obj = recipe.path.parent / "model.o"
    with pytest.raises(ValueError, match="runner defaults"):
        build.observer(recipe, "model_object")(
            [Path(sys.executable).resolve(), "-c", result.ll_path, "-o", obj],
            runner=runner,
            inputs=[result.ll_path],
            output=obj,
        )


@pytest.mark.parametrize("which", ["final", "current", "resolver"])
def test_original_owner_callback_defaults_are_frozen_before_first_call(tmp_path, monkeypatch, which):
    def verify(**kwargs):
        return None

    def resolver(*, plan, settings):
        return None

    build, recipe, prepared, lower = setup(tmp_path, verifier=verify, comparison=resolver)
    callback = {"final": verify, "current": current_owner, "resolver": resolver}[which]
    monkeypatch.setattr(callback, "__kwdefaults__", {"validation_disabled": True})
    with pytest.raises(ValueError, match="defaults changed"):
        build.options()
