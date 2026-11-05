"""Shared source transaction boundaries, without compilation or qualification.

Dummy objects exercise custody/aggregation only. They are deliberately not ELF
files, and never enter the real host-provider object/link admission gate.
"""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_pointwise_source import source_program
from test_source_pointwise_lowering import fixture as pointwise_selection
from test_source_row_lowering import selection as row_selection
from xdsl.dialects import builtin, func

from merlin.common.digest import sha256_file
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.source_model import bind_source_lowerings, source_device_work, source_provider_builder
from merlin.runtime.host_provider import HostProviderBuild, HostProviderContext, HostProviderObject
from merlin.transforms.certificates import ir_text


def witness(family, label):
    return hashlib.sha256(("unit transport only:" + family + ":" + label).encode()).hexdigest()


def transaction(tmp_path, *, changes=None, mutate=None, order=("row", "pointwise")):
    directories = {name: tmp_path / name for name in ("row", "pointwise")}
    for directory in directories.values():
        directory.mkdir()
    row, row_module, row_seen = row_selection(directories["row"], **(changes or {}).get("row", {}))
    pointwise, _, pointwise_seen = pointwise_selection(directories["pointwise"], **(changes or {}).get("pointwise", {}))
    pointwise_module, _ = source_program()
    functions = []
    for name, module in (("row", row_module), ("pointwise", pointwise_module)):
        function = next(op for op in module.body.block.ops if isinstance(op, func.FuncOp)).clone()
        function.properties["sym_name"] = builtin.StringAttr(name + "_entry")
        functions.append(function)
    module = builtin.ModuleOp(functions)
    module.verify()
    events = []

    def provider(family):
        def build(bindings, context):
            assert len(bindings) == 1
            events.append(family)
            obj, llvm = context.workdir / "dummy.o", context.workdir / "dummy.ll"
            obj.write_bytes((family + " unit-only object, never executed").encode())
            llvm.write_text(family + " unit-only companion, not parsed as LLVM")
            artifact = HostProviderObject(obj, sha256_file(obj), llvm, sha256_file(llvm), (), (), (), (), tmp_path)
            result = HostProviderBuild(
                sha256_file(context.prepared_path),
                sha256_file(context.model_llvm_path),
                sha256_file(context.model_object_path),
                (artifact,),
                witness(family, "numerical"),
                witness(family, "effects"),
            )
            return mutate(family, context, result) if mutate is not None else result

        return build

    configurations = dict(
        row=replace(row, build=provider("row")),
        pointwise=replace(pointwise, budgets=replace(pointwise.budgets), build=provider("pointwise")),
    )
    prepare, builder = bind_source_lowerings(tuple(configurations[name] for name in order), None, None, None)
    source = tmp_path / "model.mlir"
    source.write_text(ir_text(module))
    return SimpleNamespace(
        source=source,
        configurations=configurations,
        prepare=prepare,
        builder=builder,
        seen=dict(row=row_seen, pointwise=pointwise_seen),
        events=events,
        root=tmp_path,
    )


def prepare_and_context(plan):
    selected = apply_prepared_model_transform(plan.source, plan.root / "preparation", plan.prepare)
    work = plan.root / "host_provider"
    work.mkdir()
    llvm, obj = plan.root / "model.ll", plan.root / "model.o"
    llvm.write_text("immutable unit-only model LLVM")
    obj.write_bytes(b"immutable unit-only model object")

    def forbid_compile(*args, **kwargs):
        pytest.fail("transport test must never compile or execute provider code")

    return HostProviderContext(selected, llvm, obj, work, plan.root / "unused-compiler", (), forbid_compile)


@pytest.mark.parametrize("order", [("row", "pointwise"), ("pointwise", "row")])
def test_aggregate_keeps_individual_witness_labels_and_current_model_custody(tmp_path, order):
    plan = transaction(tmp_path, order=order)
    context = prepare_and_context(plan)
    result = source_provider_builder(plan.builder)(context)
    assert set(plan.events) == {"row", "pointwise"}
    assert len(result.objects) == 2
    assert (result.prepared_sha256, result.model_llvm_sha256, result.model_object_sha256) == tuple(
        sha256_file(path) for path in (context.prepared_path, context.model_llvm_path, context.model_object_path)
    )
    manifest = context.workdir / "source_lowerings.json"
    aggregate = json.loads(manifest.read_text())
    assert aggregate["schema"] == "merlin.source_provider_bundle.v1"
    assert aggregate["scope"] == "individual witness custody only; no aggregate execution or numerical qualification"
    assert {row["family"] for row in aggregate["providers"]} == {"row", "pointwise"}
    for row in aggregate["providers"]:
        for label in ("numerical", "effects"):
            assert row[label] == witness(row["family"], label)
    assert result.numerical_witness_sha256 == result.effect_witness_sha256 == sha256_file(manifest)
    for artifact in result.objects:
        pins = dict(artifact.compilation_pins)
        assert pins[manifest.resolve()] == sha256_file(manifest)
        for config in plan.configurations.values():
            for path in (config.selected_contracts.facts_path, config.selected_contracts.software_path):
                assert pins[path.resolve()] == sha256_file(path)


@pytest.mark.parametrize("family", ["row", "pointwise"])
@pytest.mark.parametrize("field", ["prepared_sha256", "model_llvm_sha256", "model_object_sha256"])
def test_child_cannot_relabel_a_different_prepared_or_compiled_model(tmp_path, family, field):
    def mutate(name, context, result):
        return replace(result, **{field: "0" * 64}) if name == family else result

    plan = transaction(tmp_path, mutate=mutate)
    context = prepare_and_context(plan)
    with pytest.raises(ValueError, match="lost current model identity"):
        plan.builder(context)
    assert not (context.workdir / "source_lowerings.json").exists()


@pytest.mark.parametrize(
    "path_field,pin_field",
    [
        ("prepared_path", "prepared_sha256"),
        ("model_llvm_path", "model_llvm_sha256"),
        ("model_object_path", "model_object_sha256"),
    ],
)
def test_child_cannot_change_model_files_before_other_family_runs(tmp_path, path_field, pin_field):
    def mutate(name, context, result):
        if len(plan.events) == 1:
            path = getattr(context, path_field)
            path.write_bytes(path.read_bytes() + b"\nchanged by first child")
            # Even a newly computed self-report cannot retarget the transaction.
            return replace(result, **{pin_field: sha256_file(path)})
        return result

    plan = transaction(tmp_path, mutate=mutate)
    context = prepare_and_context(plan)
    with pytest.raises(ValueError, match="changed|lost current model identity"):
        plan.builder(context)
    assert len(plan.events) == 1
    assert not (context.workdir / "source_lowerings.json").exists()


@pytest.mark.parametrize("field", ["numerical_witness_sha256", "effect_witness_sha256"])
def test_invalid_child_witness_identity_refuses_before_aggregate_manifest(tmp_path, field):
    def mutate(name, context, result):
        return replace(result, **{field: "unit fixture, not a qualified witness identity"})

    plan = transaction(tmp_path, mutate=mutate)
    context = prepare_and_context(plan)
    with pytest.raises(ValueError, match="lost current model identity or pinned object/witnesses"):
        plan.builder(context)
    assert len(plan.events) == 1
    assert not (context.workdir / "source_lowerings.json").exists()


@pytest.mark.parametrize("mutation", ["row_domain", "pointwise_domain", "pointwise_limits", "pointwise_budgets"])
def test_post_preparation_policy_or_execution_selection_drift_refuses_before_child_build(tmp_path, mutation):
    plan = transaction(tmp_path)
    context = prepare_and_context(plan)
    pointwise = plan.configurations["pointwise"]
    if mutation.endswith("domain"):
        object.__setattr__(plan.configurations[mutation.split("_")[0]], "execution_domain", "device")
    elif mutation == "pointwise_limits":
        object.__setattr__(pointwise.limits, "max_instructions", pointwise.limits.max_instructions + 1)
    else:
        object.__setattr__(pointwise.budgets, "max_fusions", pointwise.budgets.max_fusions + 1)
    with pytest.raises(ValueError, match="selection changed"):
        plan.builder(context)
    assert not plan.events
    assert not (context.workdir / "source_lowerings.json").exists()


@pytest.mark.parametrize("device_family", ["row", "pointwise"])
@pytest.mark.parametrize("order", [("row", "pointwise"), ("pointwise", "row")])
def test_active_device_family_cannot_be_downgraded_by_host_family(tmp_path, device_family, order):
    plan = transaction(tmp_path, changes={device_family: dict(execution_domain="device")}, order=order)
    with pytest.raises(ValueError, match="active device route requires"):
        apply_prepared_model_transform(plan.source, tmp_path / "preparation", plan.prepare)
    assert not any(plan.seen.values()) and not plan.events
    assert not (tmp_path / "preparation" / "source_schedules.mlir").exists()


@pytest.mark.parametrize("device_family", ["row", "pointwise"])
def test_unused_device_family_keeps_host_work_without_requiring_unused_policy(tmp_path, device_family):
    controls = dict(execution_domain="device")
    if device_family == "row":
        controls["assume_finite_inputs"] = False
    else:
        controls["scalar_qualification_identity"] = None
    plan = transaction(tmp_path, changes={device_family: controls})
    context = prepare_and_context(plan)
    assert not plan.seen[device_family]
    assert len(plan.seen["pointwise" if device_family == "row" else "row"]) == 1
    assert source_device_work(plan.builder) is False
    assert source_provider_builder(plan.builder) is plan.builder
    assert not plan.events
    assert context.prepared_path.is_file()
