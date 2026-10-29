"""Whole-source placement narrows ordinary dense discovery by exact SSA identity."""

import json
from dataclasses import replace

import pytest
from test_dense_source import contraction
from test_source_dense_lowering import run, selection
from xdsl.dialects import builtin, func

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.offload_plan import bind_source, realize_offload_plan
from merlin.llvmlower.source_dense import offload_context
from merlin.llvmlower.source_model import source_provider_builder
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, GlobalPlan


def attach_plan(config, module, *, indices=(1,), endpoint=None, implementation=None):
    """Bind complete source, retaining unpriced glue and the original CPU body."""
    context = offload_context(config)
    bound = bind_source(module, entry="forward", context=context)
    device = tuple(
        replace(
            bound.cpu_alternatives[index],
            id=f"admitted_dense_{index}",
            placement=endpoint or config.selected_contracts.target,
            implementation=implementation or config.adapter.qualification_identity,
        )
        for index, operation in enumerate(bound.operations)
        if operation.name == "linalg.generic" and len(operation.iterator_types) == 3
    )
    chosen = {alternative.nodes[0]: alternative for i, alternative in enumerate(device) if i in indices}
    plan = GlobalPlan(
        tuple(chosen.get(index, alternative) for index, alternative in enumerate(bound.cpu_alternatives)),
        (),
        CycleInterval.unknown("source selection does not qualify performance"),
    )
    placed = realize_offload_plan(bound, plan, device_alternatives=device, context=context)
    return replace(config, offload_plan=placed)


def planned_source(tmp_path, *, indices=(1,), shape=(3, 7, 5), hardware=None, **controls):
    """Two live equal-shape/provenance sites; return original config, source and observer."""
    config, _, seen = selection(tmp_path, hardware=hardware, **controls)
    m, k, n = shape
    dtypes = config.selected_contracts.dense.dtypes
    x, y, z = dtypes
    module = parse_mlir_text(f"""module {{func.func @forward(
      %a:tensor<{m}x{k}x{x}>,%b:tensor<{k}x{n}x{y}>)
      ->(tensor<{m}x{n}x{z}>,tensor<{m}x{n}x{z}>) {{
      {contraction(shape, dtypes, output="first", tag="same")}
      {contraction(shape, dtypes, output="second", tag="same")}
      func.return %first,%second:tensor<{m}x{n}x{z}>,tensor<{m}x{n}x{z}>
    }}}}""")
    return attach_plan(config, module, indices=indices), module, seen


@pytest.mark.parametrize("indices", [(0,), (1,), (0, 1)])
@pytest.mark.parametrize("hardware", [None, {"tile": (3, 2, 5), "widths": (2, 2, 4)}])
def test_current_exact_dense_sites_are_selected_before_backend_preparation(tmp_path, indices, hardware):
    config, module, seen = planned_source(tmp_path, indices=indices, hardware=hardware)
    result, builder = run(tmp_path, config, module)
    assert source_provider_builder(builder) is builder
    assert len(seen) == len(indices)
    expected = config.offload_plan.selected_ordinals
    assert tuple(binding.compilation.operation_ordinal for binding in seen) == expected
    parsed = mq.parse(result)
    function = next(op for op in parsed.body.block.ops if op.sym_name.data == "forward")
    returned = function.body.block.last_op.arguments
    for index, value in enumerate(returned):
        assert value.owner.name == ("func.call" if index in indices else "linalg.generic")
    calls = tuple(op for op in function.walk() if type(op) is func.CallOp)
    assert len(calls) == len(indices)
    # Both sites have identical shapes and labels; the original ordinal picks one.
    for call in calls:
        assert tuple(call.arguments[:2]) == tuple(function.body.block.args)
    report = json.loads((result.parent / "source_dense.json").read_text())
    (placement,) = report["offload_placement"]
    assert placement["actual_route"] == "candidate" and placement["comparison"] is None
    assert placement["selected_dense_ordinals"] == list(expected)
    source_generics = [i for i, op in enumerate(module.walk()) if op.name == "linalg.generic"]
    assert placement["retained_cpu_dense_ordinals"] == [i for i in source_generics if i not in expected]
    assert not config.offload_plan.plan.cycles.resolved


@pytest.mark.parametrize("supported", [False, True])
def test_all_cpu_plan_keeps_exact_source_and_never_prepares_provider(tmp_path, supported):
    # Remove execution support before binding the plan, never after its context freeze.
    config, _, seen = selection(tmp_path)
    if not supported:
        config = replace(config, adapter=replace(config.adapter, synchronous_completion=False))
    from test_offload_plan import example

    bound, _, _ = example()
    module = bound.module
    config = attach_plan(config, module, indices=())
    result, builder = run(tmp_path, config, module)
    assert source_provider_builder(builder) is None and not seen
    assert result.read_bytes() == (tmp_path / "model.mlir").read_bytes()


@pytest.mark.parametrize("mutation", ["alignment", "storage", "budget", "completion", "ownership"])
def test_changed_original_dense_configuration_refuses_stale_plan_before_callback(tmp_path, mutation):
    config, module, seen = planned_source(tmp_path)
    changed = {
        "alignment": {"allocation_alignment": 128},
        "storage": {
            "storage": replace(
                config.storage, lhs=replace(config.storage.lhs, capacity=config.storage.lhs.capacity + 8)
            )
        },
        "budget": {"static_budget": config.static_budget + 1},
        "completion": {"adapter": replace(config.adapter, synchronous_completion=False)},
        "ownership": {"private_ownership": False},
    }[mutation]
    with pytest.raises(ValueError, match="context|changed|stale|differs|replan"):
        changed_config = replace(config, **changed)
        run(tmp_path, changed_config, module)
    assert not seen


@pytest.mark.parametrize("mutation", ["source", "wiring", "facts", "endpoint", "implementation"])
def test_current_source_and_original_endpoint_are_joined_before_callback(tmp_path, mutation):
    config, module, seen = planned_source(tmp_path, shape=(5, 5, 5))
    if mutation == "source":
        module.attributes["prov.current"] = builtin.UnitAttr()
    elif mutation == "wiring":
        operation = [op for op in module.walk() if op.name == "linalg.generic"][-1]
        operation.operands = (operation.operands[0], operation.operands[0], operation.operands[2])
    elif mutation == "facts":
        path = config.selected_contracts.facts_path
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed|stale|differs|replan|endpoint|implementation"):
        if mutation in ("endpoint", "implementation"):
            config = attach_plan(
                replace(config, offload_plan=None),
                module,
                **{mutation: "different original declaration"},
            )
        run(tmp_path, config, module)
    assert not seen


@pytest.mark.parametrize("unavailable", ["completion", "ownership", "capacity", "code_budget"])
def test_plan_cannot_grant_missing_dense_execution_or_storage_support(tmp_path, unavailable):
    config, module, seen = planned_source(tmp_path)
    config = replace(config, offload_plan=None)
    if unavailable == "completion":
        config = replace(config, adapter=replace(config.adapter, synchronous_completion=False))
    elif unavailable == "ownership":
        config = replace(config, private_ownership=False)
    elif unavailable == "capacity":
        config = replace(config, storage=replace(config.storage, accumulator=None))
    else:
        config = replace(config, static_budget=0)
    config = attach_plan(config, module)
    with pytest.raises(ValueError, match="without a legal current provider"):
        run(tmp_path, config, module)
    assert not seen


def test_plan_cannot_split_fused_dense_readout_numerical_region(tmp_path):
    from test_dense_integer_readout import readout_source
    from test_source_dense_readout_lowering import selection as readout_selection

    config, _, seen = readout_selection(tmp_path)
    module, _, _ = readout_source((3, 7, 5), config.selected_contracts.dense.dtypes)
    config = attach_plan(config, module, indices=(0,))
    with pytest.raises(ValueError, match="splits an admitted dense numerical region"):
        run(tmp_path, config, module)
    assert not seen


@pytest.mark.parametrize("mutation", ["plan", "budget", "qualification"])
def test_preparation_callback_cannot_replace_original_placement_or_its_context(tmp_path, mutation):
    def mutate(bindings, directory):
        if mutation == "plan":
            object.__setattr__(config.offload_plan.plan, "selected", config.offload_plan.source.cpu_alternatives)
        elif mutation == "budget":
            object.__setattr__(config, "static_budget", config.static_budget + 1)
        else:
            object.__setattr__(config.adapter, "qualification_identity", "different implementation")

    config, module, seen = planned_source(tmp_path, prepare=mutate)
    with pytest.raises(ValueError, match="changed|stale|differs|replan"):
        run(tmp_path, config, module)
    assert len(seen) == 1
    assert not (tmp_path / "preparation" / "prepared_model_transform.json").exists()
