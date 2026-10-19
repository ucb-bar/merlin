"""Dense discovery, legal fallback and fixed-capability model preparation."""

import json
from dataclasses import replace

import pytest
from test_dense_source import source
from test_selected_dense_contracts import inputs
from xdsl.dialects import builtin, func

from merlin.common import mlir_query as mq
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.source_dense import SourceDenseLowering
from merlin.llvmlower.source_model import bind_source_lowerings, source_provider_builder
from merlin.transforms.certificates import ir_text


def selection(tmp_path, *, prepare=None, hardware=None, **controls):
    selected, problem, storage, adapter = inputs(tmp_path, **(hardware or {}))
    seen = []

    def implementation(bindings, directory):
        seen.extend(bindings)
        if prepare is not None:
            prepare(bindings, directory)

    def build(bindings, context):
        pytest.fail("discovery test must never build or execute a provider")

    config = SourceDenseLowering(
        selected, storage, adapter, "host", 64, 80, implementation, build,
        private_ownership=True, external_buffers_disjoint=True, verification_timeout_ms=20_000,
    )
    return replace(config, **controls), problem, seen


def run(tmp_path, config, module):
    path = tmp_path / "model.mlir"
    path.write_text(ir_text(module))
    original = path.read_bytes()
    transform, builder = bind_source_lowerings((config,), None, None, None)
    result = apply_prepared_model_transform(path, tmp_path / "preparation", transform)
    assert path.read_bytes() == original
    mq.parse(result).verify()
    return result, builder


@pytest.mark.parametrize("tile,widths,shape,shared", [
    ((2, 3, 4), (1, 1, 4), (5, 14, 13), False),
    ((3, 2, 5), (2, 1, 4), (7, 9, 16), True),
    ((2, 3, 4), (2, 2, 4), (1, 7, 9), False),
    ((2, 3, 4), (4, 4, 4), (3, 8, 5), False),
])
def test_current_source_and_selected_resources_drive_dense_model_discovery(tmp_path, tile, widths, shape, shared):
    config, _, seen = selection(tmp_path, hardware=dict(tile=tile, widths=widths, shape=shape, shared=shared))
    module, operation = source(shape, config.selected_contracts.dense.dtypes)
    # Provenance is not needed for either discovery or strategy selection.
    operation.attributes.pop("prov.region_id", None)
    operation.properties.pop("prov.region_id", None)
    result, builder = run(tmp_path, config, module)
    assert len(seen) == 1 and source_provider_builder(builder) is builder
    binding = seen[0]
    binding.compilation.revalidate()
    assert binding.compilation.compiled.choice.candidates_considered > 0
    body = next(op for op in mq.parse(result).body.block.ops if op.sym_name.data == "forward").body.block
    call = next(op for op in body.ops if type(op) is func.CallOp)
    assert tuple(call.arguments[:2]) == tuple(body.args)
    assert call.arguments[-1].owner.name == "tensor.empty"
    record = json.loads((result.parent / "source_dense.json").read_text())
    assert record["schema"] == "merlin.source_dense_lowering.v1"
    assert record["bindings"][0]["family"] == "dense"
    assert record["bindings"][0]["shape"] == [shape[0], shape[2]]
    assert record["bindings"][0]["schedule"]["kernel_sha256"] == binding.compilation.compiled.kernel.digest()


@pytest.mark.parametrize("unavailable", [
    "ownership", "alias", "qualification", "completion", "continuation", "accumulator", "code_budget",
    "arithmetic", "initialization", "unknown_effect", "source_budget", "capability", "reduction_tail",
])
def test_unavailable_dense_routes_preserve_exact_source_without_provider(tmp_path, unavailable):
    config, _, seen = selection(tmp_path)
    module, operation = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    if unavailable in ("ownership", "alias", "code_budget", "source_budget"):
        key, value = {
            "ownership": ("private_ownership", False), "alias": ("external_buffers_disjoint", False),
            "code_budget": ("static_budget", 0), "source_budget": ("max_source_elements", 4),
        }[unavailable]
        config = replace(config, **{key: value})
    elif unavailable in ("qualification", "completion", "continuation"):
        key, value = {
            "qualification": ("qualification_identity", None), "completion": ("synchronous_completion", False),
            "continuation": ("exact_accumulator_continuation", False),
        }[unavailable]
        config = replace(config, adapter=replace(config.adapter, **{key: value}))
    elif unavailable == "accumulator":
        config = replace(config, storage=replace(config.storage, accumulator=None))
    elif unavailable in ("capability", "reduction_tail"):
        import hashlib

        import yaml

        from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

        selected = config.selected_contracts
        software = yaml.safe_load(selected.software_bytes)
        if unavailable == "capability":
            del software["scheduling"]["dense"]
        else:
            software["scheduling"]["dense"]["support"]["partial_reduction_tiles"] = False
        software["isa"]["facts_sha256"] = hashlib.sha256(selected.facts_bytes).hexdigest()
        selected.software_path.write_text(yaml.safe_dump(software))
        current = read_scheduling_contracts(selected.facts_path, selected.software_path, target=selected.target)
        config = replace(config, selected_contracts=current,
                         adapter=replace(config.adapter, configuration_identity=current.hardware_identity))
    elif unavailable == "unknown_effect":
        operation.attributes["opaque_effect"] = builtin.UnitAttr()
    elif unavailable == "initialization":
        fill = operation.outputs[0].owner
        fill.regions[0].block.last_op.operands = (fill.regions[0].block.args[1],)
    else:
        multiply = next(op for op in operation.walk() if op.name == "arith.muli")
        multiply.operands = (multiply.operands[0], multiply.operands[0])
    result, builder = run(tmp_path, config, module)
    assert not seen and source_provider_builder(builder) is None
    assert result.read_bytes() == (tmp_path / "model.mlir").read_bytes()


@pytest.mark.parametrize("mutation", ["storage", "budget", "source", "facts", "kernel", "index_stamp"])
def test_dense_implementation_cannot_change_checked_source_or_fixed_resources(tmp_path, mutation):
    def mutate(bindings, directory):
        if mutation == "storage":
            object.__setattr__(config.storage.lhs, "capacity", config.storage.lhs.capacity + 8)
        elif mutation == "budget":
            object.__setattr__(config, "static_budget", config.static_budget + 80)
        elif mutation == "source":
            path = directory.parent / "source_dense.mlir"
            path.write_bytes(path.read_bytes() + b"\n")
        elif mutation == "facts":
            path = config.selected_contracts.facts_path
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            compiled = bindings[0].compilation.compiled
            object.__setattr__(compiled, "index_bits", 32)
            if mutation == "index_stamp":
                object.__setattr__(compiled, "checked_index_bits", 32)

    config, _, _ = selection(tmp_path, prepare=mutate)
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    with pytest.raises(ValueError, match="changed|replan"):
        run(tmp_path, config, module)
    assert not (tmp_path / "preparation" / "prepared_model_transform.json").exists()


def test_live_storage_reservations_are_not_silently_dropped_or_fallback(tmp_path):
    config, _, seen = selection(tmp_path)
    view = config.storage.lhs
    config = replace(config, reservations=(AddressWindow(view.backing, view.base * view.unit_bytes, 1),))
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    with pytest.raises(ValueError, match="overlaps packing, staging or live storage"):
        run(tmp_path, config, module)
    assert not seen


def test_same_read_only_ssa_input_is_preserved_and_output_is_fresh(tmp_path):
    config, _, seen = selection(tmp_path)
    module, operation = source((5, 5, 5), config.selected_contracts.dense.dtypes)
    operation.operands = (operation.inputs[0], operation.inputs[0], operation.outputs[0])
    result, _ = run(tmp_path, config, module)
    assert len(seen) == 1
    call = next(op for op in mq.parse(result).walk() if type(op) is func.CallOp)
    assert call.arguments[0] is call.arguments[1] and call.arguments[-1] is not call.arguments[0]


@pytest.mark.parametrize("input_kind", ["facts", "software"])
def test_direct_dense_reader_rejects_stale_selected_inputs(tmp_path, input_kind):
    from merlin.compile.scheduling.dense_source import read_source_dense

    config, expected, _ = selection(tmp_path)
    _, operation = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    assert read_source_dense(operation, selected_contracts=config.selected_contracts) == expected
    selected = config.selected_contracts
    path = selected.facts_path if input_kind == "facts" else selected.software_path
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        read_source_dense(operation, selected_contracts=selected)


def test_incorrect_actual_instruction_addresses_raise_before_backend_preparation(tmp_path):
    from merlin.compile.scheduling.block_schedule import Load

    config, _, seen = selection(tmp_path)
    original = config.adapter.encode

    def encode(operation):
        if isinstance(operation, Load) and operation.role == "lhs":
            operation = replace(operation, row=config.storage.lhs.base + config.storage.lhs.capacity)
        return original(operation)

    config = replace(config, adapter=replace(config.adapter, encode=encode))
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    with pytest.raises(ValueError, match="invariant|bounds|window|storage"):
        run(tmp_path, config, module)
    assert not seen


@pytest.mark.parametrize("active", [False, True])
def test_dense_device_selection_retains_the_original_admission_requirement(tmp_path, active):
    config, _, seen = selection(tmp_path, execution_domain="device", private_ownership=active)
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    if active:
        with pytest.raises(ValueError, match="active device route requires"):
            run(tmp_path, config, module)
        assert not seen
    else:
        result, builder = run(tmp_path, config, module)
        assert source_provider_builder(builder) is None and not seen
        assert result.read_bytes() == (tmp_path / "model.mlir").read_bytes()


def test_dense_island_budget_refuses_before_implementation_or_partial_rewrite(tmp_path):
    config, _, seen = selection(tmp_path, max_islands=1)
    module, _ = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    other = module.body.block.first_op.clone()
    other.properties["sym_name"] = builtin.StringAttr("other")
    module.body.block.add_op(other)
    with pytest.raises(ValueError, match="island budget"):
        run(tmp_path, config, module)
    assert not seen
    assert not (tmp_path / "preparation" / "source_dense.mlir").exists()


def test_shared_zero_initializer_remains_available_to_other_source_users(tmp_path):
    config, _, seen = selection(tmp_path)
    module, operation = source((5, 14, 13), config.selected_contracts.dense.dtypes)
    function = module.body.block.first_op
    terminator = function.body.block.last_op
    terminator.operands = (*terminator.operands, operation.outputs[0])
    function.properties["function_type"] = builtin.FunctionType.from_lists(
        [arg.type for arg in function.body.block.args], [value.type for value in terminator.operands],
    )
    result, _ = run(tmp_path, config, module)
    assert len(seen) == 1
    function = next(op for op in mq.parse(result).body.block.ops if op.sym_name.data == "forward")
    returned = function.body.block.last_op.arguments
    assert returned[0].owner.name == "func.call" and returned[1].owner.name == "linalg.fill"


@pytest.mark.parametrize("order", [("dense", "pointwise", "row"), ("row", "pointwise", "dense")])
def test_dense_composes_with_other_source_families_in_one_original_snapshot(tmp_path, order):
    from test_pointwise_source import source_program
    from test_source_pointwise_lowering import fixture as pointwise_selection
    from test_source_row_lowering import selection as row_selection

    directories = {name: tmp_path / name for name in order}
    for directory in directories.values():
        directory.mkdir()
    dense, _, dense_seen = selection(directories["dense"])
    dense_module, _ = source((5, 14, 13), dense.selected_contracts.dense.dtypes)
    pointwise, _, pointwise_seen = pointwise_selection(directories["pointwise"])
    pointwise_module, _ = source_program()
    row, row_module, row_seen = row_selection(directories["row"])
    functions = []
    for name, module in (("dense", dense_module), ("pointwise", pointwise_module), ("row", row_module)):
        function = module.body.block.first_op.clone()
        function.properties["sym_name"] = builtin.StringAttr(name + "_entry")
        functions.append(function)
    module = builtin.ModuleOp(functions)
    module.verify()
    configurations = dict(dense=dense, pointwise=pointwise, row=row)
    transform, _ = bind_source_lowerings(tuple(configurations[name] for name in order), None, None, None)
    path = tmp_path / "model.mlir"
    original = ir_text(module)
    path.write_text(original)
    result = apply_prepared_model_transform(path, tmp_path / "preparation", transform)
    assert path.read_text() == original
    assert len(dense_seen) == len(pointwise_seen) == len(row_seen) == 1
    for binding in (*dense_seen, *pointwise_seen, *row_seen):
        binding.compilation.revalidate()
    mq.parse(result).verify()
    report = json.loads((result.parent / "source_schedules.json").read_text())
    assert [b["family"] for b in report["bindings"]] == ["dense", "pointwise", "row"]
