"""Actual multi-provider SSA materialization and independent partition checks."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_selected_execution import roster
from xdsl.dialects import builtin, func

from merlin.compiler.partition.regions import partition_execution
from merlin.compiler.planning.execution import select_execution
from merlin.transforms.certificates import ir_digest
from merlin.verification.partition import verify_partition


@pytest.mark.parametrize("owner_name", ["preview", "semantics", "materializer", "checker"])
def test_partition_retains_actual_shared_source_owners(tmp_path, monkeypatch, owner_name):
    from merlin.compiler.ir import region, semantics
    from merlin.compiler.partition import regions
    from merlin.verification import partition as checker

    owner = {"preview": region, "semantics": semantics, "materializer": regions, "checker": checker}[owner_name]
    copied = tmp_path / "observed-owner.py"
    copied.write_bytes(Path(owner.__file__).read_bytes())
    monkeypatch.setattr(owner, "__file__", str(copied))
    source, _, cpu, device = roster(tmp_path)
    selected = select_execution(source, (*cpu, *device), preferred_providers=("vector", "array"))
    result = partition_execution(selected)
    record = result.source_observation()
    assert any(row["path"] == str(copied) for row in record["files"])
    assert result.verification.method == "complete_ssa_beta_expansion"
    copied.write_bytes(copied.read_bytes() + b"\n# changed after materialization\n")
    with pytest.raises(ValueError, match="implementation bytes changed"):
        result.revalidate()


def test_preview_rechecks_checker_source_after_actual_independent_verification(tmp_path, monkeypatch):
    from merlin.compiler.ir import region
    from merlin.verification import partition as checker

    copied = tmp_path / "checker.py"
    copied.write_bytes(Path(checker.__file__).read_bytes())
    monkeypatch.setattr(checker, "__file__", str(copied))
    source, _, _, _ = roster(tmp_path)
    original = region.verify_region_preview

    def verify(*args, **kwargs):
        receipt = original(*args, **kwargs)
        copied.write_bytes(copied.read_bytes() + b"\n# changed during preview\n")
        return receipt

    monkeypatch.setattr(region, "verify_region_preview", verify)
    with pytest.raises(ValueError, match="implementation bytes changed"):
        region.preview_region(source.module, source.entry, source.operations, symbol="preview")


def test_materialization_rechecks_actual_preview_callback_binding(tmp_path, monkeypatch):
    from merlin.compiler.partition import regions

    source, _, cpu, device = roster(tmp_path)
    selected = select_execution(source, (*cpu, *device), preferred_providers=("vector", "array"))
    original = regions.preview_region

    def preview(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(regions, "preview_region", original)
        return result

    monkeypatch.setattr(regions, "preview_region", preview)
    with pytest.raises(ValueError, match="implementation callable changed"):
        partition_execution(selected)


def test_region_preview_reuses_full_source_beta_expansion_and_keeps_dead_operations(tmp_path):
    from merlin.compiler.ir.region import preview_region
    from merlin.verification.partition import verify_region_preview

    source, _, _, _ = roster(tmp_path, shape=(5, 5, 5))
    preview = preview_region(source.module, source.entry, source.operations, symbol="preview")
    assert preview.verification.source_sha256 == ir_digest(source.module)
    actual = preview.module.clone()
    function = actual.body.block.first_op
    product = next(op for op in function.walk() if op.name == "arith.muli")
    product.attributes["required.changed_meaning"] = builtin.UnitAttr()
    with pytest.raises(ValueError, match="beta-expansion"):
        verify_region_preview(
            source.module,
            actual,
            entry=source.entry,
            operations=source.operations,
            inputs=preview.inputs,
            outputs=preview.outputs,
        )
    with pytest.raises(ValueError, match="contiguous"):
        preview_region(source.module, source.entry, (source.operations[0], source.operations[2]), symbol="bad")
    function = source.module.body.block.first_op
    function.properties["arg_attrs"] = builtin.ArrayAttr(
        builtin.DictionaryAttr({"merlin.input_domain": builtin.StringAttr("unproved")}) for _ in function.args
    )
    with pytest.raises(ValueError, match="inherited.*signature"):
        preview_region(source.module, source.entry, source.operations, symbol="bad")


@pytest.mark.parametrize("scope", ("module", "function", "return"))
def test_region_preview_never_discards_inherited_execution_or_result_policies(tmp_path, scope):
    from merlin.compiler.ir.region import preview_region

    source, _, _, _ = roster(tmp_path)
    function = source.module.body.block.first_op
    owner = {"module": source.module, "function": function, "return": function.get_return_op()}[scope]
    owner.attributes["required.numerical_policy"] = builtin.StringAttr("not_projected")
    with pytest.raises(ValueError, match="inherited"):
        preview_region(source.module, source.entry, source.operations, symbol="refused")


def test_two_provider_partition_preserves_exact_source_and_all_boundaries(tmp_path):
    source, _, cpu, device = roster(tmp_path, shape=(5, 5, 5))
    before = ir_digest(source.module)
    selected = select_execution(source, (*cpu, *device), preferred_providers=("vector", "array"))
    result = partition_execution(selected)
    assert ir_digest(source.module) == before
    assert {region.provider for region in result.regions} == {"vector", "array"}
    assert len({region.source_ordinals for region in result.regions}) == 2
    assert len({region.symbol for region in result.regions}) == 2
    assert all(len(region.inputs) == 3 for region in result.regions)  # both operands plus actual init
    assert len(result.emission.regions) == len(selected.selected)
    assert result.verification.source_sha256 == before
    from merlin.verification import partition as checker
    from merlin.verify import partition as legacy

    assert checker is legacy
    assert result.verification.verifier_sha256 == hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest()
    result.revalidate()
    # Equal-shape operands can be swapped without a type error; the independent
    # whole-source expansion must still reject the changed numerical program.
    changed = result.module.clone()
    entry = next(op for op in changed.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == source.entry)
    call = next(op for op in entry.body.block.ops if type(op) is func.CallOp)
    call.operands = (call.operands[1], call.operands[0], *call.operands[2:])
    changed.verify()
    with pytest.raises(ValueError, match="beta-expansion"):
        verify_partition(source.module, changed, entry=source.entry, symbols=tuple(r.symbol for r in result.regions))


def test_cpu_retention_and_mutated_partition_refusal(tmp_path):
    source, _, cpu, device = roster(tmp_path)
    cpu_plan = select_execution(source, (*cpu, *device), preferred_providers=("merlin.scalar",))
    unchanged = partition_execution(cpu_plan)
    assert not unchanged.regions and unchanged.module.is_structurally_equivalent(source.module)
    selected = select_execution(source, (*cpu, *device), preferred_providers=("array", "vector"))
    partitioned = partition_execution(selected)
    partitioned.regions[0].source_module.body.block.first_op.sym_name = builtin.StringAttr("changed")
    with pytest.raises(ValueError, match="partitioned source changed"):
        partitioned.revalidate()


def test_actual_source_binding_keeps_initializers_and_cannot_change_graph_contract(tmp_path):
    from test_offload_plan import example

    from merlin.llvmlower.offload_plan import bind_source, realize_offload_plan
    from merlin.xdsl_dialects.lowering.global_plan import CycleInterval, GlobalPlan

    legacy, _, context = example()
    actual = bind_source(legacy.module, entry=legacy.entry, context=context, graph_mode="ssa")
    assert actual.operations == legacy.operations
    # The legacy outliner clones init into its kernel; exact SSA must retain the
    # original explicit edge for partitioning and resource lifetime accounting.
    assert len(legacy.program.nodes[3].inputs) == 2
    assert len(actual.program.nodes[3].inputs) == 3
    assert actual.program.nodes[3].inputs[-1] == actual.program.nodes[2].outputs[0]
    assert actual.program.nodes[2].inputs[-1] == actual.program.nodes[1].outputs[0]
    with pytest.raises(ValueError, match="source binding changed"):
        replace(actual, graph_mode="outlined").revalidate()
    plan = GlobalPlan(actual.cpu_alternatives, (), CycleInterval.unknown("unpriced"))
    with pytest.raises(ValueError, match="outlined graph contract"):
        realize_offload_plan(actual, plan, device_alternatives=(), context=context)
    actual.program.nodes[3].inputs.pop()
    with pytest.raises(ValueError, match="source graph changed"):
        actual.revalidate()


@pytest.mark.parametrize("mutation", ["region-replacement", "region-metadata", "dispatch-metadata", "return-metadata"])
def test_partition_rejects_descriptor_and_emission_metadata_changes(tmp_path, mutation):
    source, _, cpu, device = roster(tmp_path)
    selected = select_execution(source, (*cpu, *device), preferred_providers=("vector", "array"))
    partitioned = partition_execution(selected)
    if mutation == "region-replacement":
        changed = replace(partitioned.regions[0], provider="different-provider")
        partitioned = replace(partitioned, regions=(changed, *partitioned.regions[1:]))
        reason = "partition ownership changed"
    elif mutation == "region-metadata":
        object.__setattr__(partitioned.regions[0], "provider", "different-provider")
        reason = "partitioned source changed"
    elif mutation == "dispatch-metadata":
        partitioned.emission.dispatch.nodes[0].prov["prov.changed"] = "different-lineage"
        reason = "partition emission changed"
    else:
        changed = partitioned.module.clone()
        helper = next(
            op
            for op in changed.body.block.ops
            if type(op) is func.FuncOp and op.sym_name.data == partitioned.regions[0].symbol
        )
        helper.body.block.last_op.attributes["prov.changed"] = builtin.StringAttr("different")
        with pytest.raises(ValueError, match="ordinary return"):
            verify_partition(
                source.module,
                changed,
                entry=source.entry,
                symbols=tuple(region.symbol for region in partitioned.regions),
            )
        return
    with pytest.raises(ValueError, match=reason):
        partitioned.revalidate()
