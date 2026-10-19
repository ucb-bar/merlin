"""Automatic model source selection from explicit row capabilities."""

import json
from dataclasses import replace

import pytest
from test_source_rows import case
from xdsl.dialects import builtin, func

from merlin.common import mlir_query as mq
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY as POLICY
from merlin.llvmlower.cpu_softmax import build_cpu_softmax_lastdim
from merlin.llvmlower.prepared_model_transform import apply_prepared_model_transform
from merlin.llvmlower.source_rows import SourceRowLowering, bind_source_row_lowering, source_row_provider_builder
from merlin.transforms.certificates import ir_text


def selection(tmp_path, *, prepare=None, **changes):
    selected, opts, original, _, _ = case(tmp_path)
    seen = []

    def implementation(bindings, work):
        seen.extend(bindings)
        if prepare is not None:
            prepare(bindings, work)

    def build(bindings, context):
        pytest.fail("discovery tests must not build or execute provider code")

    config = SourceRowLowering(
        selected, opts["adapter"], POLICY, "host", 32, implementation, build,
        assume_finite_inputs=True, private_ownership=True, disjoint_arguments=True,
    )
    return replace(config, **changes), original, seen


def run(tmp_path, config, original):
    source = tmp_path / "model.mlir"
    source.write_text(ir_text(original))
    before = source.read_bytes()
    prepare, builder = bind_source_row_lowering(config, None, None)
    path = apply_prepared_model_transform(source, tmp_path / "preparation", prepare)
    assert source.read_bytes() == before
    mq.parse(path).verify()
    return path, builder


@pytest.mark.parametrize("dependent", [False, True])
def test_all_islands_and_block_argument_bindings_are_selected_without_backend_source_routing(tmp_path, dependent):
    config, original, seen = selection(tmp_path)
    body = next(op for op in original.walk() if type(op) is func.FuncOp).body.block
    first = body.last_op.arguments[0]
    terminator = body.last_op
    operations, result = build_cpu_softmax_lastdim(
        first if dependent else body.args[0], backend_policy=POLICY, assume_finite_inputs=True,
    )
    body.insert_ops_before(operations, terminator)
    terminator.operands = (result,)
    path, builder = run(tmp_path, config, original)
    assert len(seen) == 2
    for binding in seen:
        binding.compilation.revalidate()
    assert seen[0].compilation.source.source is next(
        op for op in seen[0].compilation.module.walk() if type(op) is func.FuncOp
    ).body.block.args[0]
    parsed = mq.parse(path)
    body = next(op for op in parsed.body.block.ops if getattr(op, "sym_name", None).data == "forward").body.block
    calls = [op for op in body.ops if type(op) is func.CallOp]
    assert len(calls) == 2
    assert calls[0].arguments[0] is body.args[0]
    assert calls[1].arguments[0] is (calls[0].results[0] if dependent else body.args[0])
    assert source_row_provider_builder(builder) == builder
    report = json.loads((path.parent / "source_rows.json").read_text())
    assert report["execution_domain"] == "host"
    assert report["premises"] == dict(
        assume_finite_inputs=True, private_ownership=True, disjoint_arguments=True,
    )
    assert [row["shape"] for row in report["bindings"]] == [[2, 19], [2, 19]]


@pytest.mark.parametrize("unavailable", [
    "finite", "ownership", "alias", "completion", "state", "source", "capacity", "row_capability",
])
def test_unsupported_source_or_capabilities_keep_the_exact_source_bytes_and_no_provider(tmp_path, unavailable):
    config, original, seen = selection(tmp_path)
    if unavailable in ("finite", "ownership", "alias"):
        config = replace(config, **{dict(
            finite="assume_finite_inputs", ownership="private_ownership", alias="disjoint_arguments",
        )[unavailable]: False})
    elif unavailable in ("completion", "state"):
        config = replace(config, adapter=replace(config.adapter, **{
            "synchronous_completion" if unavailable == "completion" else "addressable_state": False,
        }))
    elif unavailable == "source":
        final = next(op for op in original.walk() if type(op) is func.ReturnOp).arguments[0].owner
        final.attributes["unknown_numerical_effect"] = builtin.UnitAttr()
    else:
        # A legal 19-column source cannot fit in a sixteen-byte scratch bank.
        import hashlib

        import yaml

        software = yaml.safe_load(config.selected_contracts.software_bytes)
        # Change selected current facts and re-resolve, never mutate an analysis.
        facts_path = config.selected_contracts.facts_path
        facts = json.loads(facts_path.read_text())
        if unavailable == "capacity":
            facts["facts"]["sizes"] = [16, 16]
            facts["facts"]["capacities"][:2] = [16, 16]
        else:
            software["scheduling"].pop("row_reduction")
        facts_path.write_text(json.dumps(facts))
        software["isa"]["facts_sha256"] = hashlib.sha256(facts_path.read_bytes()).hexdigest()
        config.selected_contracts.software_path.write_text(json.dumps(software))
        from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

        current = read_scheduling_contracts(facts_path, config.selected_contracts.software_path,
                                            target=config.selected_contracts.target)
        config = replace(config, selected_contracts=current,
                         adapter=replace(config.adapter, configuration_identity=current.hardware_identity))
    path, builder = run(tmp_path, config, original)
    assert path.read_bytes() == (tmp_path / "model.mlir").read_bytes()
    assert not seen
    assert source_row_provider_builder(builder) is None


@pytest.mark.parametrize("mutation", ["source", "selected", "facts", "software", "kernel", "index_stamp"])
def test_backend_preparation_cannot_change_checked_selection(tmp_path, mutation):
    def mutate(bindings, work):
        if mutation in ("source", "selected"):
            path = work.parent / ("input.mlir" if mutation == "source" else "source_rows.mlir")
            path.write_text(path.read_text() + "\n")
        elif mutation in ("facts", "software"):
            contracts = bindings[0].compilation.compiled.selected_contracts
            path = contracts.facts_path if mutation == "facts" else contracts.software_path
            path.write_bytes(path.read_bytes() + b"\n")
        else:
            compiled = bindings[0].compilation.compiled
            object.__setattr__(compiled, "index_bits", 32)
            if mutation == "index_stamp":
                object.__setattr__(compiled, "checked_index_bits", 32)

    config, original, _ = selection(tmp_path, prepare=mutate)
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, config, original)
    assert not (tmp_path / "preparation" / "prepared_model_transform.json").exists()


@pytest.mark.parametrize("mutation", ["source", "selected", "facts", "receipt"])
def test_post_preparation_changes_refuse_before_provider_compilation(tmp_path, mutation):
    config, original, _ = selection(tmp_path)
    path, builder = run(tmp_path, config, original)
    selected = {
        "source": path.parent / "input.mlir", "selected": path,
        "facts": config.selected_contracts.facts_path, "receipt": path.parent / "source_rows.json",
    }[mutation]
    selected.write_bytes(selected.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        source_row_provider_builder(builder)


def test_symbol_collisions_are_avoided_without_changing_source_selection(tmp_path):
    config, original, seen = selection(tmp_path)
    original.body.block.add_op(func.FuncOp.external("__merlin_source_row_0_borrowed", [], []))
    path, _ = run(tmp_path, config, original)
    assert seen[0].writer.symbol == "__merlin_source_row_1"
    mq.parse(path).verify()


def test_island_budget_refuses_before_implementation_or_partial_rewrite(tmp_path):
    config, original, seen = selection(tmp_path, max_islands=1)
    body = next(op for op in original.walk() if type(op) is func.FuncOp).body.block
    operations, _ = build_cpu_softmax_lastdim(body.args[0], backend_policy=POLICY, assume_finite_inputs=True)
    body.insert_ops_before(operations, body.last_op)
    with pytest.raises(ValueError, match="island budget"):
        run(tmp_path, config, original)
    assert not seen
    assert not (tmp_path / "preparation" / "source_rows.mlir").exists()


def test_empty_selection_preserves_existing_hooks_and_conflicting_selections_refuse(tmp_path):
    transform, builder = object(), object()
    assert bind_source_row_lowering(None, transform, builder) == (transform, builder)
    assert source_row_provider_builder(builder) is builder
    config, _, _ = selection(tmp_path)
    for hooks in ((transform, None), (None, builder)):
        with pytest.raises(ValueError, match="cannot replace"):
            bind_source_row_lowering(config, *hooks)
    with pytest.raises(ValueError, match="before ordinary preparation"):
        source_row_provider_builder(bind_source_row_lowering(config, None, None)[1])


def test_ordinary_builder_rejects_conflicting_hooks_before_creating_build_products(tmp_path):
    from merlin.runtime.backends.spike_model import build

    config, _, _ = selection(tmp_path)
    work = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="cannot replace"):
        build(tmp_path / "absent", work, shared_row_lowering=config, prepared_model_transform=lambda *args: None)
    assert not work.exists()


@pytest.mark.parametrize("field,value", [("execution_domain", "device"), ("allocation_alignment", 64)])
def test_callback_cannot_change_frozen_execution_domain_or_abi(tmp_path, field, value):
    def mutate(bindings, work):
        object.__setattr__(config, field, value)

    config, original, _ = selection(tmp_path, prepare=mutate)
    with pytest.raises(ValueError, match="selection changed"):
        run(tmp_path, config, original)


def test_device_rows_without_policy_refuse_before_backend_preparation(tmp_path):
    config, original, seen = selection(tmp_path, execution_domain="device")
    with pytest.raises(ValueError, match="active device route requires"):
        run(tmp_path, config, original)
    assert not seen


def test_unsupported_device_rows_keep_source_without_requiring_unused_policy(tmp_path):
    config, original, seen = selection(tmp_path, execution_domain="device", assume_finite_inputs=False)
    path, builder = run(tmp_path, config, original)
    assert not seen and source_row_provider_builder(builder) is None
    assert path.read_bytes() == (tmp_path / "model.mlir").read_bytes()
