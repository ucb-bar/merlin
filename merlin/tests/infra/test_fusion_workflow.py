"""Native source boundaries and actual pass products; no target simulation."""
import json
from dataclasses import replace

import pytest
from xdsl.dialects import func

from merlin.compile import fusion_workflow as F
from merlin.transforms.certificates import ir_digest
from test_multioutput_fusion import model, source_program
from test_physical_fusion import fixture as physical_fixture
from test_shared_input_schedule import fixture as shared_fixture


def test_discovery_on_distinct_graphs_retains_boundaries_and_unknown_legality():
    fanout, operations = source_program()
    before = ir_digest(fanout)
    report = F.discover(fanout, capability_contract={}, numerical_policy_sha256="a" * 64)
    assert report["pointwise_fanout"]["planned"]["source_regions"] == 1
    assert all(c["status"] == "unknown" and c["whole_region_cost"] is None for c in report["candidates"])
    assert ir_digest(fanout) == before
    json.dumps(report)
    # An external observer remains a boundary; the stricter native fanout reader refuses it.
    ret = next(op for op in fanout.walk() if isinstance(op, func.ReturnOp))
    ret.operands = [*ret.operands, operations[0].results[0]]
    ret.parent_op().update_function_type()
    observed = F.discover(fanout, capability_contract={}, numerical_policy_sha256="a" * 64)
    assert observed["pointwise_fanout"]["planned"]["source_regions"] == 0
    assert observed["pointwise_fanout"]["candidates"][0]["status"] == "declined"
    assert any(c["live_outputs"] for c in observed["candidates"])
    module, _, _, _, _ = shared_fixture()
    siblings = F.discover(module, capability_contract={}, numerical_policy_sha256="a" * 64)
    assert siblings["routes"]["shared_input"]["candidates"] == 1
    row = next(c for c in siblings["candidates"] if c["route"] == "shared_input")
    assert len(row["live_outputs"]) == 3
    assert row["storage_lifetimes"] == "unproved"
    json.dumps(siblings)


def test_adjacent_realization_returns_actual_calls_and_original_alternative():
    kernel, objects, banks, adapter, capability, encoding, _, options = physical_fixture()
    result = F.realize_adjacent(kernel, objects, banks, adapter, (capability,), **options)
    assert result.receipt["status"] == "applied"
    assert result.unfused is kernel and len(kernel.body) == 2
    assert len(result.artifact.body) == 1 and result.artifact.body[0].instr == "fused"
    assert encoding.decode_calls(encoding.encode(result.artifact)) == result.artifact.body
    assert result.objects == ()
    json.dumps(result.receipt)
    fallback = F.realize_adjacent(kernel, objects, banks, adapter, (), **options)
    assert fallback.artifact is kernel and fallback.receipt["status"] == "no-match"
    assert fallback.objects == objects


def test_shared_input_realization_keeps_independent_outputs_and_declines_capacity():
    module, group, tile, storage, resources = shared_fixture()
    before = ir_digest(module)
    result = F.realize_shared_input(module, group, primitive_tile=tile, storage=storage, resources=resources)
    assert result.receipt["status"] == "applied"
    assert result.artifact.realization and result.artifact.operations
    assert result.artifact.group.widths == (7, 3, 9)
    assert result.unfused is module and ir_digest(module) == before
    assert result.receipt["maturity"] == "logical_schedule_not_target_instruction_image"
    json.dumps(result.receipt)
    fallback = F.realize_shared_input(module, group, primitive_tile=tile, storage=storage,
                                     resources=replace(resources, operand_bytes=1))
    assert fallback.artifact is module and fallback.receipt["status"] == "no-match"


def test_multioutput_returns_guarded_actual_images(tmp_path):
    module, operations, options = model(tmp_path)
    result = F.realize_multioutput(module, operations, **options)
    assert result.receipt["status"] == "applied"
    assert result.artifact.images()
    assert result.receipt["instruction_images"][0]["bytes"] == len(result.artifact.images()[0])
    assert result.receipt["execution_qualification"] == "not_established"
    json.dumps(result.receipt)


def test_costs_require_complete_correct_same_scope_and_qualified_model():
    a = dict(source_ir_sha256="a"*64, numeric_policy_sha256="b"*64, hardware_sha256="c"*64,
             scope_sha256="d"*64, artifact_sha256="e"*64, method="physical", scope="complete_region",
             complete=True, correctness="pass", cycles=100)
    b = dict(a, artifact_sha256="f"*64, cycles=70)
    assert F.compare_region_costs(a, b)["saved_cycles"] == 30
    for mutation in ({"scope_sha256":"0"*64},{"complete":False},{"correctness":"unknown"},
                     {"scope":"compute_only"},{"cycles":float("nan")}):
        assert F.compare_region_costs(a, dict(b, **mutation))["status"] == "unknown"
    model_a = dict(a, method="model", model_sha256="0"*64, model_qualification="provisional")
    model_b = dict(b, method="model", model_sha256="0"*64, model_qualification="provisional")
    assert F.compare_region_costs(model_a, model_b)["status"] == "unknown"
    assert F.compare_region_costs(dict(model_a,model_qualification="qualified"),
                                  dict(model_b,model_qualification="qualified"))["saved_cycles"] == 30


def test_compact_loop_route_is_retained_without_unrolling():
    from merlin.sched.ir import Loop
    kernel, objects, banks, adapter, capability, _, _, options = physical_fixture()
    compact = replace(kernel, body=(Loop("tile", 1000000, kernel.body),))
    result = F.realize_adjacent(compact, objects, banks, adapter, (capability,), **options)
    assert result.artifact is compact and result.unfused is compact
    assert result.receipt["status"] == "unsupported"
    assert result.receipt["hardware_unsupported"] is False
    assert result.receipt["emitted"] == {}


def test_native_receipt_writer_accepts_source_discovery_without_emitted_claim(tmp_path):
    from merlin.compile.compiler_observations import write_receipt
    module, _ = source_program()
    section = F.discover(module, capability_contract={}, numerical_policy_sha256="a"*64)
    observation = write_receipt(tmp_path / "fusion.json", compiler_sha256="b"*64,
                               capsule_sha256="c"*64, scope_sha256="d"*64,
                               sections={"connected_fusion": section})
    assert observation["connected_fusion"]["emitted"] == {}
    saved = json.loads((tmp_path / "fusion.json").read_text())
    assert saved["sections"]["connected_fusion"]["candidates"]
    assert saved["promotion_authority"] is False


def test_discovery_boundary_includes_scalar_captures_inside_nested_regions():
    from xdsl.dialects import arith
    module, operations = source_program()
    external = arith.ConstantOp.from_int_and_width(3, 16)
    operations[0].parent.insert_op_before(external, operations[0])
    add = next(op for op in operations[0].walk() if isinstance(op, arith.AddiOp))
    add.operands = [add.operands[0], external.result]
    module.verify()
    report = F.discover(module, capability_contract={}, numerical_policy_sha256="a"*64)
    ordinals = {op:i for i,op in enumerate(module.walk())}
    captured = {"producer_ordinal":ordinals[external],"result":0,"type":"i16"}
    touched = [row for row in report["candidates"] if ordinals[operations[0]] in row["operation_ordinals"]]
    assert touched and all(captured in row["live_inputs"] for row in touched)
    # Internal scalar block arguments must not be mislabeled as external inputs.
    assert all(not any(v.get("block_owner_ordinal") == ordinals[operations[0]]
                       for v in row["live_inputs"]) for row in touched)


def test_discovery_reads_public_declarations_without_private_eligibility(monkeypatch):
    import builtins

    original = builtins.__import__
    def public_only(name, *args, **kwargs):
        if (name in {'merlin.targetgen.eligibility', 'merlin.targetgen.target_registry',
                     'merlin.xdsl_dialects.lowering.compute_groups'}
                or name == 'merlin.xdsl_dialects.lowering' and 'compute_groups' in kwargs.get('fromlist', args[2] if len(args)>2 else ())):
            pytest.fail('fusion discovery imported protected eligibility or target discovery')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', public_only)
    module, _ = source_program()
    empty = F.discover(module, capability_contract={}, numerical_policy_sha256='a'*64)
    families = {c['family'] for r in empty['candidates'] for c in r['capability_observations']}
    assert families
    family = sorted(families)[0]
    declaration = {'family':family, 'composed_with':['contraction'], 'forms':['explicit-form'],
                   'dtypes':['int8'], 'custom_constraint':{'retained':True}}
    contract = {'compute_units':[{'name':'public_unit','semantic_capabilities':[declaration]}]}
    report = F.discover(module, capability_contract=contract, numerical_policy_sha256='a'*64)
    for row in report['candidates']:
        assert row['status']=='unknown'  # Declaration is not legality or offload proof.
        for cap in row['capability_observations']:
            if cap['family']==family:
                assert cap['status']=='declared'
                assert cap['declarations']==[{'unit':'public_unit','declaration':declaration}]
                assert cap['composed_with']==[['contraction']]
            else:
                assert cap['status']=='unknown' and cap['declarations']==[]
            assert cap['applicability']=='unknown_until_native_legality_checks'
    # Another declared family cannot imply support for a derived/composite family.
    unrelated = F.discover(module, capability_contract={'compute_units':[
        {'semantic_capabilities':[{'family':'unrelated_explicit_family'}]}]},
        numerical_policy_sha256='a'*64)
    assert all(c['status']=='unknown' for r in unrelated['candidates'] for c in r['capability_observations'])


def test_structural_discovery_leaves_unclassified_reduction_visible():
    from xdsl.dialects.builtin import ArrayAttr
    module, operations = source_program()
    op = operations[0]
    iterators = op.properties['iterator_types'].data
    reduction = type(iterators[0])(type(iterators[0].data).REDUCTION)
    op.properties['iterator_types'] = ArrayAttr([reduction, *iterators[1:]])
    assert F._structural_family(op) is None
    report = F.discover(module, capability_contract={}, numerical_policy_sha256='a'*64)
    ordinal = next(i for i, member in enumerate(module.walk()) if member is op)
    assert {'ordinal':ordinal, 'operation':'linalg.generic'} in report['unclassified_tensor_operations']
    assert all(ordinal not in row['operation_ordinals'] for row in report['candidates'])
