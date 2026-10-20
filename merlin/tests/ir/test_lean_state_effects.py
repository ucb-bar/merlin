"""Actual SSA/config/memory schedule projection and conditional Lean state lemmas."""

from __future__ import annotations

from dataclasses import replace

import pytest

from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.planning import blocks_in
from merlin.verify.lean_state_effects import (
    EffectOperation,
    EffectScheduleContract,
    UnsupportedStateEffects,
    derive_effect_schedule,
    project_effect_schedule,
)

IDENTITY = "a" * 64


def _operation(name, inputs=(), outputs=(), reads=(), writes=(), observable=False):
    return EffectOperation(
        name,
        inputs,
        outputs,
        reads,
        writes,
        IDENTITY,
        True,
        True,
        observable=observable,
        sequential_atomic_complete=True,
    )


def _contract(**kwargs):
    return EffectScheduleContract(
        operations=(
            _operation("engine.config", ("i8",), writes=("configuration",)),
            _operation("engine.read", outputs=("i8",), reads=("configuration",)),
            _operation("engine.left", writes=("left",)),
            _operation("engine.right", writes=("right",)),
            _operation("engine.observe_left", reads=("left",), observable=True),
            _operation("engine.observe_right", reads=("right",), observable=True),
        ),
        **kwargs,
    )


def _source():
    return """builtin.module {
      func.func @kernel(%a:i8,%b:i8) -> i8 {
        %first = arith.addi %a,%b : i8
        "engine.config"(%a) : (i8) -> ()
        %dead = arith.muli %a,%b : i8
        %state = "engine.read"() : () -> i8
        %result = arith.addi %first,%state : i8
        func.return %result : i8
      }
    }"""


def _reorder(source, order, block=1):
    module = parse_ir(source)
    destination = blocks_in(module)[block]
    operations = tuple(destination.ops)
    for operation in operations:
        destination.detach_op(operation)
    destination.add_ops(operations[index] for index in order)
    return ir_text(module)


def test_infers_pure_work_moved_across_admitted_config_and_read_from_actual_ir():
    source = _source()
    after = _reorder(source, (0, 2, 1, 3, 4, 5))
    projection = derive_effect_schedule(source, after, entry="kernel", contract=_contract())
    assert projection.order == (0, 2, 1, 3, 4, 5)
    assert [(swap.first, swap.second) for swap in projection.exchanges] == [(1, 2)]
    assert (1, 3) in projection.dependencies and (3, 4) in projection.dependencies
    statements, proofs = projection.lean_obligations()
    assert len(statements) == len(proofs) == 1 and "StateEffects.Action" in statements[0]
    assert "by decide" in proofs[0]
    assert projection.contract_sha256 == _contract().identity()


def test_configuration_observation_cannot_move_before_its_writer():
    source = _source()
    after = _reorder(source, (0, 3, 2, 1, 4, 5))
    with pytest.raises(ValueError, match="dependency"):
        derive_effect_schedule(source, after, entry="kernel", contract=_contract())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda source: source.replace("arith.muli", "arith.subi"),
        lambda source: source.replace("%first,%state", "%dead,%state"),
        lambda source: source.replace('"engine.config"(%a)', '"engine.config"(%b)'),
    ],
)
def test_actual_changed_computation_or_state_operands_are_rejected(mutation):
    source = _source()
    with pytest.raises(ValueError):
        derive_effect_schedule(
            source, _reorder(mutation(source), (0, 2, 1, 3, 4, 5)), entry="kernel", contract=_contract()
        )


def test_untouched_function_change_is_not_hidden_by_selected_entry():
    source = _source().rsplit("}", 1)[0] + "func.func @other(%other_arg:i8) -> i8 { func.return %other_arg : i8 } }"
    after = _reorder(source, (0, 2, 1, 3, 4, 5)).replace('sym_name = "other"', 'sym_name = "different"')
    with pytest.raises(ValueError, match="outside its permutation"):
        derive_effect_schedule(source, after, entry="kernel", contract=_contract())


def test_unknown_instruction_effects_refuse_even_if_their_order_is_unchanged():
    source = _source().replace("engine.config", "unknown.configure")
    with pytest.raises(UnsupportedStateEffects, match="unknown operation"):
        derive_effect_schedule(source, _reorder(source, (0, 2, 1, 3, 4, 5)), entry="kernel", contract=_contract())


def test_poison_arithmetic_is_not_inferred_pure_from_traits():
    source = _source().replace("arith.muli %a,%b", "arith.muli %a,%b overflow<nsw>")
    with pytest.raises(UnsupportedStateEffects):
        derive_effect_schedule(source, _reorder(source, (0, 2, 1, 3, 4, 5)), entry="kernel", contract=_contract())


def test_distinct_resource_names_alone_never_assert_disjoint_state():
    source = """builtin.module { func.func @kernel() {
       "engine.left"() : () -> ()
       "engine.right"() : () -> ()
       func.return
    }}"""
    after = _reorder(source, (1, 0, 2))
    with pytest.raises(ValueError, match="dependency"):
        derive_effect_schedule(source, after, entry="kernel", contract=_contract())
    projection = derive_effect_schedule(
        source, after, entry="kernel", contract=_contract(disjoint_resources=(("left", "right"),))
    )
    assert len(projection.exchanges) == 1


def test_observable_event_order_is_preserved_even_with_disjoint_declared_resources():
    source = """builtin.module { func.func @kernel() {
       "engine.observe_left"() : () -> ()
       "engine.observe_right"() : () -> ()
       func.return
    }}"""
    with pytest.raises(ValueError, match="observation"):
        derive_effect_schedule(
            source,
            _reorder(source, (1, 0, 2)),
            entry="kernel",
            contract=_contract(disjoint_resources=(("left", "right"),)),
        )


def test_caller_memrefs_are_conservatively_aliasing_even_when_argument_names_differ():
    source = """builtin.module { func.func @kernel(%a:memref<7xi8>,%b:memref<7xi8>,%x:i8,%y:i8) {
       %zero = arith.constant 0 : index
       memref.store %x, %a[%zero] : memref<7xi8>
       memref.store %y, %b[%zero] : memref<7xi8>
       func.return
    }}"""
    with pytest.raises(ValueError, match="alias"):
        derive_effect_schedule(
            source,
            _reorder(source, (0, 2, 1, 3)),
            entry="kernel",
            contract=EffectScheduleContract(standard_memory_sequential=True, index_bits=64),
        )


def _owned_source(size=7):
    return f"""builtin.module {{ func.func @kernel(%x:i8,%y:i8) -> i8 {{
       %zero = arith.constant 0 : index
       %a = memref.alloc() : memref<{size}xi8>
       %b = memref.alloc() : memref<{size}xi8>
       memref.store %x, %a[%zero] : memref<{size}xi8>
       memref.store %y, %b[%zero] : memref<{size}xi8>
       %result = memref.load %a[%zero] : memref<{size}xi8>
       memref.dealloc %a : memref<{size}xi8>
       memref.dealloc %b : memref<{size}xi8>
       func.return %result : i8
    }}}}"""


@pytest.mark.parametrize("size", [7, 10**12 + 39])
def test_distinct_nonescaping_allocations_permit_store_reordering_without_element_expansion(size):
    source = _owned_source(size)
    projection = derive_effect_schedule(
        source,
        _reorder(source, (0, 1, 2, 4, 3, 5, 6, 7, 8)),
        entry="kernel",
        contract=EffectScheduleContract(
            standard_memory_sequential=True, standard_allocations_successful=True, index_bits=64
        ),
    )
    assert len(projection.nodes) == 9 and len(projection.exchanges) == 1
    assert len(projection.cells) < 30


def test_load_cannot_cross_a_store_to_its_own_allocation():
    source = _owned_source()
    with pytest.raises(ValueError, match="dependency"):
        derive_effect_schedule(
            source,
            _reorder(source, (0, 1, 2, 5, 3, 4, 6, 7, 8)),
            entry="kernel",
            contract=EffectScheduleContract(
                standard_memory_sequential=True, standard_allocations_successful=True, index_bits=64
            ),
        )


def test_unsafe_allocation_or_dynamic_access_premises_are_not_inferred():
    source = _owned_source()
    with pytest.raises(UnsupportedStateEffects, match="allocation premise"):
        derive_effect_schedule(
            source,
            source,
            entry="kernel",
            contract=EffectScheduleContract(standard_memory_sequential=True, index_bits=64),
        )
    source = """builtin.module { func.func @kernel(%a:memref<7xi8>, %i:index) -> i8 {
       %value = memref.load %a[%i] : memref<7xi8>
       func.return %value : i8
    }}"""
    with pytest.raises(UnsupportedStateEffects, match="defined-access"):
        derive_effect_schedule(
            source,
            source,
            entry="kernel",
            contract=EffectScheduleContract(standard_memory_sequential=True, index_bits=64),
        )


def test_structured_large_loop_effects_are_aggregated_when_moving_independent_pure_work():
    source = """builtin.module { func.func @kernel(%a:memref<1000000000039xi8>,%x:i8,%y:i8) -> i8 {
       %zero = arith.constant 0 : index
       %upper = arith.constant 1000000000039 : index
       %step = arith.constant 1 : index
       scf.for %i = %zero to %upper step %step {
          memref.store %x, %a[%i] : memref<1000000000039xi8>
       }
       %result = arith.addi %x,%y : i8
       func.return %result : i8
    }}"""
    projection = derive_effect_schedule(
        source,
        _reorder(source, (0, 1, 2, 4, 3, 5)),
        entry="kernel",
        contract=EffectScheduleContract(standard_memory_sequential=True, index_bits=64),
    )
    assert len(projection.nodes) == 6 and len(projection.exchanges) == 1
    assert len(projection.lean_obligations()[0][0]) < 2000


def test_opaque_nested_region_remains_explicitly_unsupported():
    source = """builtin.module { func.func @kernel(%a:i8,%b:i8)->i8 {
       "opaque.region"() ({ "engine.left"() : () -> () }) : () -> ()
       %sum = arith.addi %a,%b : i8
       func.return %sum : i8
    }}"""
    with pytest.raises(UnsupportedStateEffects, match="region semantics"):
        derive_effect_schedule(source, _reorder(source, (1, 0, 2)), entry="kernel", contract=_contract())


def test_projection_is_pinned_to_host_semantics_and_rejects_mutable_or_unadmitted_contracts():
    first = _contract()
    second = replace(first, disjoint_resources=(("left", "right"),))
    assert first.identity() != second.identity()
    with pytest.raises(ValueError, match="immutable"):
        replace(first, operations=list(first.operations)).validate()
    with pytest.raises(ValueError, match="totality"):
        replace(first, operations=(replace(first.operations[0], deterministic_total=False),)).validate()


def test_exchange_budget_refuses_large_candidate_permutations_before_lean_generation():
    source = _source()
    with pytest.raises(UnsupportedStateEffects, match="proof budget"):
        project_effect_schedule(
            source,
            _reorder(source, (2, 0, 1, 3, 4, 5)),
            block_index=1,
            order=(2, 0, 1, 3, 4, 5),
            contract=_contract(maximum_exchanges=1),
        )


def test_missing_atomic_completion_admission_refuses_even_a_pure_move():
    contract = _contract()
    unsafe = replace(contract, operations=(replace(contract.operations[0], sequential_atomic_complete=False),))
    with pytest.raises(ValueError, match="atomic semantics"):
        derive_effect_schedule(_source(), _reorder(_source(), (0, 2, 1, 3, 4, 5)), entry="kernel", contract=unsafe)


def test_unused_opaque_result_width_change_cannot_hide_in_xdsl_structural_equivalence():
    source = """builtin.module { func.func @kernel() {
      %unused = "engine.read"() : () -> i8
      func.return
    }}"""
    with pytest.raises(ValueError, match="correspondence"):
        derive_effect_schedule(source, source.replace("-> i8", "-> i16"), entry="kernel", contract=_contract())


def _isa(variant):
    import hashlib
    import json

    from merlin.targetgen.isa_spec import resolve_isa

    namespace, bits, width = ("alpha", 8, 4) if variant == 0 else ("beta", 16, 8)
    target = "synthetic_" + namespace
    facts = {
        "facts": {
            "target": target,
            "word": width + 2,
            "bits": bits,
            "fields": [{"offset": 0, "width": 2, "value": variant + 1}, {"offset": 2, "width": width}],
            "capacity": 5 + variant * 2,
        }
    }
    pin = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()

    def ref(*path):
        return {"fact": list(path)}

    declaration = {
        "schema": "merlin.isa_spec.v1",
        "namespace": namespace,
        "facts_sha256": pin,
        "word_bits": ref("facts", "word"),
        "resources": [{"name": "state", "kind": "state", "unit": "state_value"}],
        "instructions": [
            {
                "name": "issue",
                "operands": [
                    {
                        "name": "input",
                        "type": {"kind": "integer", "bits": ref("facts", "bits"), "signed": False},
                        "unit": "register",
                    }
                ],
                "results": [],
                "immediates": [],
                "effects": {"reads": [], "writes": ["state"]},
                "encoding": [
                    {
                        "offset": ref("facts", "fields", 0, "offset"),
                        "width": ref("facts", "fields", 0, "width"),
                        "fixed": ref("facts", "fields", 0, "value"),
                    },
                    {
                        "offset": ref("facts", "fields", 1, "offset"),
                        "width": ref("facts", "fields", 1, "width"),
                        "argument": "input",
                    },
                ],
            }
        ],
    }
    return resolve_isa(declaration, facts, target=target, facts_sha256=pin)


@pytest.mark.parametrize("variant", [0, 1])
def test_real_resolved_isa_factory_binds_widths_effects_facts_and_atomic_policy(variant):
    from merlin.verify.lean_state_effects import effect_contract_from_isa

    model = _isa(variant)
    name = model.namespace + ".issue"
    contract = effect_contract_from_isa((model,), admitted_instructions=(name,), atomic_instructions=(name,))
    width = model.instructions[0].operands[0].type.bits
    source = f'''builtin.module {{ func.func @kernel(%a:i{width},%b:i{width}) -> i{width} {{
      "{name}"(%a) : (i{width}) -> ()
      %sum = arith.addi %a,%b : i{width}
      func.return %sum : i{width}
    }}}}'''
    projection = derive_effect_schedule(source, _reorder(source, (1, 0, 2)), entry="kernel", contract=contract)
    assert contract.operations[0].writes == (model.namespace + ":state",)
    assert contract.operations[0].semantic_sha256 == model.identity()
    assert projection.contract_sha256 == contract.identity() and len(projection.exchanges) == 1
    with pytest.raises(ValueError, match="admission"):
        effect_contract_from_isa((model,), admitted_instructions=(name,), atomic_instructions=())
    wrong = source.replace(f"i{width}", f"i{width + 1}")
    from xdsl.utils.exceptions import VerifyException

    with pytest.raises(VerifyException):  # The independent generated type verifier refuses before projection.
        derive_effect_schedule(wrong, wrong, entry="kernel", contract=contract)


def test_isa_unknown_effects_cannot_be_repaired_by_an_atomic_admission_flag():
    from merlin.targetgen.isa_spec import resolved_isa_from_record
    from merlin.verify.lean_state_effects import effect_contract_from_isa

    record = _isa(0).record()
    record["instructions"][0]["effects_json"] = '{"status":"unknown"}'
    model = resolved_isa_from_record(record)
    with pytest.raises(UnsupportedStateEffects, match="unknown declared effects"):
        effect_contract_from_isa((model,), admitted_instructions=("alpha.issue",), atomic_instructions=("alpha.issue",))


def test_semantic_identity_pins_all_projection_and_math_sources():
    from merlin.verify.lean_state_effects import effect_semantic_identity

    identity = effect_semantic_identity()
    assert len(identity) == 64 and identity == effect_semantic_identity()


def test_actual_lean_obligations_and_false_frame_control(tmp_path):
    import os
    import subprocess
    from pathlib import Path

    from merlin.common.paths import repo_root

    executable = os.environ.get("MERLIN_STATE_EFFECTS_LEAN")
    if not executable:
        pytest.skip("explicit original Lean executable not selected")
    executable = Path(executable)
    assert executable.is_absolute() and executable.is_file()
    library = tmp_path / "lib"
    (library / "Merlin").mkdir(parents=True)
    math = repo_root() / "src" / "merlin" / "verify" / "lean" / "Merlin" / "StateEffects.lean"
    retained_math = tmp_path / "StateEffects.lean"
    retained_math.write_bytes(math.read_bytes())
    subprocess.run(
        [str(executable), "-o", str(library / "Merlin" / "StateEffects.olean"), str(retained_math)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    source = _source()
    projection = derive_effect_schedule(
        source, _reorder(source, (0, 2, 1, 3, 4, 5)), entry="kernel", contract=_contract()
    )
    statements, proofs = projection.lean_obligations()
    query = "import Merlin.StateEffects\n" + "\n".join(
        f"theorem actual_exchange_{index} : {statement} := {proof}\n#print axioms actual_exchange_{index}"
        for index, (statement, proof) in enumerate(zip(statements, proofs, strict=True))
    )
    path = tmp_path / "ActualSchedule.lean"
    path.write_text(query)
    environment = dict(os.environ, LEAN_PATH=str(library))
    checked = subprocess.run(
        [str(executable), str(path)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    assert "sorryAx" not in checked.stdout
    # Make the actual queried footprint conflict: the closed independence proof
    # must fail in Lean, irrespective of a caller's desired verified status.
    nodes = list(projection.nodes)
    nodes[2] = replace(nodes[2], writes=nodes[1].writes)
    forged = replace(projection, nodes=tuple(nodes))
    statements, proofs = forged.lean_obligations()
    path.write_text("import Merlin.StateEffects\n" + f"theorem false_exchange : {statements[0]} := {proofs[0]}\n")
    refused = subprocess.run(
        [str(executable), str(path)], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30
    )
    assert refused.returncode != 0 and "error" in refused.stdout
