"""Mixed-width original exits share a live LHS through actual tail stores."""

from copy import deepcopy
from dataclasses import replace

import pytest
import yaml
from test_compact_dense_group import run, schedule, selected_fixture
from test_dense_integer_readout import QUALIFICATION, SCALAR_IR

from merlin.compile.scheduling.block_schedule import Store
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Ptr, as_expr


def mixed_fixture(tmp_path, *, mutate=None):
    def declare(facts, software, dense):
        facts["output_bits"] = 8
        facts["codes"].append(5)
        narrow = deepcopy(software["isa"]["instructions"][-1])
        narrow["name"] = "store_i8"
        narrow["encoding"][0]["fixed"]["fact"][-1] = 5
        software["isa"]["instructions"].append(narrow)
        dense["output_readout"] = dict(
            schema="merlin.dense_integer_readout.v1",
            scalar_ir=SCALAR_IR,
            output=dict(dtype="i8", bits={"fact": ["facts", "output_bits"]}),
            instruction="store_i8",
        )
        if mutate is not None:
            mutate(facts, software, dense)

    selected, model = selected_fixture(tmp_path, mutate=declare)
    raw_software = yaml.safe_load(selected.software_bytes)
    raw_software["scheduling"]["dense"].pop("output_readout")
    raw_path = tmp_path / "raw-software.yaml"
    raw_path.write_text(yaml.safe_dump(raw_software))
    raw = read_scheduling_contracts(selected.facts_path, raw_path, target=selected.target)
    readout = selected.dense.output_readout
    original_encoder, original_projector = model[-2:]

    def encode(member, operation):
        call = original_encoder(member, operation)
        if type(operation) is Store and operation.readout is not None:
            assert member == 0 and operation.readout == readout.identity
            p = model[0][member]
            pointer = Ptr(f"output{member}", as_expr(operation.dram_row) * p.n + operation.dram_col)
            return replace(
                call, instr="store_i8", args=tuple((k, pointer if k == "buffer" else v) for k, v in call.args)
            )
        return call

    def project(call):
        member, operation = original_projector(replace(call, instr="store") if call.instr == "store_i8" else call)
        if call.instr == "store_i8":
            operation = replace(operation, readout=readout.identity)
        return member, operation

    return (selected, raw), (*model[:-2], encode, project), (readout, None)


def compile_mixed(contracts, model, readouts, **kwargs):
    return schedule(
        model,
        selected_contracts=contracts,
        output_readouts=readouts,
        readout_qualification_identities=(QUALIFICATION, None),
        **kwargs,
    )


def test_mixed_exits_preserve_all_tail_values_and_one_lhs(tmp_path):
    contracts, model, readouts = mixed_fixture(tmp_path)
    baseline = schedule(model, selected_contracts=contracts)
    result = compile_mixed(contracts, model, readouts)
    assert result.output_dtypes == ("i8", "i32")
    assert result.output_shapes == ((5, 5), (5, 9))
    assert result.realization["requested_lhs_bytes"] == baseline.realization["requested_lhs_bytes"] == 25
    assert result.realization["requested_output_bytes"] == 25 + 45 * 4
    assert baseline.realization["requested_output_bytes"] == (25 + 45) * 4

    # The test machine decodes the actual emitted instruction; the independent
    # reference applies each original member's mathematical exit separately.
    def clamp(x):
        return max(0, min(127, x))

    assert run(result.kernel, model, output_transforms={"store_i8": clamp}, expected_transforms={0: clamp}) == 25
    assert result.realization["output_readouts"] == (readouts[0].identity, None)
    assert (
        tuple(record["software_spec"]["sha256"] for record in result.realization["pipeline"]["selected_contracts"])[0]
        != result.realization["pipeline"]["selected_contracts"][1]["software_spec"]["sha256"]
    )
    result.revalidate()


def test_member_readout_cannot_borrow_other_original_declaration(tmp_path):
    contracts, model, readouts = mixed_fixture(tmp_path)
    with pytest.raises(ValueError, match="original source declaration"):
        compile_mixed(tuple(reversed(contracts)), model, readouts)
    with pytest.raises(ValueError, match="original source declaration"):
        compile_mixed(contracts, model, (replace(readouts[0], scalar_ir=SCALAR_IR.replace("127", "126")), None))


def test_wrong_actual_readout_and_member_source_mutation_refuse(tmp_path):
    contracts, model, readouts = mixed_fixture(tmp_path)
    result = compile_mixed(contracts, model, readouts)
    original = model[-1]

    def corrupt(call):
        member, operation = original(call)
        return member, replace(operation, readout=None) if type(operation) is Store else operation

    with pytest.raises(ValueError, match="readout"):
        compile_mixed(contracts, (*model[:-1], corrupt), readouts)
    contracts[1].software_path.write_bytes(contracts[1].software_bytes + b"\n")
    with pytest.raises(ValueError, match="changed"):
        result.revalidate()


@pytest.mark.parametrize("mutation", ["missing", "space", "integer"])
def test_readout_pointer_must_have_its_original_full_output_binding(tmp_path, mutation):
    def corrupt(facts, software, dense):
        instruction = software["isa"]["instructions"][-1]
        if mutation == "missing":
            instruction["operands"] = [a for a in instruction["operands"] if a["name"] != "buffer"]
            instruction["encoding"] = [f for f in instruction["encoding"] if f.get("argument") != "buffer"]
        else:
            pointer = next(a for a in instruction["operands"] if a["name"] == "buffer")
            if mutation == "space":
                pointer["type"]["space"] = "input_lhs"
            else:
                pointer["type"] = dict(kind="integer", signed=True, bits={"fact": ["facts", "arg_bits"]})

    with pytest.raises(ValueError, match="readout pointer|uncovered|cover"):
        mixed_fixture(tmp_path, mutate=corrupt)


def test_low_level_typed_effect_proof_does_not_grant_source_selection(tmp_path):
    from merlin.compile.scheduling.compact_dense_group import emit_compact_dense_group
    from merlin.compile.scheduling.compact_dense_group_check import verify_compact_dense_group
    from merlin.transforms.dense_planning import DenseResources

    contracts, model, readouts = mixed_fixture(tmp_path)
    storages = model[2]
    views = (storages[0].lhs, *(s.rhs for s in storages), *(s.accumulator for s in storages))
    resources = DenseResources(
        sum(v.capacity * v.unit_bytes for v in views[:3]),
        sum(v.capacity * v.unit_bytes for v in views[3:]),
        3,
        3,
        2,
        partial_output_tiles=True,
        partial_output_blocks=True,
    )
    model = (model[0], resources, *model[2:])
    raw = schedule(model)
    with pytest.raises(ValueError, match="original selected semantics"):
        schedule(model, output_readouts=readouts, readout_qualification_identities=(QUALIFICATION, None))
    typed = emit_compact_dense_group(
        model[0],
        raw.choice.blocks,
        storages,
        encode_member=model[5],
        name="conditional",
        dtypes=("i8", "i8", "i32"),
        output_readouts=readouts,
    )
    proof = verify_compact_dense_group(
        typed,
        model[0],
        resources,
        storages,
        model[3],
        project_member=model[6],
        retention_law=model[4],
        dtypes=("i8", "i8", "i32"),
        static_budget=160,
        external_buffers_disjoint=True,
        output_readouts=readouts,
        readout_qualification_identities=(QUALIFICATION, None),
        timeout_ms=30000,
    )
    assert proof["readout_source_binding"] == "caller_owned_conditional"
    assert proof["output_dtypes"] == ("i8", "i32")
    assert proof["requested_lhs_bytes"] == 25
