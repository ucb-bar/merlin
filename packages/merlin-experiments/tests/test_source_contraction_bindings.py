"""Current typed source and complete plan names precede executed-effect checks."""

import copy
import hashlib

import pytest
from merlin_experiments.phase0.connected_integer_bounds import derive
from test_connected_integer_bounds import program

from merlin.common import mlir_query as mq
from merlin.llvmlower.source_contraction_bindings import bind_rank_two_contractions


def fixture(tmp_path):
    path = tmp_path / "source.mlir"
    text = program()
    path.write_text(text)
    proof = derive(mq.parse(text), [(-3, 3), (-3, 3)])
    proof["interface_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    names = ["initial", "Y0", "empty", "Y1"]
    plan = dict(
        schema="mixed_program_plan_v1",
        source_sha256=proof["interface_sha256"],
        source_op_count=5,
        entry_bindings=["arg0", "arg1"],
        source_values=[
            dict(op_index=i, result_index=0, tensor=name) for i, name in zip(range(1, 5), names, strict=True)
        ],
        output_bindings=["Y0", "Y1"],
        compiler_temporaries=[],
        schedule_instruction_count=5,
        prologue_instruction_range=[0, 0],
        epilogue_instruction_range=[5, 5],
        tasks=[
            dict(
                task_index=i,
                kind="device" if i == 2 else "host",
                source_op_indices=[i],
                instruction_start=i,
                instruction_end=i + 1,
                reads=[],
                writes=[] if i == 0 else [names[i - 1]],
            )
            for i in range(5)
        ],
    )
    cb = dict(
        tensors={
            "arg0": dict(shape=[2, 3], dtype="i8", role="input"),
            "arg1": dict(shape=[3, 2], dtype="i8", role="input"),
            **{
                name: dict(
                    shape=[2, 2],
                    dtype="i8" if i > 1 else "i32",
                    role="output" if name.startswith("Y") else "intermediate",
                )
                for i, name in enumerate(names)
            },
        },
        kernel_abi=dict(
            kind="whole_program",
            outputs=["Y0", "Y1"],
            args=[
                dict(tensor=name, access="read" if name.startswith("arg") else "write")
                for name in ["arg0", "arg1", *names]
            ],
        ),
        params=dict(global_program_plan=plan),
    )
    return path, cb, proof


def test_exact_source_rank_two_contraction_names(tmp_path):
    source, cb, proof = fixture(tmp_path)
    assert bind_rank_two_contractions(source, cb, proof) == (
        dict(ordinal=2, lhs="arg0", rhs="arg1", output="Y0", m=2, k=3, n=2),
    )


@pytest.mark.parametrize("mutation", ["source", "plan", "initial", "arithmetic"])
def test_stale_source_names_or_scalar_recurrence_refuse(tmp_path, mutation):
    source, cb, proof = fixture(tmp_path)
    if mutation == "source":
        source.write_text(source.read_text() + "\n")
    elif mutation == "plan":
        cb["params"]["global_program_plan"]["source_values"][1]["tensor"] = "empty"
    elif mutation == "initial":
        proof["operations"][2]["initial_interval"] = [1, 1]
    else:
        source.write_text(source.read_text().replace("arith.addi %acc, %p", "arith.subi %acc, %p"))
        proof["interface_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        cb["params"]["global_program_plan"]["source_sha256"] = proof["interface_sha256"]
    with pytest.raises(ValueError):
        bind_rank_two_contractions(source, cb, proof)


def test_proof_and_source_are_not_mutated(tmp_path):
    source, cb, proof = fixture(tmp_path)
    before = source.read_bytes(), copy.deepcopy(cb), copy.deepcopy(proof)
    bind_rank_two_contractions(source, cb, proof)
    assert (source.read_bytes(), cb, proof) == before
