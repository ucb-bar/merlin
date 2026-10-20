"""Only the actual original source and live binding may authorize unpublished scratch."""

import copy
import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.targetgen.contract.source_ownership import (
    SourceOwnershipError,
    active_source_owned_mutables,
    source_mutable_scope,
    source_owned_mutables,
)

SOURCE = """module { func.func @forward(%x: tensor<1xi32>) -> tensor<1xi32> {
  %a = arith.addi %x, %x : tensor<1xi32>
  %b = arith.addi %a, %x : tensor<1xi32>
  func.return %b : tensor<1xi32>
} }"""


def _buffer():
    return {
        "tensors": {
            name: {"shape": [1], "dtype": "i32", "role": role}
            for name, role in [("x", "input"), ("tmp", "intermediate"), ("out", "output")]
        },
        "kernel_abi": {
            "kind": "whole_program",
            "args": [
                {"tensor": "x", "access": "read"},
                {"tensor": "tmp", "access": "readwrite"},
                {"tensor": "out", "access": "write"},
            ],
            "outputs": ["out"],
        },
        "params": {
            "global_program_plan": {
                "schema": "mixed_program_plan_v1",
                "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                "source_op_count": 2,
                "schedule_instruction_count": 2,
                "prologue_instruction_range": [0, 0],
                "epilogue_instruction_range": [2, 2],
                "entry_bindings": ["x"],
                "output_bindings": ["out"],
                "compiler_temporaries": [],
                "source_values": [
                    {"op_index": 0, "result_index": 0, "tensor": "tmp"},
                    {"op_index": 1, "result_index": 0, "tensor": "out"},
                ],
                "tasks": [
                    {
                        "task_index": 0,
                        "kind": "mixed",
                        "source_op_indices": [0],
                        "instruction_start": 0,
                        "instruction_end": 1,
                        "reads": ["x"],
                        "writes": ["tmp"],
                    },
                    {
                        "task_index": 1,
                        "kind": "mixed",
                        "source_op_indices": [1],
                        "instruction_start": 1,
                        "instruction_end": 2,
                        "reads": ["tmp", "x"],
                        "writes": ["out"],
                    },
                ],
            }
        },
    }


def test_live_scope_authorizes_scratch_and_is_closed_afterward():
    cb = _buffer()
    assert source_owned_mutables(SOURCE, cb) == ("tmp",)
    assert active_source_owned_mutables(cb) == ()
    with source_mutable_scope(SOURCE, cb):
        assert active_source_owned_mutables(copy.deepcopy(cb)) == ("tmp",)
        with ThreadPoolExecutor(1) as executor:
            assert executor.submit(active_source_owned_mutables, cb).result() == ()
    assert active_source_owned_mutables(cb) == ()


@pytest.mark.parametrize(
    "mutation", ["wrong_source", "read_before_write", "wrong_writer", "hide_output", "changed_type", "entry_readwrite"]
)
def test_roles_alone_cannot_authorize_scratch(mutation):
    cb = _buffer()
    plan = cb["params"]["global_program_plan"]
    if mutation == "wrong_source":
        plan["source_sha256"] = "0" * 64
    elif mutation == "read_before_write":
        plan["tasks"][0]["reads"].append("tmp")
    elif mutation == "wrong_writer":
        plan["tasks"][0]["writes"] = []
        plan["tasks"][1]["writes"].append("tmp")
    elif mutation == "hide_output":
        cb["tensors"]["out"]["role"] = "intermediate"
        cb["kernel_abi"]["outputs"] = []
        plan["output_bindings"] = []
    elif mutation == "entry_readwrite":
        cb["kernel_abi"]["args"][0]["access"] = "readwrite"
    else:
        cb["tensors"]["tmp"]["shape"] = [2]
    with pytest.raises(SourceOwnershipError):
        source_owned_mutables(SOURCE, cb)


def test_live_scope_rejects_binding_changes_and_always_resets():
    cb = _buffer()
    with pytest.raises(SourceOwnershipError, match="changed during execution"):
        with source_mutable_scope(SOURCE, cb):
            cb["tensors"]["tmp"]["shape"] = [2]
            with pytest.raises(SourceOwnershipError, match="differs from original"):
                active_source_owned_mutables(cb)
    assert active_source_owned_mutables(cb) == ()


def test_live_scratch_admission_cannot_be_reused_through_legacy_elf_cache(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from merlin.runtime.backends import base
    from merlin.targetgen import build_cache
    from merlin.targetgen.contract import compile as compiler

    cb = _buffer()
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace())
    for name in ("build_identity", "reuse", "store"):
        monkeypatch.setattr(build_cache, name, lambda *args, **kwargs: pytest.fail("live scratch reached legacy cache"))
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *args, **kwargs: tmp_path / "kernel.o")
    observed = []

    def link(buffer, *args, **kwargs):
        observed.append(active_source_owned_mutables(buffer))
        return tmp_path / "program.elf"

    monkeypatch.setattr(compiler, "link_elf", link)
    with source_mutable_scope(SOURCE, cb):
        assert compiler.compile_lowered_to_elf(cb, "llvm", tmp_path, target="fixture", inputs={"x": [1]})
    assert observed == [("tmp",)]


def test_capsule_scope_reads_original_source_instead_of_candidate_copy(tmp_path):
    from merlin.targetgen.capsule_lane_evidence import oracle_source_scope

    cb = _buffer()
    cb["params"]["whole_source"] = "a candidate-controlled replacement"
    (tmp_path / "source.mlir").write_text(SOURCE)
    capsule = {"__dir__": str(tmp_path), "linalg_mlir": "source.mlir"}
    with oracle_source_scope(capsule, cb, None):
        assert active_source_owned_mutables(cb) == ("tmp",)
    (tmp_path / "source.mlir").write_text(SOURCE.replace("%x, %x", "%x, %a"))
    with pytest.raises(SourceOwnershipError):
        with oracle_source_scope(capsule, cb, None):
            pytest.fail("candidate copy replaced the original source")


def test_capsule_scope_never_accepts_missing_original_source():
    from merlin.targetgen.capsule_lane_evidence import oracle_source_scope

    cb = _buffer()
    cb["params"]["whole_source"] = SOURCE
    with pytest.raises(ValueError, match="original capsule source"):
        oracle_source_scope({}, cb, None)


def test_renderer_receives_only_live_scratch_and_no_alternate_caller_authority():
    from merlin.targetgen.contract.source_ownership import renderer_scratch_options

    cb = _buffer()

    def renderer(cb, *, source_owned_mutables):
        return source_owned_mutables

    assert renderer_scratch_options(cb, renderer) == {}
    with source_mutable_scope(SOURCE, cb):
        assert renderer_scratch_options(cb, renderer) == {"source_owned_mutables": ("tmp",)}
        with pytest.raises(ValueError, match="alternate caller"):
            renderer_scratch_options(cb, renderer, alternate_caller=True)
        with pytest.raises(NotImplementedError, match="cannot consume"):
            renderer_scratch_options(cb, lambda cb: "")
