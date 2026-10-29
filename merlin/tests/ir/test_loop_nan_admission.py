"""Current scalar-loop metadata custody; host-only producer/projection checks."""

import json
from pathlib import Path

import pytest

from merlin.common.paths import module_source_path
from merlin.transforms.certificates import ir_text, parse_ir
from merlin.transforms.loops import segment_loops
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_intake as intake
from merlin.verify.gate_errors import GateRejected, GateUnavailable, GateUnsupported


def actual(value, kinds):
    source = f"""builtin.module {{
func.func @forward(%x:i32, %y:i32) -> i32 attributes {{tag = {value}}} {{
  %lb = arith.constant 3 : index
  %ub = arith.constant 31 : index
  %st = arith.constant 2 : index
  %r = scf.for %i = %lb to %ub step %st iter_args(%c = %x) -> (i32) {{
    %a = arith.addi %c, %y : i32
    scf.yield %a : i32
  }}
  func.return %r : i32
}}
}}"""
    module = parse_ir(source)
    buffers = [ir_text(module)]
    if intake.UNROLL in kinds:
        assert unroll_ordered_loops(module, (2,), index_bits=64)
        module.verify()
        buffers.append(ir_text(module))
    if intake.TILE in kinds:
        assert segment_loops(module, (3,), index_bits=64)
        module.verify()
        buffers.append(ir_text(module))
    return tuple(buffers)


@pytest.mark.parametrize(
    "kinds,endpoint",
    [
        ((intake.UNROLL,), 0),
        ((intake.UNROLL,), 1),
        ((intake.TILE,), 0),
        ((intake.TILE,), 1),
        ((intake.UNROLL, intake.TILE), 0),
        ((intake.UNROLL, intake.TILE), 1),
        ((intake.UNROLL, intake.TILE), 2),
    ],
)
@pytest.mark.parametrize("bits", ["0x7F800001", "0x7FC00001"])
def test_scalar_nan_refuses_before_any_actual_loop_profile_or_fallback(kinds, endpoint, bits, monkeypatch):
    buffers = list(actual("0.0 : f32", kinds))
    module = parse_ir(buffers[endpoint])
    poison = parse_ir(f"builtin.module attributes {{tag = {bits} : f32}} {{}}")
    module.body.block.first_op.attributes["tag"] = poison.attributes["tag"]
    buffers[endpoint] = ir_text(module)
    monkeypatch.setattr(
        "merlin.verify.compiler_unroll_modes_intake.prepare",
        lambda *a, **k: pytest.fail("scalar NaN reached mode dispatch"),
    )
    with pytest.raises(intake.Unsupported, match="scalar-nan"):
        intake.prepare(tuple(buffers), kinds=kinds, factor=2, tile=3)


@pytest.mark.parametrize("kinds", [(intake.UNROLL,), (intake.TILE,), (intake.UNROLL, intake.TILE)])
def test_actual_loop_dense_nan_bytes_keep_positive_and_changed_target_refusal(kinds):
    first = 'dense<"0x0100807F"> : tensor<1xf32>'
    buffers = actual(first, kinds)
    prepared = intake.prepare(buffers, kinds=kinds, factor=2, tile=3)
    assert prepared.actual_target == buffers[-1]
    changed = (*buffers[:-1], buffers[-1].replace("0100807F", "0100C07F"))
    assert changed[-1] != buffers[-1]
    with pytest.raises(ValueError, match="interface/attributes changed"):
        intake.prepare(changed, kinds=kinds, factor=2, tile=3)
    required = {
        module_source_path("merlin.verify.memory_scalar_nan_admission").resolve(),
        module_source_path("merlin.verify.lean_graph_input_admission").resolve(),
        module_source_path("xdsl.dialects.builtin").resolve(),
    }
    assert required <= {Path(row["path"]).resolve() for row in json.loads(prepared.sources_json)}


@pytest.mark.parametrize(
    "value", ["0x7F800000 : f32", "0xFF800000 : f32", "0x00000000 : f32", "0x80000000 : f32", '"nan"']
)
def test_actual_loop_infinity_zero_and_string_metadata_remain_admitted(value):
    prepared = intake.prepare(actual(value, (intake.UNROLL,)), kinds=(intake.UNROLL,), factor=2)
    assert prepared.actual_target == prepared.buffers[1]


def _handler_job(tmp_path, buffers):
    before, after = tmp_path / "before.mlir", tmp_path / "after.mlir"
    before.write_text(buffers[0])
    after.write_text(buffers[-1])
    return {
        "before": str(before),
        "after": str(after),
        "directory": str(tmp_path / "proof"),
        "params": {
            "intermediates": list(buffers[1:-1]),
            "kinds": [intake.UNROLL],
            "factor": 2,
            "tile": 3,
            "index_bits": 64,
            "sdk_bin": str(tmp_path / "never-opened-SDK"),
        },
    }


def test_public_loop_handler_excludes_nan_source_before_any_proof_check(tmp_path, monkeypatch):
    from merlin.verify import compiler_transforms, loop_transform_gate

    buffers = actual("0x7F800001 : f32", (intake.UNROLL,))
    monkeypatch.setattr(loop_transform_gate, "check", lambda *a, **k: pytest.fail("excluded input reached SDK"))
    with pytest.raises(GateUnsupported, match="scalar-nan"):
        compiler_transforms.loop_check(_handler_job(tmp_path, buffers))


def test_public_loop_handler_keeps_dense_budget_exhaustion_unavailable(tmp_path, monkeypatch):
    from merlin.verify import compiler_transforms, loop_transform_gate, memory_scalar_nan_admission

    buffers = actual('dense<"0x0000000000000000"> : tensor<2xf32>', (intake.UNROLL,))
    monkeypatch.setattr(memory_scalar_nan_admission, "MAX_LOOP_DENSE_BYTES", 4)
    monkeypatch.setattr(loop_transform_gate, "check", lambda *a, **k: pytest.fail("resource refusal reached SDK"))
    with pytest.raises(GateUnavailable, match="decoded-storage resource budget"):
        compiler_transforms.loop_check(_handler_job(tmp_path, buffers))


@pytest.mark.parametrize("failure", ["known-proof-false", "missing-SDK"])
def test_public_loop_handler_does_not_relabel_proof_or_tool_failures_as_excluded(tmp_path, monkeypatch, failure):
    from merlin.verify import compiler_transforms, loop_transform_gate

    buffers = actual("0.0 : f32", (intake.UNROLL,))

    def fail(*args, **kwargs):
        if failure == "known-proof-false":
            raise loop_transform_gate.NotEstablished("known closed proof false")
        raise intake.Unavailable("selected SDK unavailable")

    monkeypatch.setattr(loop_transform_gate, "check", fail)
    expected = GateRejected if failure == "known-proof-false" else GateUnavailable
    with pytest.raises(expected):
        compiler_transforms.loop_check(_handler_job(tmp_path, buffers))
