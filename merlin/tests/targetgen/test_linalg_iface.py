"""The linalg-on-tensors interface reader (:mod:`merlin.targetgen.contract.linalg_iface`).

This is the second frozen input grammar the experiment ABI hands a backend package (alongside
``merlin_iface`` v0.1). The reader must:

* parse every shipped linalg-on-tensors capsule structurally (xDSL, no regex, no textual fixups);
* surface a faithful op inventory (provenance, operand/result shapes+dtypes, matmul extents, the inner
  arithmetic op names that name softmax/rmsnorm/elementwise semantics, and the dataflow DAG);
* discriminate the two grammars so a package can route between them;
* drive a valid command buffer for the matmul family (the highest-leverage slice — it reuses the
  existing residency path with no new opcodes).
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.contract.linalg_iface import (
    is_linalg_on_tensors, matmul_records, parse_linalg_mlir)

_CAPS = repo_root() / "merlin" / "contract" / "capsules" / "radiance"


def _iface(rel: str) -> str:
    return (_CAPS / rel / "capsule.interface.mlir").read_text(encoding="utf-8")


def _all_linalg_capsules() -> list[str]:
    out = []
    for cy in sorted(_CAPS.rglob("capsule.interface.mlir")):
        if is_linalg_on_tensors(cy.read_text(encoding="utf-8")):
            out.append(str(cy.parent.relative_to(_CAPS)))
    return out


def test_discovers_the_shipped_linalg_corpus():
    caps = _all_linalg_capsules()
    # The known linalg-on-tensors model-slice capsules (RP4..RP18) plus the two hidden ones.
    assert len(caps) >= 17, caps
    assert any("RP15_fused_matmul_bias" in c for c in caps)
    assert any("RPH2_softmax" in c for c in caps)


def test_discriminator_distinguishes_the_two_grammars():
    # linalg-on-tensors carries the prov.level marker ...
    assert is_linalg_on_tensors(_iface("model_slices/RP15_fused_matmul_bias_bf16_pt"))
    # ... a merlin_iface v0.1 capsule (R0 gemm) does not.
    assert not is_linalg_on_tensors(_iface("isa/R0_gemm_fp32"))


@pytest.mark.parametrize("rel", _all_linalg_capsules())
def test_every_linalg_capsule_parses_to_a_clean_inventory(rel):
    parsed = parse_linalg_mlir(_iface(rel))
    assert parsed["level"] == "linalg-on-tensors"
    assert parsed["entry"] == "forward"
    assert parsed["args"] and all(a["shape"] and a["dtype"] for a in parsed["args"])
    assert parsed["results"] and all(r["dtype"] for r in parsed["results"])
    assert parsed["ops"], "no payload ops surfaced"
    # No structural init/terminator op leaks into the payload list (would double-count body ops).
    leaked = [o["kind"] for o in parsed["ops"]
              if o["kind"] in ("linalg.yield", "func.return", "tensor.empty",
                               "arith.constant", "linalg.fill", "tensor.splat")]
    assert not leaked, leaked
    # Every op carries a dataflow source for each input (arg / prior-op / init / const).
    for o in parsed["ops"]:
        for inp in o["ins"]:
            assert inp["source"][0] in ("arg", "op", "init", "const", "other")


def test_matmul_family_extents_are_derived():
    # plain 2D matmul
    rp15 = parse_linalg_mlir(_iface("model_slices/RP15_fused_matmul_bias_bf16_pt"))
    mm = matmul_records(rp15)
    assert len(mm) == 1 and mm[0]["extents"] == {"m": 16, "k": 16, "n": 16}
    # batched (gemv) expressed as a generic still classified as a contraction with a batch extent
    rp10 = parse_linalg_mlir(_iface("model_slices/RP10_gemv_batched_fp16_pt"))
    mm10 = matmul_records(rp10)
    assert mm10 and mm10[0]["extents"]["batch"] == 2 and mm10[0]["extents"]["m"] == 16
    # a chained matmul surfaces two contraction records
    rp17 = parse_linalg_mlir(_iface("model_slices/RP17_k_chain_fp16_pt"))
    assert len(matmul_records(rp17)) == 2


def test_elementwise_and_reduction_semantics_are_named():
    # softmax decomposes into a max-reduce, a broadcast subtract, math.exp, a sum-reduce, a divide.
    sm = parse_linalg_mlir(_iface("model_slices/RP4_softmax_fp32_pt"))
    bodies = [tuple(o["body_ops"]) for o in sm["ops"]]
    assert any("arith.maximumf" in b for b in bodies)
    assert any("math.exp" in b for b in bodies)
    assert any("arith.divf" in b for b in bodies)
    # the reduce over the row dimension is surfaced with its reduction dimension
    reduces = [o for o in sm["ops"] if o["kind"] == "linalg.reduce"]
    assert reduces and all(r["reduction_dims"] == [1] for r in reduces)
    # rmsnorm names its square / rsqrt / multiply
    rms = parse_linalg_mlir(_iface("model_slices/RP13_gemma_4norm_bf16_pt"))
    rbodies = [op for o in rms["ops"] for op in o["body_ops"]]
    assert "math.rsqrt" in rbodies and "math.powf" in rbodies


def test_matmul_family_lowers_to_a_command_buffer_that_executes():
    """End-to-end: the reader's inventory drives a residency command buffer whose simulated output
    matches an independent numpy matmul+bias — proving the matmul-family linalg slice is passable
    through the existing command-buffer path with no new opcodes."""
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.runtime.reference import reference_outputs
    from merlin.runtime.simulator import simulate

    parsed = parse_linalg_mlir(_iface("model_slices/RP15_fused_matmul_bias_bf16_pt"))
    mm = matmul_records(parsed)[0]
    assert mm["extents"] == {"m": 16, "k": 16, "n": 16}

    cb = {
        "abi_version": "0.1", "target": "radiance",
        "tensors": {
            "X": {"shape": parsed["args"][0]["shape"], "dtype": "i8", "role": "input"},
            "W": {"shape": parsed["args"][1]["shape"], "dtype": "i8", "role": "weight"},
            "B": {"shape": parsed["args"][2]["shape"], "dtype": "i32", "role": "bias"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "Wp"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "X", "rhs": "Wp", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y", "bias": "B"},
             "attributes": {"epilogue": ["bias_add"], "output_dtype": "i32"}},
        ],
        "outputs": ["Y"],
    }

    sim = simulate(cb)["outputs"]["Y"]
    ref = reference_outputs(cb)["Y"]
    env = materialize_inputs(cb)
    X = np.array(env["X"].to_list(), dtype=np.int64)
    W = np.array(env["W"].to_list(), dtype=np.int64)
    B = np.array(env["B"].to_list(), dtype=np.int64)
    expected = (X @ W + B).tolist()
    assert sim == ref == expected
