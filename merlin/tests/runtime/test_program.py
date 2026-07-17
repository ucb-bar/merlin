"""``runtime.program.build_program`` — the kernel table + memory plan the C replay consumes.

Regression focus (S0 provenance spine): the dispatch node's provenance keys are the model2MLIR
``prov.*`` spellings (``prov.op`` / ``prov.fqn`` / ``prov.region_id``). ``build_program`` must carry
that model-layer identity onto each :class:`KernelEntry` so an emitted kernel can be resolved back to
its original layer (the join key the cross-compiler compare and the section slicer both use).
"""
from __future__ import annotations

from merlin.runtime.program import build_program
from merlin.xdsl_dialects.lowering.dispatch_program import Buffer, DispatchProgram, Node


def _one_dispatch_program(prov: dict[str, str]) -> DispatchProgram:
    """A minimal well-formed 1-dispatch DAG: two args -> one kernel -> one result."""
    buffers = {
        "b0": Buffer(id="b0", shape=[4, 8], dtype="f32", kind="arg", arg_index=0),
        "b1": Buffer(id="b1", shape=[8, 6], dtype="f32", kind="arg", arg_index=1),
        "b2": Buffer(id="b2", shape=[4, 6], dtype="f32", kind="intermediate"),
    }
    node = Node(kind="dispatch", op="forward$kernel_0", inputs=["b0", "b1"],
                outputs=["b2"], prov=prov)
    return DispatchProgram(entry="forward", args=[0, 1], buffers=buffers,
                           nodes=[node], results=["b2"])


def test_kernel_entry_carries_model_layer_provenance():
    # The prov dict as it actually arrives from the outliner (prov.* spellings).
    prov = {"prov.op": "matmul", "prov.fqn": "blocks.0.attn.q", "prov.region_id": "matmul_0"}
    prog = build_program(_one_dispatch_program(prov), capability="rvv")
    entry = prog.kernels["forward$kernel_0"]
    # roots was ALWAYS [] before the fix (the old filter matched bare root/op/name, never prov.*).
    assert entry.roots == ["matmul"]
    assert entry.region_id == "matmul_0"
    assert entry.fqn == "blocks.0.attn.q"
    assert entry.capability == "rvv"


def test_kernel_entry_defaults_when_prov_absent():
    # A pre-provenance capture (no prov.*) must not crash and must leave the fields empty.
    prog = build_program(_one_dispatch_program({}), capability="scalar")
    entry = prog.kernels["forward$kernel_0"]
    assert entry.roots == []
    assert entry.region_id == "" and entry.fqn == ""


def test_program_serializes_with_new_fields():
    prov = {"prov.op": "matmul", "prov.fqn": "blocks.0.mlp.g", "prov.region_id": "matmul_3"}
    prog = build_program(_one_dispatch_program(prov), capability="rvv")
    d = prog.to_dict()
    k = d["kernels"]["forward$kernel_0"]
    assert k["region_id"] == "matmul_3" and k["fqn"] == "blocks.0.mlp.g"
