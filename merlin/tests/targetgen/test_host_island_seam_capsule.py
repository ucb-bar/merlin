"""A host-island differential is one honest mixed program, never harness-side arithmetic.

Both members contain the same two contractions over the same external tensors.  The ``island``
member alone carries an explicit scalar-only integer map between them; the ``no_island`` member wires
the first contraction directly into the second.  The standard linalg surface makes this portable to
any backend that declares the generic whole-program ABI capability.
"""
from __future__ import annotations

from merlin.runtime.tensor import Tensor
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.contract.linalg_iface import parse_linalg_mlir
from merlin.targetgen.contract.linalg_iface import make_linalg_context
from xdsl.parser import Parser


def _binding() -> CS.CorpusBinding:
    return CS.CorpusBinding(
        target="fixture", tile_dim=4, operand_dtype="int8", accum_dtype="i32",
        integer=True, tiers=["L0", "L1", "L2", "L3"], compare="exact_int",
        classes_for=lambda **_: ["CONTRACTION"],
    )


def _entry(role: str) -> dict:
    return {
        "name": f"PB_{role}", "kind": "model_slice", "source_role": "derived_sweep",
        "source_reference": "paired host-island fixture", "label": "dev",
        "op": "host_island_seam", "M": 4, "K": 8, "H": 4, "N": 4,
        "comparison_role": role, "host_transform": ("xor_low_bit" if role == "island" else "none"),
        "xor_mask": 1,
        "semantic": {"semantic_family": "contraction", "generalization_axis": "composition",
                     "must_accelerate": True},
    }


def test_pair_has_identical_external_work_and_only_one_host_map() -> None:
    island, island_mlir = CS.build(_entry("island"), _binding())
    control, control_mlir = CS.build(_entry("no_island"), _binding())

    assert island["operation"]["op"] == control["operation"]["op"] == "host_island_seam"
    assert island["inputs"] == control["inputs"]
    assert island["operation"]["attributes"]["accelerator_contractions"] == 2
    assert control["operation"]["attributes"]["accelerator_contractions"] == 2

    iops = parse_linalg_mlir(island_mlir)["ops"]
    cops = parse_linalg_mlir(control_mlir)["ops"]
    assert [op["family"] for op in iops] == [
        "contraction", "elementwise_map", "host_scalar", "contraction"]
    assert [op["family"] for op in cops] == [
        "contraction", "elementwise_map", "contraction"]
    assert iops[2]["body_ops"] == ["arith.xori"]
    assert "arith.xori" not in control_mlir
    # Parsing an inventory is weaker than verifying the actual IR. In particular, named matmul's
    # xDSL verifier rejects mixed i8-input/i32-output types, so the widening contractions are explicit
    # linalg.generic regions and both full modules must verify before a backend ever sees them.
    Parser(make_linalg_context(), island_mlir).parse_module().verify()
    Parser(make_linalg_context(), control_mlir).parse_module().verify()


def test_exact_golden_computes_the_host_map_between_the_two_contractions() -> None:
    island, _ = CS.build(_entry("island"), _binding())
    control, _ = CS.build(_entry("no_island"), _binding())
    env = CG.materialize_capsule_leaves(island)

    mid = env["A0"].matmul(env["W0"]).to_i8()
    transformed = Tensor(mid.shape, [x ^ 1 for x in mid.data], "i8")
    expected_island = transformed.matmul(env["W1"]).to_list()
    expected_control = mid.matmul(env["W1"]).to_list()

    assert CG.golden(island) == {"Y0": expected_island}
    assert CG.golden(control) == {"Y0": expected_control}
    assert expected_island != expected_control
