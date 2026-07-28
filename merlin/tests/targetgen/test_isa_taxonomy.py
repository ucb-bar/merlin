"""AW6 Layer 2: the atlas instruction taxonomy is DERIVED from the repo's ISA definition, not hardcoded.

For a self-hosted-ISA core mlc's behavioural role probe is RoCC-only, so the authoritative op taxonomy
comes from introspecting the shipped ``isa_definition.py``. These tests assert the derivation yields the
REAL atlas op classes (MXU systolic datapath + tensor load/store) and that a matmul's required classes are
selected from them — NOT the fabricated CONFIG_EX/GMEM_LD/FMA/GMEM_ST set the corpus used to hardcode.
Gated on the model venv (npu_model) being present.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import isa_taxonomy as IT
from merlin.targetgen.target_experiment import load_target_experiment
from merlin.common.paths import merlin_dir

_ATLAS = merlin_dir() / "experiments/capsule_bench/targets/atlas/target_experiment.yaml"


def _atlas_taxonomy():
    if not _ATLAS.is_file():
        pytest.skip("atlas descriptor absent")
    IT.clear_cache()
    tax = IT.derive_isa_taxonomy(load_target_experiment(_ATLAS))
    if not tax or not tax.get("by_class"):
        pytest.skip("atlas ISA taxonomy not derivable (model venv / isa_definition absent)")
    return tax


def test_taxonomy_has_the_real_mxu_datapath_not_the_fabricated_classes():
    tax = _atlas_taxonomy()
    classes = set(tax["by_class"])
    # the REAL atlas MXU systolic datapath + operand load/store, from isa_definition.py
    for real in ("MXUWeightPush", "MXUMatMul", "MXUAccumulatorPop", "TensorBaseOffset"):
        assert real in classes, f"derived taxonomy missing real class {real}: {sorted(classes)}"
    # the fabricated corpus classes are NOT real ISA semantic patterns
    assert not ({"CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"} & classes)
    # mnemonics map back to their class (e.g. VMATMUL_MXU0 -> MXUMatMul)
    assert tax["by_mnemonic"]["VMATMUL_MXU0"]["class"] == "MXUMatMul"


def test_matmul_required_classes_are_derived_from_the_taxonomy():
    tax = _atlas_taxonomy()
    req = IT.required_classes_for_op(tax, op="matmul", output_dtype="bf16")
    # the real systolic sequence: load operands -> push weight -> matmul -> pop accumulator
    assert req[:3] == ["TensorBaseOffset", "MXUWeightPush", "MXUMatMul"]
    assert req[-1] in ("MXUAccumulatorPop", "MXUAccumulatorPopE1")
    # a movement capsule needs the load/store copy but NO MXU compute
    mv = IT.required_classes_for_op(tax, movement=True)
    assert mv == ["TensorBaseOffset"] and not any(c.startswith("MXU") for c in mv)


def test_committed_atlas_corpus_matches_the_live_derivation():
    """The atlas capsules' expected.instruction_classes must EQUAL the live derivation — so the corpus is
    derived-and-enforced (never silently re-hardcoded, and an ISA change surfaces as drift here)."""
    import yaml
    tax = _atlas_taxonomy()
    cap = merlin_dir() / "contract/capsules/atlas"
    if not cap.is_dir():
        pytest.skip("atlas corpus absent")
    caps = sorted(cap.glob("*/*/capsule.yaml"))
    assert caps, "no atlas capsules found"
    for cy in caps:
        doc = yaml.safe_load(cy.read_text())
        op = (doc.get("operation") or {}).get("op", "matmul")
        attrs = (doc.get("operation") or {}).get("attributes", {}) or {}
        modes = (doc.get("expected") or {}).get("modes", {}) or {}
        movement = op in ("movement", "copy") or bool(modes.get("movement"))
        out_dt = attrs.get("output_dtype") or (doc.get("numeric_policy") or {}).get("dtype", "bf16")
        want = IT.required_classes_for_op(tax, op=op, output_dtype=out_dt,
                                          epilogue=tuple(attrs.get("epilogue", []) or []), movement=movement)
        got = (doc.get("expected") or {}).get("instruction_classes")
        assert got == want, f"{cy.parent.name}: corpus classes {got} != derived {want}"
        # the fabricated taxonomy must be gone
        assert not ({"CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"} & set(got or []))
