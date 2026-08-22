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


def test_classes_carry_derived_roles_not_hardcoded_names():
    """OV13#2: every class carries a semantic ROLE derived structurally from its operand datapath — the
    MXU/tensor structural checks select BY ROLE, not by a hardcoded pattern name. Assert the real atlas
    classes derive the expected roles and role_classes picks the matmul/memory classes by role."""
    tax = _atlas_taxonomy()
    role_of = {c: next((e.get("role") for e in ents if e.get("role")), None)
               for c, ents in tax["by_class"].items()}
    assert role_of.get("MXUMatMul") == "matmul"
    assert role_of.get("TensorBaseOffset") == "memory"
    assert role_of.get("MXUWeightPush") == "weight_load"
    assert role_of.get("MXUAccumulatorPop") == "acc_readout"
    assert role_of.get("MXUAccumulatorPopE1") == "acc_readout_scaled"
    assert role_of.get("TensorComputeUnary") == "tensor_compute_unary"
    rc = IT.role_classes(tax)
    assert rc == {"compute": "MXUMatMul", "memory": "TensorBaseOffset"}


def test_role_selection_is_name_independent():
    """The selectors key on the derived ROLE only — a target whose ISA names its patterns ANYTHING still
    resolves, proving there is no atlas-name overfit left. Synthetic taxonomy, invented class names."""
    tax = {"by_class": {
        "OpFoo":  [{"role": "memory"}],
        "OpBar":  [{"role": "weight_load"}],
        "OpBaz":  [{"role": "matmul"}],
        "OpQux":  [{"role": "acc_readout"}],
        "OpQuxS": [{"role": "acc_readout_scaled"}],
        "OpRelu": [{"role": "tensor_compute_unary"}],
    }}
    assert IT.role_classes(tax) == {"compute": "OpBaz", "memory": "OpFoo"}
    assert IT.required_classes_for_op(tax, op="matmul", output_dtype="bf16") == \
        ["OpFoo", "OpBar", "OpBaz", "OpQux"]
    assert IT.required_classes_for_op(tax, op="matmul", output_dtype="fp8_e4m3")[-1] == "OpQuxS"
    assert IT.required_classes_for_op(tax, op="matmul", output_dtype="bf16", epilogue=("relu",))[-1] == "OpRelu"
    assert IT.required_classes_for_op(tax, movement=True) == ["OpFoo"]
    # a taxonomy with no matmul role -> a matmul op yields no systolic classes (honest, non-fabricated)
    assert IT.role_classes({"by_class": {"OpX": [{"role": "scalar"}]}}) == {"compute": None, "memory": None}


def test_asm_mnemonic_of_reads_the_op_class_classvar():
    """The per-op assembler-mnemonic fallback reads a class's OWN declared syntax (``mnemonic``/``asm``/…),
    fail-closed on NotImplemented/non-str/absent — so a spec that names its ops but exposes no container
    ``operations`` dict still yields an asm map. Pure/hermetic (no model venv)."""
    from merlin.targetgen.oracle_helpers.isa_introspect import _asm_mnemonic_of

    class WithMnemonic:
        mnemonic = "vmatmul.mxu0"

    class WithAsm:
        asm = "dma.config"

    class Unnamed:
        mnemonic = NotImplemented          # the ClassVar sentinel — not yet named

    class NonStr:
        mnemonic = 123

    class Bare:
        pass

    assert _asm_mnemonic_of(WithMnemonic) == "vmatmul.mxu0"
    assert _asm_mnemonic_of(WithAsm) == "dma.config"
    assert _asm_mnemonic_of(Unnamed) is None
    assert _asm_mnemonic_of(NonStr) is None
    assert _asm_mnemonic_of(Bare) is None


def test_atlas_asm_mnemonics_and_reference_kernel_classes_derive():
    """Regression: the derived taxonomy populates ``asm_mnemonics`` from each op class's own mnemonic (even
    though the standalone ``isa_definition.py`` load exposes no reachable ``operations`` container), so an
    example kernel written in the target's real assembler syntax maps back to semantic classes. Previously
    ``asm_mnemonics`` was empty and ``classes_from_kernel`` returned []. Gated on the model venv."""
    tax = _atlas_taxonomy()
    asm = tax.get("asm_mnemonics") or {}
    assert asm, "asm_mnemonics empty — the per-op mnemonic fallback did not populate the map"
    # a class-name key resolves back to its own class (derived, not hardcoded)
    assert asm.get("vmatmul.mxu0") == "VMATMUL_MXU0"
    # every shipped reference kernel now yields a non-empty, matmul-bearing class sequence
    kernels = IT._example_kernels(load_target_experiment(_ATLAS))
    assert kernels, "no shipped atlas example kernels"
    for k in kernels:
        classes = IT.classes_from_kernel(k.read_text(), tax)
        assert classes, f"{k.name}: classes_from_kernel returned [] (asm map not applied)"
        assert "MXUMatMul" in classes, f"{k.name}: matmul class missing from {classes}"


def _atlas_binding():
    """The same per-target binding the corpus generator uses (carries the class deriver)."""
    import yaml as _y
    from merlin.targetgen import corpus_spec as _CS
    from merlin.targetgen.target_experiment import load_target_experiment as _lte
    prof = merlin_dir() / "contract/capsules/profiles/atlas.yaml"
    datapath = (_y.safe_load(prof.read_text()) or {}).get("datapath") or {}
    return _CS.derive_binding(_lte(_ATLAS), datapath)


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
        got = (doc.get("expected") or {}).get("instruction_classes")

        if doc.get("kind") == "model":
            # A WHOLE-MODEL capsule is derived differently, and must be — ``model`` is not a semantic
            # family and never will be, so ``required_classes_for_op(op="model")`` correctly yields
            # nothing. Its demand comes from the MODEL's own captured linalg crossed with this target's
            # capabilities and role census. Re-deriving it the same way the generator does keeps the
            # derived-and-enforced property for the capstone instead of exempting the one capsule the
            # whole suite builds toward.
            from merlin.targetgen.capsule_source import model_accelerator_demand
            lin = cy.parent / str(doc.get("linalg_mlir") or doc.get("interface_mlir") or "")
            if not lin.is_file():
                continue
            _fam, want = model_accelerator_demand(lin.read_text(), _atlas_binding())
            assert got == want, f"{cy.parent.name}: corpus classes {got} != model-derived {want}"
            continue

        want = IT.required_classes_for_op(tax, op=op, output_dtype=out_dt,
                                          epilogue=tuple(attrs.get("epilogue", []) or []), movement=movement)
        assert got == want, f"{cy.parent.name}: corpus classes {got} != derived {want}"
        # the fabricated taxonomy must be gone
        assert not ({"CONFIG_EX", "GMEM_LD", "FMA", "GMEM_ST"} & set(got or []))
