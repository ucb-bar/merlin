"""Scalar-immediate affine forms (add.Scalar / sub.Scalar / mul.Scalar) of the standalone elementwise map.

The spec row ``elementwise_map_scalar_affine`` declares the forms; ``FORM_TABLE`` says which builder writes each;
the written region has ONE tensor operand and literal immediates, so the existing observation (affine, one
operand, no broadcast) admits it under the reviewed standalone row. The golden is the independent exact evaluator.
"""

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen import capsule_golden
from merlin.targetgen import corpus_form_table as ft
from merlin.targetgen import corpus_spec as SPEC
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves
from merlin.targetgen.corpus_cell_forms import cell_forms
from merlin.targetgen.corpus_synth import available_ops

pytestmark = pytest.mark.target("gemmini")
_FORMS = {
    "scalar_add": (1, 3, "arith.addi"), "scalar_sub": (1, -3, "arith.subi"), "scalar_mul": (3, 0, "arith.muli"),
    "scalar_neg": (-1, 0, "arith.muli"), "cast_widen": (1, 0, "arith.extsi"),
}  # fmt: skip


def _binding():
    return SPEC.CorpusBinding(
        target="t", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip


@pytest.mark.parametrize("form", sorted(_FORMS))
@pytest.mark.parametrize("m,n", [(16, 16), (16, 15), (4, 4)])
def test_form_builds_and_grades_exactly(form, m, n):
    from merlin_experiments.phase0.component_integer_bounds import derive, source_for_capsule

    weight, bias, body_op = _FORMS[form]
    entry = {
        "name": f"T_{form}", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": "scalar_affine", "M": m, "N": n, "lhs": "A0", "out": "Y0", "label": "public",
        **ft.entry_fields("elementwise_map", form, m, n),
    }  # fmt: skip
    assert (entry.get("weight", 1), entry.get("bias", 0)) == (weight, bias)
    capsule, text = SPEC.build(entry, _binding())
    assert capsule["semantic"]["semantic_family"] == "elementwise_map"
    assert body_op in text and ('prov.family = "elementwise"' in text or form == "cast_widen")
    assert text.count("ins(%A0") == 1  # one tensor operand; the immediate is a body literal
    semantics = {"model": {"engine": "integer_reference"}, "overflow": "bounded_exact"}
    capsule["numerical_semantics"] = semantics
    capsule["integer_partial_sum_bound"] = derive(source_for_capsule(capsule), semantics)
    leaves = materialize_capsule_leaves(capsule)["A0"]
    want = [[int(leaves.data[r * n + c]) * weight + bias for c in range(n)] for r in range(m)]
    assert capsule_golden.golden(capsule, None)["Y0"] == want


@pytest.mark.parametrize("form", sorted(_FORMS))
def test_written_region_is_admitted_by_the_reviewed_standalone_row(form):
    from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from phase0_fixture_selection import fixture_resolved

    base = repo_root() / "examples/gemmini/target"
    selection = {
        "spec": fixture_resolved(base, "gemmini"),
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
        "capability_contract": load_yaml(base / "contracts/target_contract.yaml"),
    }
    entry = {
        "name": f"T_{form}", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": "scalar_affine", "M": 16, "N": 16, "lhs": "A0", "out": "Y0", "label": "public",
        **ft.entry_fields("elementwise_map", form, 16, 16),
    }  # fmt: skip
    _capsule, text = SPEC.build(entry, _binding())
    with IR_LOCK:
        sites = non_contraction_sites(mq.parse(text), selection, contract=selection["capability_contract"])
    generics = [s for s in sites if s["source_operation"] == "linalg.generic"]
    assert generics and all(s["status"] == "admitted" for s in generics), [s["reasons"] for s in generics]
    assert generics[0]["signature"]["tensor_operands"] == 1 and generics[0]["signature"]["expression"] == "affine"


def test_declared_forms_come_from_the_reviewed_spec_row():
    spec = load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml")
    from merlin.targetgen.software_spec import validate_software_spec

    doc = validate_software_spec(spec, "gemmini")
    kept, refused = ft.declared_forms(doc, {})
    assert kept.get("elementwise_map") == list(_FORMS), (kept, refused)
    forms = cell_forms("elementwise_map", "i8", None, available_ops(), tuple(kept["elementwise_map"]))
    assert {f.form: f.op for f in forms if f.form} == dict.fromkeys(_FORMS, "scalar_affine")
