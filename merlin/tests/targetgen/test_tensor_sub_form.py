"""The two-operand difference form (sub.Tensor) of the standalone elementwise map: weights (1, -1)."""

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen import capsule_golden
from merlin.targetgen import corpus_form_table as ft
from merlin.targetgen import corpus_spec as SPEC
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

pytestmark = pytest.mark.target("gemmini")


def _binding():
    return SPEC.CorpusBinding(
        target="t", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip


def _entry(m, n):
    return {
        "name": "T_sub", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": "tensor_sub", "M": m, "N": n, "out": "Y0", "label": "public",
        **ft.entry_fields("elementwise_map", "tensor_sub", m, n),
    }  # fmt: skip


@pytest.mark.parametrize("m,n", [(16, 16), (16, 15), (4, 4)])
def test_difference_builds_and_grades_exactly(m, n):
    from merlin_experiments.phase0.component_integer_bounds import derive, source_for_capsule

    capsule, text = SPEC.build(_entry(m, n), _binding())
    assert capsule["semantic"]["semantic_family"] == "elementwise_map"
    assert "arith.subi" in text and text.count("ins(%A0, %A1") == 1
    semantics = {"model": {"engine": "integer_reference"}, "overflow": "bounded_exact"}
    capsule["numerical_semantics"] = semantics
    capsule["integer_partial_sum_bound"] = derive(source_for_capsule(capsule), semantics)
    leaves = materialize_capsule_leaves(capsule)
    want = [[int(leaves["A0"].data[r * n + c]) - int(leaves["A1"].data[r * n + c]) for c in range(n)] for r in range(m)]
    assert capsule_golden.golden(capsule, None)["Y0"] == want


def test_written_region_is_admitted_by_the_reviewed_standalone_row():
    from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from phase0_fixture_selection import fixture_resolved

    base = repo_root() / "examples/gemmini/target"
    selection = {
        "spec": fixture_resolved(base, "gemmini"),
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
        "capability_contract": load_yaml(base / "contracts/target_contract.yaml"),
    }
    _capsule, text = SPEC.build(_entry(16, 16), _binding())
    with IR_LOCK:
        sites = non_contraction_sites(mq.parse(text), selection, contract=selection["capability_contract"])
    generics = [s for s in sites if s["source_operation"] == "linalg.generic"]
    assert generics and all(s["status"] == "admitted" for s in generics), [s["reasons"] for s in generics]
    assert generics[0]["signature"]["tensor_operands"] == 2 and generics[0]["signature"]["expression"] == "affine"


def test_form_is_declared_by_the_reviewed_row():
    from merlin.targetgen.software_spec import validate_software_spec

    doc = validate_software_spec(load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml"), "gemmini")
    kept, refused = ft.declared_forms(doc, {})
    assert "tensor_sub" in kept.get("elementwise_map", []), (kept, refused)
