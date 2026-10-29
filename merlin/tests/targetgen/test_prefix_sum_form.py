"""The prefix-sum form (cumsum) of the standalone integer sum: Y[i] = sum_{k<=i} X[k]."""

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
        "name": "T_cumsum", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": "prefix_sum", "M": m, "N": n, "out": "Y0", "label": "public",
        **ft.entry_fields("reduction", "cumsum", m, n),
    }  # fmt: skip


@pytest.mark.parametrize("m,n", [(16, 16), (16, 15), (5, 4)])
def test_scan_builds_and_grades_exactly(m, n):
    from merlin_experiments.phase0.component_integer_bounds import derive, source_for_capsule

    capsule, text = SPEC.build(_entry(m, n), _binding())
    assert capsule["semantic"]["semantic_family"] == "reduction"
    assert 'prov.aten = "aten.cumsum.default"' in text and "arith.select" in text
    semantics = {"model": {"engine": "integer_reference"}, "overflow": "bounded_exact"}
    capsule["numerical_semantics"] = semantics
    capsule["integer_partial_sum_bound"] = derive(source_for_capsule(capsule), semantics)
    leaves = materialize_capsule_leaves(capsule)["A0"]
    want = [[sum(int(leaves.data[k * n + c]) for k in range(r + 1)) for c in range(n)] for r in range(m)]
    assert capsule_golden.golden(capsule, None)["Y0"] == want


def _selection():
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from phase0_fixture_selection import fixture_resolved

    base = repo_root() / "examples/gemmini/target"
    return {
        "spec": fixture_resolved(base, "gemmini"),
        "host": validate_host_capabilities(load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")),
        "capability_contract": load_yaml(base / "contracts/target_contract.yaml"),
    }


def test_written_region_is_admitted_by_the_reviewed_sum_row():
    from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites

    selection = _selection()
    _capsule, text = SPEC.build(_entry(16, 16), _binding())
    with IR_LOCK:
        sites = non_contraction_sites(mq.parse(text), selection, contract=selection["capability_contract"])
    generics = [s for s in sites if s["source_operation"] == "linalg.generic"]
    assert generics and all(s["status"] == "admitted" for s in generics), [s["reasons"] for s in generics]


def test_a_select_against_a_nonzero_arm_is_not_a_zero_pad_sum():
    from merlin.targetgen.core_aten_spec_int8 import non_contraction_sites

    selection = _selection()
    _capsule, text = SPEC.build(_entry(16, 16), _binding())
    bad = text.replace("arith.constant 0 : i32", "arith.constant 1 : i32", 1)
    assert bad != text
    with IR_LOCK:
        sites = non_contraction_sites(mq.parse(bad), selection, contract=selection["capability_contract"])
    generics = [s for s in sites if s["source_operation"] == "linalg.generic"]
    assert generics and all(s["status"] != "admitted" for s in generics)


def test_form_is_declared_by_the_reviewed_row():
    from merlin.targetgen.software_spec import validate_software_spec

    doc = validate_software_spec(load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml"), "gemmini")
    kept, refused = ft.declared_forms(doc, {})
    assert kept.get("reduction") == ["cumsum"], (kept, refused)


@pytest.mark.parametrize("m,n", [(16, 16), (4, 8)])
def test_pointwise_convolution_backward_grades_exactly(m, n):
    from merlin_experiments.phase0.component_integer_bounds import derive, source_for_capsule

    entry = {
        "name": "T_cb", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": "conv_backward", "label": "public", **ft.entry_fields("contraction", "convolution_backward", m, n),
    }  # fmt: skip
    capsule, text = SPEC.build(entry, _binding())
    assert text.count('prov.aten = "aten.convolution_backward.default"') == 3
    semantics = {"model": {"engine": "integer_reference"}, "overflow": "bounded_exact"}
    capsule["numerical_semantics"] = semantics
    capsule["integer_partial_sum_bound"] = derive(source_for_capsule(capsule), semantics)
    leaves = materialize_capsule_leaves(capsule)
    x, dy, w = ([int(v) for v in leaves[k].data] for k in ("X", "dY", "W"))
    got = capsule_golden.golden(capsule, None)
    assert got["grad_weight"] == [[sum(x[i * n + k] * dy[i * n + c] for i in range(m)) for c in range(n)] for k in range(n)]
    assert got["grad_input"] == [[sum(dy[i * n + c] * w[k * n + c] for c in range(n)) for k in range(n)] for i in range(m)]
    assert got["grad_bias"] == [[sum(dy[i * n + c] for i in range(m)) for c in range(n)]]


def test_convolution_backward_form_is_declared_by_the_contraction_row():
    from merlin.targetgen.software_spec import validate_software_spec

    doc = validate_software_spec(load_yaml(repo_root() / "examples/gemmini/target/software-spec.yaml"), "gemmini")
    kept, refused = ft.declared_forms(doc, {})
    assert kept.get("contraction") == ["convolution_backward"], (kept, refused)
