"""A written standalone elementwise-map region is admitted from its own program, not left unresolved.

The inventory row of an elementwise ``linalg.generic`` carries its operand count, scalar-expression class and
broadcast class (``access_observations.primitive_signature``) and the pipeline's per-tensor calibration policy,
the same observations the site screen reads; the reviewed standalone elementwise rows constrain exactly these.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0.program_admission import account_program
from phase0_fixture_selection import fixture_resolved

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen import corpus_form_table as ft
from merlin.targetgen import corpus_spec as SPEC

pytestmark = pytest.mark.target("gemmini")

_FORMS = [
    ("elementwise_map", "tensor_sub", "tensor_sub", "sub"),
    ("elementwise_map", "scalar_add", "scalar_affine", "affine"),
    ("elementwise_map", "scalar_sub", "scalar_affine", "affine"),
    ("elementwise_map", "scalar_mul", "scalar_affine", "affine"),
    ("elementwise_map", "scalar_neg", "scalar_affine", "affine"),
    ("elementwise_map", "cast_widen", "scalar_affine", "cast"),
    ("elementwise_map", "relu", "scalar_affine", "relu"),
    ("reduction", "cumsum", "prefix_sum", "aten.cumsum.default"),
]


def _evidence():
    base = repo_root() / "examples/gemmini/target"
    host = load_yaml(repo_root() / "examples/gemmini/platform/host-capabilities-core-aten.yaml")
    selection = {
        "capability_spec": host,
        "package_sha256": host["compiler"]["package_sha256"],
        "capability_spec_sha256": "b" * 64,
        "dtype_strategy": host["compiler"]["dtype_strategy"],
    }
    return SimpleNamespace(
        software_spec=fixture_resolved(base, "gemmini"),
        contract=load_yaml(base / "contracts/target_contract.yaml"),
        host_capabilities={"profile": selection},
        whole_program_admission=None,
    )


@pytest.mark.parametrize("family,form,op,operation", _FORMS)
def test_the_written_form_places_its_compute_operation_on_the_accelerator(family, form, op, operation):
    binding = SPEC.CorpusBinding(
        target="t", tile_dim=16, operand_dtype="int8", accum_dtype="int32", integer=True,
        tiers=["L2", "L3"], compare="exact",
    )  # fmt: skip
    entry = {
        "name": f"T_{form}", "kind": "isa", "source_role": "derived_sweep", "source_reference": "test",
        "op": op, "M": 16, "N": 16, "lhs": "A0", "out": "Y0", "label": "public",
        **ft.entry_fields(family, form, 16, 16),
    }  # fmt: skip
    _capsule, text = SPEC.build(entry, binding)
    path = Path(tempfile.mkdtemp()) / "capsule.interface.mlir"
    path.write_text(text)
    rows = [r for r in account_program(path, name=entry["name"], target="gemmini", evidence=_evidence()) if r["operation"] == operation]
    assert rows, operation
    assert all((r["accelerator_admission"] or {}).get("status") == "admitted" for r in rows), [
        (r["operation"], [d["reason"] for d in r.get("software_admissions", []) if d["status"] != "admitted"][:3]) for r in rows
    ]
