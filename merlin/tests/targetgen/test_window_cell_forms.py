"""The window-sum declared forms: one capsule per form the SUM reduction row declares, each a valid geometry."""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.runtime import integer_ops
from merlin.runtime.tensor import Tensor
from merlin.targetgen.corpus_cell_forms import cell_forms
from merlin.targetgen.corpus_form_table import FORM_TABLE, declared_forms, entry_fields
from merlin.targetgen.corpus_synth import available_ops
from merlin.targetgen.corpus_window_forms import WINDOW_FORMS
from merlin.targetgen.eligibility import capability_map_from_contract

_CONTRACT = {
    "name": "fixture",
    "compute_units": [
        {
            "name": "array",
            "kind": "systolic",
            "dtypes": ["int8"],
            "ops": ["matmul"],
            "semantic_capabilities": [
                {"family": "contraction", "dtypes": ["int8"]},
                {"family": "reduction", "dtypes": ["int8"], "composed_with": ["contraction", "movement"]},
                {"family": "reduction", "dtypes": ["int8"], "kinds": ["sum"]},
            ],
        }
    ],
}


def _spec_rows():
    doc = yaml.safe_load((repo_root() / "examples/gemmini/target/software-spec.yaml").read_text())
    return doc["operations"]["reduction_standalone"]


def test_the_reviewed_sum_row_declares_exactly_the_table_forms():
    row = _spec_rows()
    assert row["status"] == "reviewed"
    assert "approved in operator chat 2026-10-29" in row["evidence"]["reviewed_by"]
    assert sorted(row["forms"]) == sorted(WINDOW_FORMS) and len(row["forms"]) == len(WINDOW_FORMS)
    kept, refused = declared_forms(
        {"operations": [{**row, "id": "reduction_standalone", "families": ["reduction"]}]}, _CONTRACT
    )
    assert sorted(kept.get("reduction", [])) == sorted(WINDOW_FORMS) and list(kept) == ["reduction"] and not refused


def test_each_declared_form_becomes_one_window_sum_cellform():
    cap = capability_map_from_contract(_CONTRACT)["reduction"]
    forms = cell_forms("reduction", "i8", cap, available_ops(), tuple(WINDOW_FORMS))
    assert [f.form for f in forms if f.form] == list(WINDOW_FORMS)
    assert {f.op for f in forms if f.form} == {"window_sum"}
    assert all(FORM_TABLE[("reduction", f.form)].once for f in forms if f.form)


@pytest.mark.parametrize("form", sorted(WINDOW_FORMS))
def test_every_form_is_a_geometry_the_exact_definition_computes(form):
    entry = entry_fields("reduction", form, 16, 16)
    attrs = {k: v for k, v in entry.items() if k not in {"shape", "out_shape"}}
    wide = attrs.pop("output_dtype", None) == "i32"
    attrs.update(readout_scale=1.0, output_dtype="i32" if wide else "i8", layout="nchw")
    values = np.random.default_rng(0).integers(-4, 4, entry["shape"])
    src = Tensor(tuple(entry["shape"]), [int(v) for v in values.reshape(-1)], "i8")
    out = integer_ops.window_sum(src, attrs, entry["out_shape"])
    assert list(out.shape) == entry["out_shape"]


@pytest.mark.parametrize("form", sorted(WINDOW_FORMS))
def test_every_form_builds_a_window_sum_capsule(form):
    from merlin.targetgen import corpus_spec as CS

    binding = CS.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="i32", integer=True,
        tiers=["L0"], compare="exact_int",
    )  # fmt: skip
    row = FORM_TABLE[("reduction", form)]
    entry = {
        "name": f"T_{form}", "kind": "isa", "op": row.op, "source_role": "synthesized", "source_reference": "t",
        **entry_fields("reduction", form, 16, 16),
    }  # fmt: skip
    cap, mlir = CS.BUILDERS[row.op](entry, binding)
    assert cap["operation"]["op"] == "window_sum" and "merlin_iface.window_sum" in mlir
