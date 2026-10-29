"""Declared capsule forms: the table's entry fields evaluate in the exact golden at every alignment."""

import math

import pytest

from merlin.runtime import integer_ops
from merlin.runtime.tensor import Tensor
from merlin.targetgen import corpus_form_table as ft
from merlin.targetgen.corpus_cell_forms import cell_forms
from merlin.targetgen.corpus_shapes import extents_for

PROBES = [{"boundary": "tile_edge", "edge": 16}]


def _t(shape):
    return Tensor(tuple(shape), list(range(math.prod(shape))), "i8")


@pytest.mark.parametrize("alignment", ["aligned", "partial", "sub_tile"])
@pytest.mark.parametrize("form", sorted(f for fam, f in ft.FORM_TABLE if fam == "movement"))
def test_form_fields_evaluate(form, alignment):
    ext = extents_for(alignment, PROBES)
    m, n = ft.resolve_extents(ext, PROBES)
    f = ft.entry_fields("movement", form, m, n)
    if "value_code" in f and "form" not in f:
        out = integer_ops.fill(f["shape"], {"value_code": f["value_code"], "output_dtype": "i8"})
        assert out.shape == tuple(f["shape"])
    elif "sources" in f:
        out = integer_ops.concat([_t(s) for s in f["shapes"]], {"axis": f["axis"]})
        assert out.shape[f["axis"]] == 2 * f["shapes"][0][f["axis"]]
    elif "dest" in f:
        out = integer_ops.insert_slice(_t(f["dest_shape"]), _t(f["src_shape"]), {"offsets": f["offsets"], "strides": f["strides"]})
        assert out.shape == tuple(f["dest_shape"])
    else:
        attrs = {k: v for k, v in f.items() if k not in ("shape", "out_shape")}
        out = integer_ops.move(_t(f["shape"]), attrs, f.get("out_shape"))
        assert out.shape


def test_declared_forms_become_cell_forms_with_distinct_names():
    pool = {"movement", "concat", "insert_slice"}
    forms = cell_forms("movement", "i8", None, pool, ("slice", "flip", "concat", "nope"))
    by_form = {f.form: f for f in forms}
    assert by_form["slice"].op == "movement" and by_form["flip"].op == "movement"
    assert by_form["concat"].op == "concat" and by_form["nope"].op is None
    assert len({f.qualifier() for f in forms if f.form}) == 4


def test_declared_forms_checked_against_contract_primitives():
    spec = {"operations": [{"id": "r", "families": ["movement"], "forms": ["slice", "pad_constant"]}]}
    contract = {"compute_units": [{"semantic_capabilities": [{"family": "movement", "forms": ["copy"]}]}]}
    kept, refused = ft.declared_forms(spec, contract)
    assert kept == {"movement": ["slice"]} and refused[0]["form"] == "pad_constant"


@pytest.mark.parametrize("form", sorted(f for fam, f in ft.FORM_TABLE if fam == "movement"))
def test_every_form_builds_a_capsule_and_golden(form):
    from merlin.targetgen import corpus_spec as CS
    binding = CS.CorpusBinding(
        target="gemmini", tile_dim=16, operand_dtype="int8", accum_dtype="i32", integer=True,
        tiers=["L0"], compare="exact_int",
    )  # fmt: skip
    row = ft.FORM_TABLE[("movement", form)]
    entry = {
        "name": f"T_{form}", "kind": "isa", "op": row.op, "source_role": "synthesized", "source_reference": "t",
        **ft.entry_fields("movement", form, 16, 15),
    }  # fmt: skip
    cap, mlir = CS.BUILDERS[row.op](entry, binding)
    assert cap["operation"]["op"] == row.op and "merlin_iface." + row.op in mlir
