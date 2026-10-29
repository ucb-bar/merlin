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


@pytest.mark.parametrize(("src", "dst"), [([4, 6], [24]), ([4, 6], [4, 1, 6]), ([4, 1, 6], [4, 6]), ([2, 3, 4], [6, 4])])
def test_reshape_keeps_row_major_order(src, dst):
    out = integer_ops.move(_t(src), {"form": "reshape"}, dst)
    assert list(out.shape) == dst and out.data == list(range(math.prod(src)))


def test_reshape_refuses_a_different_element_count_or_no_shape():
    with pytest.raises(integer_ops.IntegerOpError):
        integer_ops.move(_t([4, 6]), {"form": "reshape"}, [5, 5])
    with pytest.raises(integer_ops.IntegerOpError):
        integer_ops.move(_t([4, 6]), {"form": "reshape"}, None)


def test_upsample_is_nearest_per_axis():
    np = pytest.importorskip("numpy")
    out = integer_ops.move(_t([3, 2, 5]), {"form": "upsample", "scales": [2, 2, 1]}, None)
    ref = np.repeat(np.repeat(np.arange(30).reshape(3, 2, 5), 2, axis=0), 2, axis=1)
    assert out.shape == (6, 4, 5) and out.data == ref.reshape(-1).tolist()
    with pytest.raises(integer_ops.IntegerOpError):
        integer_ops.move(_t([3, 2]), {"form": "upsample", "scales": [0, 1]}, None)


def test_as_strided_reads_row_major_storage_and_diagonal_is_one_of_them():
    np = pytest.importorskip("numpy")
    base = np.arange(36).reshape(6, 6)
    diag = integer_ops.move(_t([6, 6]), {"form": "as_strided", "sizes": [6], "strides": [7], "storage_offset": 0}, None)
    assert diag.data == np.diagonal(base).tolist()
    off = integer_ops.move(_t([6, 6]), {"form": "as_strided", "sizes": [5], "strides": [7], "storage_offset": 1}, None)
    assert off.data == np.diagonal(base, 1).tolist()
    sliding = integer_ops.move(_t([6, 6]), {"form": "as_strided", "sizes": [3, 6], "strides": [3, 1], "storage_offset": 0}, None)
    assert sliding.data == [i + 3 * r for r in range(3) for i in range(6)]
    for bad in (
        {"sizes": [6], "strides": [7], "storage_offset": 8},  # reads past the end
        {"sizes": [6], "strides": [0], "storage_offset": 0},  # stride must be positive
    ):
        with pytest.raises(integer_ops.IntegerOpError):
            integer_ops.move(_t([6, 6]), {"form": "as_strided", **bad}, None)


@pytest.mark.parametrize(("code", "broadcasting", "work"), [(0, "none", "zero_fill"), (5, "outer", "broadcast")])
def test_fill_command_observes_its_class_and_work(code, broadcasting, work):
    from merlin.targetgen.interface_observations import command_rows

    cb = {
        "abi_version": "0.1", "target": "fixture",
        "tensors": {"Y": {"shape": [4, 4], "dtype": "i8", "role": "output"}},
        "commands": [{"opcode": "FILL", "operands": {"dst": "Y"}, "attributes": {"value_code": code, "output_dtype": "i8"}}],
    }  # fmt: skip
    [row] = [r for r in command_rows(cb) if r["operation"] == "fill"]
    assert (row["broadcasting"], row["movement_form"], row["operand_dtypes"]) == (broadcasting, work, ["i8"])
