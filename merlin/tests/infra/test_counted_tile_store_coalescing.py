"""Actual local/output coordinates and original state survive wider readouts."""
from dataclasses import replace

import pytest

from merlin.compile.scheduling.tile_load_coalescing import (
    TileStoreAddressLaw,
    TileStoreMapping,
    expand_tile_store_mapping,
    propose_tile_store_coalescing,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, Var, concretize, instances


def example(tiles=5):
    law = TileStoreAddressLaw(8, 8, 32, 1, 257, 11, "source-rne-scale-1-over-128", "selected-store")
    store = Call("store", (("byte", Var("row") * 8192 + Var("column") * 8),
                            ("local", Var("row") * 1024 + Var("column") * 11),
                            ("rows", Const(3)), ("cols", Const(8)), ("state", Const(0))))
    kernel = Kernel("readout", (), (Call("producer_complete", ()), Loop("row", 2,
                    (Loop("column", tiles, (store,)),)), Call("final_complete", ())))

    def project(call):
        if call.instr != "store":
            return None
        original_state = law if call.arg("state") == Const(0) else replace(law, readout_identity="changed")
        return TileStoreMapping("local", "output", call.arg("byte"), call.arg("local"),
                                call.arg("rows"), call.arg("cols"), original_state)

    def encode(call, columns):
        return replace(call, args=tuple((k, Const(columns) if k == "cols" else v) for k, v in call.args))

    return kernel, law, project, encode


def actual_cells(kernel, project):
    cells = []
    for call, env in instances(kernel):
        if project(call) is None:
            continue
        x = concretize(call, env)
        law = project(call).law
        for row in range(x["rows"]):
            for column in range(x["cols"]):
                cells.append((x["local"] + column // law.tile_columns * law.local_block_stride + row,
                              column % law.tile_columns,
                              x["byte"] + row * law.output_pitch_bytes + column * law.element_bytes,
                              law.readout_identity))
    return sorted(cells)


@pytest.mark.parametrize("tiles", [2, 5, 6, 17])
def test_exact_wide_output_coordinates_all_bands_and_original_completion(tiles):
    source, law, project, encode = example(tiles)
    actual, receipt = propose_tile_store_coalescing(
        source, project=project, encode=encode, max_tiles=4, coalesce_tail=True,
    )
    assert actual_cells(actual, project) == actual_cells(source, project)
    assert actual.body[0] == source.body[0] and actual.body[-1] == source.body[-1]
    assert receipt["schema"] == "merlin.counted_tile_store_coalescing.v1"
    assert receipt["changes"] and receipt["changes"][0]["output_pitch_bytes"] == 257
    for call, _ in instances(actual):
        mapping = project(call)
        if mapping is None:
            continue
        bands = expand_tile_store_mapping(mapping)
        assert len(bands) == call.arg("cols").value // law.tile_columns
        assert all(b.law == law and b.columns == Const(8) for b in bands)


@pytest.mark.parametrize("field", ["byte", "local", "rows", "cols", "state"])
def test_actual_encoded_address_or_original_numerical_state_change_refuses(field):
    source, _, project, good = example()

    def corrupt(call, columns):
        encoded = good(call, columns)
        return replace(encoded, args=tuple((key, value + 1 if key == field else value)
                                           for key, value in encoded.args))

    with pytest.raises(ValueError, match="actual target operands"):
        propose_tile_store_coalescing(source, project=project, encode=corrupt, max_tiles=4)


def test_no_merge_across_dependency_completion_configuration_or_partial_tile():
    source, _, project, encode = example()
    row = source.body[1]
    column = row.body[0]
    store = column.body[0]
    variants = [replace(column, body=(store, Call("configure", ()))),
                replace(column, body=(store, Call("complete", ()))),
                replace(column, body=(replace(store, unit="store_unit"),)),
                replace(column, body=(replace(store, args=tuple(
                    (k, Const(7) if k == "cols" else v) for k, v in store.args)),))]
    for changed in variants:
        kernel = replace(source, body=(source.body[0], replace(row, body=(changed,)), source.body[-1]))
        actual, receipt = propose_tile_store_coalescing(kernel, project=project, encode=encode, max_tiles=4)
        assert actual == kernel and receipt["changes"] == []


def test_invalid_state_and_actual_wide_bounds_refuse_without_mutating_source():
    source, law, project, _ = example()
    with pytest.raises(ValueError, match="original readout state"):
        replace(law, readout_identity="")
    mapping = project(source.body[1].body[0].body[0])
    with pytest.raises(ValueError, match="column bound"):
        expand_tile_store_mapping(replace(mapping, columns=Const(33)))
    assert source.digest() == example()[0].digest()


def test_wider_store_never_introduces_overlapping_rows():
    source, law, project, encode = example()

    def close_rows(call):
        original = project(call)
        return None if original is None else replace(original, law=replace(law, output_pitch_bytes=16))

    actual, receipt = propose_tile_store_coalescing(source, project=close_rows, encode=encode, max_tiles=4)
    assert receipt["changes"][0]["tiles_per_command"] == 2
    assert all(call.arg("cols").value <= 16 for call, _ in instances(actual) if call.instr == "store")
    with pytest.raises(ValueError, match="disjoint original output"):
        replace(law, output_pitch_bytes=7)
