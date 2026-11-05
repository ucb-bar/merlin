"""Actual emitted load mappings, including bounded counted groups and refusals."""
from dataclasses import replace

import pytest

from merlin.compile.scheduling.tile_load_coalescing import (
    TileLoadLaw,
    TileLoadMapping,
    coalesce_tile_load_loops,
)
from merlin.sched.ir import Call, Const, Kernel, Loop, TensorArg, Var, concretize, instances


def example(*, tiles=9, tile=8, rows=3, stride=11, max_columns=32):
    law = TileLoadLaw(tile, tile, max_columns, 2, 2048, stride, "synthetic-two-byte-tiled-copy")
    def project(call):
        if call.instr != "read":
            return None
        return TileLoadMapping("external", "private", call.arg("byte"), call.arg("row"),
                               call.arg("rows"), call.arg("cols"), law)
    def encode(call, columns):
        return replace(call, args=tuple((k, Const(columns) if k == "cols" else v) for k, v in call.args))
    load = Call("read", (("byte", Var("outer") * 4096 + Var("column") * tile * 2),
                         ("row", Var("outer") * 1024 + Var("column") * stride),
                         ("rows", Const(rows)), ("cols", Const(tile))))
    kernel = Kernel("copy", (TensorArg("input", (1,), "i8", "read"),),
                    (Loop("outer", 2, (Loop("column", tiles, (load,)),)),))
    return kernel, law, project, encode


def bytes_moved(kernel, law):
    writes = []
    for call, environment in instances(kernel):
        a = concretize(call, environment)
        for row in range(a["rows"]):
            for col in range(a["cols"]):
                for byte in range(law.element_bytes):
                    writes.append((a["byte"] + row * law.source_pitch_bytes + col * law.element_bytes + byte,
                                   a["row"] + col // law.tile_columns * law.local_block_stride + row,
                                   col % law.tile_columns, byte))
    return sorted(writes)


@pytest.mark.parametrize("tiles", [2, 3, 4, 5, 9, 17])
@pytest.mark.parametrize("tile,rows,stride", [(8, 3, 11), (16, 16, 16)])
def test_exact_actual_source_byte_to_local_cell_mapping(tiles, tile, rows, stride):
    original, law, project, encode = example(tiles=tiles, tile=tile, rows=rows, stride=stride,
                                           max_columns=tile * 4)
    actual, report = coalesce_tile_load_loops(original, project=project, encode=encode, max_tiles=3)
    assert bytes_moved(actual, law) == bytes_moved(original, law)
    assert report["changes"] and report["actual_sha256"] == actual.digest()
    assert len(tuple(instances(actual))) < len(tuple(instances(original)))
    assert report["changes"][0]["source_pitch_bytes"] != report["changes"][0]["local_block_stride"]


@pytest.mark.parametrize("bad_field", ["byte", "row", "rows", "cols"])
def test_actual_encoder_mutations_are_rejected(bad_field):
    original, law, project, good = example()
    def changed(call, columns):
        encoded = good(call, columns)
        return replace(encoded, args=tuple((k, v + 1 if k == bad_field else v) for k, v in encoded.args))
    with pytest.raises(ValueError, match="actual target operands"):
        coalesce_tile_load_loops(original, project=project, encode=changed, max_tiles=4)


def test_dependencies_other_effects_partial_tiles_and_unknown_mapping_keep_original():
    original, law, project, encode = example()
    outer = original.body[0]
    loop = outer.body[0]
    load = loop.body[0]
    variants = [replace(loop, body=(replace(load, unit="dma"),)),
                replace(loop, body=(load, Call("wait", ()))),
                replace(loop, body=(replace(load, args=tuple(
                    (k, Const(law.tile_columns-1) if k == "cols" else v) for k, v in load.args)),))]
    for variant in variants:
        kernel = replace(original, body=(replace(outer, body=(variant,)),))
        actual, report = coalesce_tile_load_loops(kernel, project=project, encode=encode, max_tiles=4)
        assert actual == kernel and not report["changes"]
    actual, report = coalesce_tile_load_loops(original, project=lambda call: None, encode=encode, max_tiles=4)
    assert actual == original and not report["changes"]


def test_million_tile_loop_remains_counted_and_bad_contracts_refuse():
    original, law, project, encode = example(tiles=1_000_000)
    actual, report = coalesce_tile_load_loops(original, project=project, encode=encode, max_tiles=4)
    assert report["static_nodes"] < 8
    assert actual.body[0].body[0].extent == 250_000
    for update in ({"synchronous": False}, {"exact_copy": False}, {"max_columns": 7},
                   {"source_pitch_bytes": True}, {"local_block_stride": 0}):
        with pytest.raises(ValueError, match="bounded exact synchronous"):
            replace(law, **update)


def test_final_actual_wide_access_expands_every_local_tile_band_for_hazards():
    from merlin.compile.scheduling.tile_load_coalescing import expand_tile_load_mapping
    original, law, project, encode = example(tiles=5)
    actual, report = coalesce_tile_load_loops(original, project=project, encode=encode, max_tiles=4)
    wide = actual.body[0].body[0].body[0]
    bands = expand_tile_load_mapping(project(wide))
    assert len(bands) == 4 and all(b.columns == Const(law.tile_columns) for b in bands)
    assert report["changes"][0]["actual_access"]["columns"] == 32
    with pytest.raises(ValueError, match="column bound"):
        expand_tile_load_mapping(replace(project(wide), columns=Const(law.max_columns+1)))


@pytest.mark.parametrize("tiles", [5, 6, 17, 32])
def test_optin_shorter_tail_preserves_every_byte_and_actual_band(tiles):
    from merlin.compile.scheduling.tile_load_coalescing import expand_tile_load_mapping

    original, law, project, encode = example(tiles=tiles, tile=16, rows=16, stride=16, max_columns=127)
    control, _ = coalesce_tile_load_loops(original, project=project, encode=encode, max_tiles=7)
    actual, report = coalesce_tile_load_loops(original, project=project, encode=encode,
                                             max_tiles=7, coalesce_tail=True)
    assert bytes_moved(actual, law) == bytes_moved(original, law)
    assert len(tuple(instances(actual))) <= len(tuple(instances(control)))
    for call, _ in instances(actual):
        assert len(expand_tile_load_mapping(project(call))) == call.arg("cols").value // 16
    change = report["changes"][0]
    assert change["actual_commands"] == (tiles+6)//7
    if tiles % 7 > 1 and tiles > 7:
        assert change["tail_commands"] == 1 and change["coalesced_tail_tiles"] == tiles % 7


def test_tail_encoder_is_checked_separately_from_valid_full_width():
    original, _, project, encode = example(tiles=8)

    def corrupt_only_tail(call, columns):
        encoded = encode(call, columns)
        return encoded if columns == 24 else replace(encoded, args=tuple(
            (key, value+1 if key == "row" else value) for key, value in encoded.args))

    with pytest.raises(ValueError, match="tail actual target operands"):
        coalesce_tile_load_loops(original, project=project, encode=corrupt_only_tail,
                                 max_tiles=3, coalesce_tail=True)
