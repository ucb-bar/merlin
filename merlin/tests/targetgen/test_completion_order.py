"""The completion-order hazard check over executed command streams, against derived ordering facts.

The facts fixture is the command_ordering record derived from the pinned Gemmini elaboration
(``merlin.targetgen.rtl.command_ordering``); the stream fixtures are the head of the fresh4 56x56 conv
kernel's executed RoCC stream in its frozen, fence-inserted and elide-full-tiles forms. Streams are
``(funct, rs1, rs2)`` per accelerator command and ``None`` per host fence.
"""

from __future__ import annotations

import copy
import gzip
import json
import struct
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import completion_order as C

DATA = merlin_dir() / "tests" / "data" / "completion_order"
MASKS = {"scratchpad": 0x3FFF, "accumulator": 0x3FF}
ACC, ACCUMULATE, GARBAGE = 0x8000_0000, 0x4000_0000, 0xFFFF_FFFF


def test_raw_executed_stream_replays_register_values_and_fences():
    """The commit-log reader the check consumes: operand VALUES from replayed register writes."""
    from merlin.perf.streamed_executed_trace import iter_source_commands

    word = (2 << 25) | (11 << 20) | (10 << 15) | (3 << 12) | 0x7B
    log = [
        "core   0: 3 0x1000 (0x00000513) x10 0x0000000000001234",
        "core   0: 3 0x1004 (0x00000593) x11 0x0010001000000003",
        f"core   0: 0x1008 (0x{word:08x}) custom",
        f"core   0: 3 0x1008 (0x{word:08x})",
        "core   0: 3 0x100c (0x0ff0000f)",
    ]

    class View:
        def splitlines(self):
            return iter(log)

    got = list(iter_source_commands(View(), isa={"CUSTOM_OPCODE": 0x7B}, raw=True))
    assert got == [(2, 0x1234, 0x0010001000000003), None]


def _record():
    return json.loads((DATA / "gemmini_command_ordering.json").read_text())


@pytest.fixture(scope="module")
def model():
    return C.OrderingModel.from_facts(_record(), row_masks=MASKS)


def _tile(rows, cols, addr):
    return (rows << 48) | (cols << 32) | addr


def _head(name):
    data = gzip.decompress((DATA / f"fresh4_conv56_{name}_head.bin.gz").read_bytes())
    out = []
    for at in range(0, len(data), 20):
        funct, rs1, rs2 = struct.unpack_from("<iQQ", data, at)
        out.append(None if funct < 0 else (funct, rs1, rs2))
    return out


# the configuration prefix the prototype self-tests carry: CONFIG_LD (stride 64, scale 1.0 bits)
CONFIG_LD = (0, 0x100001, 0x40)
CONFIG_EX_STRIDES = (0, 1 << 16, 1 << 48)  # a_stride = 1, c_stride = 1


# ---------------------------------------------------------------------- the five prototype self-tests
def test_accumulator_overwrite_then_accumulate_race_is_flagged(model):
    stream = [
        None,
        CONFIG_LD,
        (2, 0x1 << 48, _tile(16, 16, ACC)),
        (1, 0x2 << 48, _tile(16, 16, ACC | ACCUMULATE)),
        None,
    ]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "hazard"
    assert result["hazard_pairs"] == 1
    assert result["by_kind"][0]["space"] == "accumulator"
    assert result["examples"][0]["first"]["index"] == 1 and result["examples"][0]["second"]["index"] == 2


def test_the_same_pair_with_a_fence_between_is_clear(model):
    stream = [
        None,
        CONFIG_LD,
        (2, 0x1 << 48, _tile(16, 16, ACC)),
        None,
        (1, 0x2 << 48, _tile(16, 16, ACC | ACCUMULATE)),
        None,
    ]
    assert C.check_commands(stream, model)["verdict"] == "clear"


def test_disjoint_rows_after_a_compute_are_clear(model):
    stream = [
        None,
        CONFIG_LD,
        (2, 1 << 48, _tile(16, 16, 0)),
        (6, _tile(16, 16, GARBAGE), _tile(16, 16, ACC)),
        (4, _tile(16, 16, 0), _tile(16, 16, GARBAGE)),
        (2, (1 << 48) | 0x100, _tile(16, 16, 0x100)),
        None,
    ]
    assert C.check_commands(stream, model)["verdict"] == "clear"


def test_zero_fill_then_data_into_the_same_rows_is_flagged(model):
    stream = [None, CONFIG_LD, (2, 0xFF << 48, _tile(16, 16, 0)), (2, 1 << 48, _tile(1, 16, 3)), None]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "hazard"
    first, second = result["examples"][0]["first"], result["examples"][0]["second"]
    assert first["rows"] == [0, 16] and second["rows"] == [3, 4]
    assert "zero-fill" in C.reason(result) and "#1" in C.reason(result) and "#2" in C.reason(result)


def test_zero_fill_ordered_through_a_compute_that_reads_it(model):
    """The compute waits for the fill (RAW) and the data load waits for the compute (WAR), so the fill
    completes before the data load issues -- once the compute's span covers the rows. At reset the
    RTL latches a zero A stride, the compute names only its start row and the pair stays unordered."""
    body = [
        (2, 0xFF << 48, _tile(16, 16, 0)),
        (6, _tile(16, 16, GARBAGE), _tile(16, 16, ACC)),
        (4, _tile(16, 16, 0), _tile(16, 16, GARBAGE)),
        (2, 1 << 48, _tile(1, 16, 3)),
    ]
    assert C.check_commands([None, CONFIG_LD, CONFIG_EX_STRIDES, *body, None], model)["verdict"] == "clear"
    assert C.check_commands([None, CONFIG_LD, *body, None], model)["verdict"] == "hazard"


# ---------------------------------------------------------------------- fresh4's 56x56 conv
def test_fresh4_conv56_frozen_stream_flags_zero_seed_then_gather(model):
    result = C.check_commands(_head("frozen"), model)
    assert result["verdict"] == "hazard"
    ex = result["examples"][0]
    assert (ex["first"]["index"], ex["second"]["index"]) == (71, 72)
    assert ex["first"]["rows"] == [0, 16] and ex["second"]["rows"] == [0, 1]
    assert ex["space"] == "scratchpad" and ex["first"]["queue"] == "load"
    # every pair is a fill then a gathered row, 1..16 commands apart
    assert set(result["distance_histogram"]) <= {str(d) for d in range(1, 17)}
    assert "no ordering point" in C.reason(result)


@pytest.mark.parametrize("variant", ["fence", "elide"])
def test_fresh4_fixed_variants_raise_no_hazard(model, variant):
    result = C.check_commands(_head(variant), model)
    assert result["verdict"] != "hazard"
    assert result["hazard_pairs"] == 0


@pytest.mark.parametrize("variant", ["fence", "elide"])
def test_fresh4_streams_using_the_third_load_state_are_decided(model, variant):
    """funct 14 (the bias loads) maps to MVIN3, whose span the facts derive (load state 2), so a
    fresh4 stream is judged CLEAR rather than UNKNOWN: no opaque command, nothing uncertain."""
    stream = _head(variant)
    assert sum(1 for item in stream if item is not None and item[0] == 14) == 64
    result = C.check_commands(stream, model)
    assert result["verdict"] == "clear" and result["uncertain"] == {}
    assert result["by_class"].get("MVIN3") == 64


def test_explicit_fsm_library_style_conv_with_disjoint_rows_is_clear(model):
    """Each input row is moved in once (padding from address 0, the zero source), so no two loads
    write one row before the compute that reads them; the next block's loads wait on that compute."""
    stream = [None, CONFIG_LD, CONFIG_EX_STRIDES]
    for block in range(3):
        for row in range(16):
            source = 0 if row in (0, 15) else (1 << 48) | (block * 16 + row) * 64
            stream.append((2, source, _tile(1, 16, row)))
        stream.append((1, 2 << 48, _tile(16, 16, 4096)))
        stream.append((6, _tile(16, 16, 4096), _tile(16, 16, ACC | (block * 16))))
        stream.append((4, _tile(16, 16, 0), _tile(16, 16, GARBAGE)))
    stream += [(3, 3 << 48, _tile(16, 16, ACC | block * 16)) for block in range(3)]
    stream.append(None)
    result = C.check_commands(stream, model)
    assert result["verdict"] == "clear", result


def test_hardware_loop_conv_is_opaque_not_clear_and_not_failed(model):
    """A loop unroller issues loads the stream cannot see: UNKNOWN, with the reason, never clear."""
    stream = [None, *[(f, 0, 0) for f in range(16, 22)], (15, 0, 0), None]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "unknown" and result["hazard_pairs"] == 0
    assert any("local accesses not derived" in why for why in result["uncertain"])


# ---------------------------------------------------------------------- UNKNOWN facts
def test_absent_or_underived_facts_are_unavailable_never_clear():
    with pytest.raises(C.Unavailable, match="not derived"):
        C.OrderingModel.from_facts({}, row_masks=MASKS)
    with pytest.raises(C.Unavailable, match="not derived"):
        C.OrderingModel.from_facts({"status": "unknown", "reason": "no dispatch"}, row_masks=MASKS)
    record = _record()
    record["same_queue"]["load"]["release"] = "unknown"
    with pytest.raises(C.Unavailable, match="UNKNOWN"):
        C.OrderingModel.from_facts(record, row_masks=MASKS)


def test_unknown_completion_order_is_uncertain_never_a_proven_hazard():
    record = _record()
    record["completion_order"]["load"] = {"order": "unknown", "reason": "no client facts"}
    model = C.OrderingModel.from_facts(record, row_masks=MASKS)
    stream = [None, CONFIG_LD, (2, 0xFF << 48, _tile(16, 16, 0)), (2, 1 << 48, _tile(1, 16, 3)), None]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "unknown" and result["hazard_pairs"] == 0
    assert result["queues"]["load"]["writes_may_reorder"] is None
    assert any("UNKNOWN completion order" in a for a in result["uncertain"])


def test_in_order_completion_from_the_facts_is_never_a_same_queue_hazard():
    record = _record()
    record["completion_order"]["load"] = {"order": "in_order", "basis": "test"}
    model = C.OrderingModel.from_facts(record, row_masks=MASKS)
    stream = [None, CONFIG_LD, (2, 0xFF << 48, _tile(16, 16, 0)), (2, 1 << 48, _tile(1, 16, 3)), None]
    assert C.check_commands(stream, model)["verdict"] == "clear"


def test_underived_extent_is_uncertain_not_clear():
    record = _record()
    record["extents"]["MVIN.spad_addr"] = {"status": "unknown", "reason": "test"}
    model = C.OrderingModel.from_facts(record, row_masks=MASKS)
    result = C.check_commands([None, CONFIG_LD, (2, 1 << 48, _tile(16, 16, 0)), None], model)
    assert result["verdict"] == "unknown"
    assert any("span formula not modeled" in why for why in result["uncertain"])


def test_a_configuration_outside_the_verified_formula_is_uncertain(model):
    """a_transpose swaps which count field the compute's A span follows; the facts verified the span
    only while it is clear, so a stream that sets it is not judged clear."""
    transpose = (0, (1 << 16) | (1 << 8), 1 << 48)
    stream = [
        None,
        CONFIG_LD,
        transpose,
        (6, _tile(16, 16, GARBAGE), _tile(16, 16, ACC)),
        (4, _tile(16, 16, 0), _tile(16, 16, GARBAGE)),
        None,
    ]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "unknown"


def test_block_stride_is_tracked_per_load_state(model):
    """Two 2-block loads whose blocks interleave (block stride 32) never write the same row."""
    config = (0, (32 << 16) | (0 << 3) | 1, 64)  # CONFIG_LD state 0, block stride 32
    stream = [None, config, (2, 1 << 48, _tile(16, 32, 0)), (2, 2 << 48, _tile(16, 32, 16)), None]
    assert C.check_commands(stream, model)["verdict"] == "clear"
    overlap = [None, config, (2, 1 << 48, _tile(16, 32, 0)), (2, 2 << 48, _tile(16, 16, 32)), None]
    assert C.check_commands(overlap, model)["verdict"] == "hazard"


def test_untested_relation_is_reported_when_it_occurs():
    record = copy.deepcopy(_record())
    record["cross_queue"] = [
        r
        for r in record["cross_queue"]
        if not (r["older"].startswith("MVIN") and r["younger"].startswith(("COMPUTE", "PRELOAD")))
    ]
    model = C.OrderingModel.from_facts(record, row_masks=MASKS)
    stream = [
        None,
        CONFIG_LD,
        CONFIG_EX_STRIDES,
        (2, 1 << 48, _tile(16, 16, 0)),
        (4, _tile(16, 16, 0), _tile(16, 16, GARBAGE)),
        None,
    ]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "unknown"
    assert any("not derived" in why for why in result["uncertain"])


def test_unknown_queue_does_not_suppress_independent_known_hazard():
    record = _record()
    record["completion_order"]["execute"] = {"order": "unknown"}
    model = C.OrderingModel.from_facts(record, row_masks=MASKS)
    stream = [CONFIG_LD, (2, 0, _tile(16, 16, 0)), (2, 1, _tile(1, 16, 3))]
    result = C.check_commands(stream, model)
    assert result["verdict"] == "hazard" and result["hazard_pairs"] == 1


@pytest.mark.parametrize(("writer", "verdict"), [(48, "hazard"), (64, "hazard"), (50, "clear")])
def test_wide_read_checks_every_band_but_not_the_gap(writer, verdict):
    # Explicit derived geometry: two one-row read bands at 48 and 64, not a dense hull.
    address, rows = C._Slot(0, 0, 255), C._Slot(1, 0, 255)
    write = C._Access("load.local", address, True, C._Extent(rows, None))
    read = C._Access("read.local", address, False, C._Extent(rows, None, C._Slot(1, 8, 255), 4, 16))
    model = C.OrderingModel(
        queues=("load", "execute"),
        by_funct={1: C._Command("load", 0, (write,)), 2: C._Command("read", 1, (read,))},
        config_funct=None,
        selector=None,
        by_code={},
        same_completion=(False, False),
        reorders=(True, True),
        tracked={(0, True, 1, False, 0): False},
        spaces=("scratchpad",),
        row_masks=(255,),
        select_flag=None,
        sentinel=None,
    )
    assert C.check_commands([(1, writer, 1), (2, 48, 1 | (8 << 8))], model)["verdict"] == verdict
    assert C.check_commands([(1, writer, 1), None, (2, 48, 1 | (8 << 8))], model)["verdict"] == "clear"
    tracked = replace(model, tracked={(0, True, 1, False, 0): True})
    assert C.check_commands([(1, writer, 1), (2, 48, 1 | (8 << 8))], tracked)["verdict"] == "clear"


def test_unknown_relation_against_a_fence_completed_access_is_not_required():
    record = _record()
    record["cross_queue"] = []
    model = C.OrderingModel.from_facts(record, row_masks=MASKS)
    stream = [
        CONFIG_LD,
        CONFIG_EX_STRIDES,
        (2, 0, _tile(16, 16, 0)),
        None,
        (4, _tile(16, 16, 0), _tile(16, 16, GARBAGE)),
    ]
    assert C.check_commands(stream, model)["verdict"] == "clear"
    assert C.check_commands([c for c in stream if c is not None], model)["verdict"] == "unknown"
