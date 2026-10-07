"""The fast tiers' pieces that do not need a target, held to what makes them safe to show an agent.

* which groups an edit changed is read off the two build records, by content, and nothing else;
* a one-group program embeds exactly what its step reads, typed by the program's own ABI header;
* a screen missing its output is refused, never read as a pass;
* the simulator's calibration is REFIT from the pairs the store holds, and moves when they do;
* the emulator's direction is scored against the board only where the board moved.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from merlin.perf import whole_model_group_timing as T
from merlin.perf import whole_model_screen as S
from merlin.perf import whole_model_verdict as V


def _record(rows, elf="e" * 64):
    return {"elf_sha256": elf, "attribution": {"per_group": rows}}


def test_changed_groups_are_read_off_the_records_by_content():
    base = _record(
        [
            {"group": 1, "op": "conv2d", "on": "package", "object_sha256": "a", "gather": True},
            {"group": 2, "op": "matmul", "on": "package", "object_sha256": "b", "gather": False},
            {"group": 3, "op": "residual_add", "on": "vendor", "cause": "caller_declined"},
            {"group": 4, "op": "matmul", "on": "vendor", "cause": "operand_not_bound_by_shape"},
        ]
    )
    cand = _record(
        [
            {"group": 1, "op": "conv2d", "on": "package", "object_sha256": "a2", "gather": False},  # new kernel
            {"group": 2, "op": "matmul", "on": "package", "object_sha256": "b", "gather": False},  # same bytes
            {"group": 3, "op": "residual_add", "on": "package", "object_sha256": "c"},  # now answered
            {"group": 4, "op": "matmul", "on": "vendor", "cause": "operand_not_bound_by_shape"},  # same call
        ],
        elf="f" * 64,
    )
    diff = T.changed_groups(cand, base)
    assert diff["changed"] == [1, 3] and diff["unchanged"] == [2, 4]
    assert set(diff["why"][1]) == {"object_sha256", "gather"}
    assert diff["why"][3]["on"] == {"candidate": "package", "baseline": "vendor"}
    assert diff["kinds"] == {1: "conv2d", 3: "residual_add"}
    with pytest.raises(T.GroupTimingError):
        T.changed_groups({"elf_sha256": "x"}, base)


HEADER = """
#include <stdint.h>
#define DIM 16
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
typedef int8_t elem_t;   /* the datapath element, after a macro with parentheses */
typedef uint64_t acc_t;
typedef acc_t wide_acc_t;
typedef char plain_t;
typedef struct { int x; } not_scalar_t;
"""


def test_buffer_types_are_read_from_the_programs_own_header(tmp_path):
    header = tmp_path / "params.h"
    header.write_text(HEADER)
    found = T.ctype_dtypes(header)
    assert found["elem_t"] == "<i1" and found["acc_t"] == "<u8" and found["wide_acc_t"] == "<u8"
    assert "plain_t" not in found and "not_scalar_t" not in found  # unknown signedness, not a scalar


MODEL = {
    "steps": [
        {"group": 1, "kind": "conv2d", "in": "IMAGE", "out": "B_g1", "weight": "W_g1", "bias": "BIAS_g1"},
        {"group": 2, "kind": "matmul", "in": "B_g1", "out": "B_g2", "weight": "W_g2", "bias": None, "scale": 0.5},
        {"group": 3, "kind": "add", "lhs": "B_g1", "rhs": "B_g2", "out": "B_g3"},
    ],
    "buffers": [
        {"name": "B_g1", "elements": 4, "ctype": "elem_t"},
        {"name": "B_g2", "elements": 2, "ctype": "acc_t"},
        {"name": "B_g3", "elements": 2, "ctype": "elem_t"},
    ],
    "arrays": {
        "W_g1": np.zeros(3, np.int8),
        "BIAS_g1": np.zeros(2, np.int32),
        "W_g2": np.ones(8, np.int8),
        "IMAGE_DATA": np.zeros(5, np.int8),
        "GOLDEN": np.zeros(10, np.float32),
    },
    "classes": 10,
}


def test_a_one_group_model_embeds_what_its_step_reads_and_closes_on_its_own_output():
    dtypes = {"elem_t": "<i1", "acc_t": "<u8"}
    one = T._one_group_model(MODEL, 2, {"B_g1": np.array([[1, -2], [3, 4]])}, dtypes)
    assert [s["group"] for s in one["steps"]] == [2]
    assert [b["name"] for b in one["buffers"]] == ["B_g2"]  # its output stays a writable buffer
    assert one["arrays"]["B_g1"].dtype == np.int8 and one["arrays"]["B_g1"].tolist() == [1, -2, 3, 4]
    assert "W_g1" not in one["arrays"] and "BIAS_g1" not in one["arrays"]  # no other group's weights
    assert {"W_g2", "GOLDEN"} <= set(one["arrays"])
    assert one["steps"][0]["dequantize"] == 1.0 and one["classes"] == 1 and one["embedded_inputs"] == ["B_g1"]
    assert MODEL["steps"][1].get("dequantize") is None  # the model itself is untouched
    # Two inputs, each in the width its header type has.
    add = T._one_group_model(MODEL, 3, {"B_g1": np.arange(4), "B_g2": np.array([5, 6])}, dtypes)
    assert add["embedded_inputs"] == ["B_g1", "B_g2"] and add["arrays"]["B_g2"].dtype == np.dtype("<u8")
    with pytest.raises(T.GroupTimingError, match="does not produce"):
        T._one_group_model(MODEL, 2, {}, dtypes)
    with pytest.raises(T.GroupTimingError, match="element"):
        T._one_group_model(MODEL, 2, {"B_g1": np.zeros(3)}, dtypes)
    with pytest.raises(T.GroupTimingError, match="no fixed-width type"):
        T._one_group_model(MODEL, 2, {"B_g1": np.zeros(4)}, {})


def test_direction_is_scored_only_where_the_board_moved_and_poor_kinds_are_flagged():
    rows = [
        {"group": 1, "kind": "conv2d", "ratio": 0.5},
        {"group": 2, "kind": "conv2d", "ratio": 0.8},
        {"group": 3, "kind": "residual_add", "ratio": 0.7},
        {"group": 4, "kind": "residual_add", "ratio": 0.9},
        {"group": 5, "kind": "matmul", "ratio": 1.2},
    ]
    board = {1: (40, 100), 2: (70, 100), 3: (130, 100), 4: (120, 100), 5: (101, 100)}  # g5 moved 1%: no direction
    found = T.direction_agreement(rows, board)
    assert found["groups_counted"] == 4 and found["agreement"] == 0.5
    assert found["by_kind"]["conv2d"] == {"n": 2, "agree": 2, "agreement": 1.0, "direction_unreliable": False}
    assert found["by_kind"]["residual_add"]["direction_unreliable"] is True
    assert "matmul" not in found["by_kind"]


# ------------------------------------------------------------------------------ structure screen
def _console(*, drop=None, wrong=None, cycles=None):
    t = V.PROTOCOL_TEMPLATES
    cycles = cycles or {"1": 100, "2": 50, "3": 70}
    lines = [t["invocations"], t["full_model"].format(cycles=sum(cycles.values()))]
    kinds = {"1": "conv2d", "2": "sum", "3": "matmul"}
    for g in ("1", "2", "3"):
        if g != drop:
            lines.append(t["group"].format(group=g, kind=kinds[g], cycles=cycles[g], sum=0, checksum=0))
    lines.append(t["bounded"].format(group=2, max_abs=1, over=0, bound=1))
    for g in ("1", "3"):
        bad = 5 if g == wrong else 0
        lines.append(t["local"].format(group=g, mismatches=bad, elements=64, first=3 if bad else -1))
    lines.append(t["window_end"].format(label="group_model"))
    return "\n".join(lines) + "\n"


GROUPS = {"1": "exact", "2": "bounded_int", "3": "exact"}


def test_a_screen_missing_a_group_line_is_refused():
    screen = S.screen_console(_console(drop="3"), GROUPS)
    assert screen["status"] == "refused" and "2 of the 3" in screen["refusal"] and "groups" not in screen


def test_a_screen_reports_local_correctness_and_never_feeds_the_objective():
    screen = S.screen_console(_console(wrong="3"), GROUPS)
    assert screen["status"] == "screened" and screen["feeds_objective"] is False
    assert screen["all_groups_correct"] is False and screen["groups_not_correct"] == [3]
    rows = {r["group"]: r for r in screen["groups"]}
    assert rows[3]["local"] == "wrong" and rows[3]["failure"] == {"mismatches": 5, "of": 64}
    assert rows[1]["local"] == "correct" and rows[1]["spike_cycles"] == 100
    assert rows[1]["spike_blind"] is None  # no calibration: blindness is unknown, not false


def _pairs(kind_ratios: dict, *, source="s"):
    pairs = []
    for index, (kind, ratios) in enumerate(kind_ratios.items()):
        for n, ratio in enumerate(ratios):
            pairs.append(
                {
                    "source": f"{source}{n}",
                    "group": str(index + 1),
                    "kind": kind,
                    "spike": 100,
                    "board": int(100 * ratio),
                }
            )
    return pairs


def test_the_calibration_is_refit_from_the_pairs_and_flags_blind_kinds():
    fit = S.fit_calibration(_pairs({"conv2d": [4, 5], "sum": [80, 95], "matmul": [5, 6]}))
    assert fit["pairs"] == 6 and fit["kinds"]["sum"]["spike_blind"] is True
    assert fit["kinds"]["conv2d"] == {
        "n": 2,
        "median_ratio": 4.5,
        "p10": 4.1,
        "p90": 4.9,
        "max_ratio": 5.0,
        "spike_blind": False,
    }
    screen = S.screen_console(_console(), GROUPS, calibration=fit)
    rows = {r["group"]: r for r in screen["groups"]}
    assert rows[2]["spike_blind"] is True and "class_ratio" not in rows[2]
    assert rows[1]["spike_blind"] is False and rows[1]["class_ratio"]["median_ratio"] == 4.5
    assert not any("estimate" in key for row in rows.values() for key in row)  # no unmeasured board number
    # A matmul that turns out memory-bound in new pairs moves the fit: nothing about it is a constant.
    refit = S.fit_calibration(_pairs({"conv2d": [4, 5], "sum": [80, 95], "matmul": [5, 6, 90, 110, 120]}))
    assert refit["kinds"]["matmul"]["spike_blind"] is True
    assert {r["group"]: r for r in S.screen_console(_console(), GROUPS, calibration=refit)["groups"]}[3][
        "spike_blind"
    ] is True


def test_pairs_are_collected_from_the_store_once_per_elf(tmp_path):
    base = tmp_path / "store"
    t = V.PROTOCOL_TEMPLATES
    board = _console(cycles={"1": 400, "2": 5000, "3": 350})
    elf = "a" * 64
    for root, key in (("r1", "job1"), ("r2", "job2")):  # the same ELF measured under two stores
        job = base / root / key
        job.mkdir(parents=True)
        (job / "uart.log").write_text(board)
        (job / "result.json").write_text(
            json.dumps(
                {
                    "build": {"elf_sha256": elf},
                    "device": {"rung": "fpga_firesim"},
                    "run": {"completed": True, "uart_log": str(job / "uart.log")},
                }
            )
        )
    unscreened = base / "r3" / "job3"
    unscreened.mkdir(parents=True)
    (unscreened / "result.json").write_text(json.dumps({"build": {"elf_sha256": "b" * 64}}))
    assert S.collect_pairs([base]) == []  # a board result with no screen of its ELF teaches nothing
    screens = S.screen_dir(base, elf)
    screens.mkdir(parents=True)
    (screens / "console.txt").write_text(_console())
    (screens / S.SCREEN_FILE).write_text(json.dumps({"elf_sha256": elf}))
    pairs = S.collect_pairs([base])
    assert len(pairs) == 3 and {p["group"]: p["board"] / p["spike"] for p in pairs} == {"1": 4.0, "2": 100.0, "3": 5.0}
    assert t["group"]  # the protocol the consoles were written in is the program's own


def test_a_board_variant_is_named_by_the_drivers_own_receipt_never_by_a_file_name_assumed_here():
    receipt = {
        "elf": "/p/prog.elf",
        "program_source": "/p/prog.c",
        "program_object": "/p/prog.o",
        "support_objects": ["/p/crt.o"],
        "linked_objects": [{"path": "/k/g3.o", "sha256": "a" * 64}],
        "program_sha256": "b" * 64,
        "compiler": "cc",
        "flags": ["-O2"],
    }
    variant = T._board_variant(receipt)
    assert variant["program_object"] == "/p/prog.o" and variant["supports"] == ["/p/crt.o"]
    assert variant["objects"] == ["/k/g3.o"] and variant["program"]["compiler"] == "cc"
    with pytest.raises(T.GroupTimingError, match="names no program_source"):
        T._board_variant({k: v for k, v in receipt.items() if k != "program_object"})


def test_a_kept_interface_says_when_nothing_was_asked(tmp_path):
    lower = tmp_path / "lower"
    lower.mkdir()
    (lower / "g4.iface.mlir").write_text("module {}")
    kept = T._kept_interface(lower, 4, tmp_path / "g4")
    assert kept["interface"].endswith("interface.mlir") and len(kept["interface_sha256"]) == 64
    assert T._kept_interface(lower, 5, tmp_path / "g5")["interface"] is None


def test_the_debug_companion_is_the_recorded_recipe_plus_debug_information(tmp_path):
    """Every flag the program was built with is kept, in order; the debug option is appended to them."""
    from pathlib import Path

    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe

    recipe = HarnessBuildRecipe(
        compiler=Path("/toolchain/bin/cc"),
        include_roots=(),
        support_sources=(),
        link_script=Path("/link.ld"),
        load_address=0,
        cflags=("-O2", "-march=one"),
        ldflags=("-lm",),
    )
    seen = []

    def link(directory, chosen):
        seen.append((directory, chosen))
        return {"elf": str(directory / "p.elf"), "elf_sha256": "a" * 64, "program_object": str(directory / "p.o")}

    record = T._debug_companion(link, recipe, tmp_path / "program.debug")
    ((directory, chosen),) = seen
    assert directory == tmp_path / "program.debug"
    assert chosen.cflags == ("-O2", "-march=one", T.DEBUG_INFO_OPTION) and chosen.ldflags == recipe.ldflags
    assert chosen.compiler == recipe.compiler and chosen.link_script == recipe.link_script
    assert record["compiler"] == "/toolchain/bin/cc" and record["elf"].endswith("p.elf")

    def broken(directory, chosen):
        raise SystemExit("link failed")

    refused = T._debug_companion(broken, recipe, tmp_path / "again")
    assert "link failed" in refused["refusal"] and "elf" not in refused
