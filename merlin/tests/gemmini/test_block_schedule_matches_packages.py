"""The library block-schedule pass reproduces the hand-written package variants it was lifted from.

Each variant under ``out/artifacts/targets/gemmini/`` is one measured scheduling policy (the capsule
head-to-head's Verilator and FPGA tables). This test runs each package's OWN lowering in a subprocess
(the packages vendor their dependencies, so they import cleanly with only their ``mlir_oot`` on the
path), decodes its instruction stream back to block moves, and requires the pass with the matching
knob values to produce exactly that stream. The packages are read, never imported into this session
and never modified.

Two variants do NOT match on the deepest shape, and that is the point of the pass: there the two
operand regions overlap, and hoisting every load (or grouping them by operand) overwrites an input
block before the preload that reads it as weights. The pass refuses; the hand-written variant emits
it, and an in-order executor would compute on the wrong bytes without failing.
"""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from merlin.common.paths import artifacts_dir, runs_dir
from merlin.compile.scheduling import (BANK_ALIGNED, CONTIGUOUS, Compute, Contraction, Geometry,
                                       Knobs, LHS, Load, NEST, Preload, ROLE, Store, WEIGHT,
                                       BlockScheduleError, schedule_contraction,
                                       schedule_interface_program)

PACKAGES = artifacts_dir() / "targets" / "gemmini"

#: Each measured variant and the knob values that should reproduce it.
VARIANTS = {
    "gemmini_xdsl_rtl_v0": Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST),
    "gemmini_xdsl_rtl_v1_l1": Knobs(load_on_index_change=True, lookahead_steps=0, load_grouping=NEST),
    "gemmini_xdsl_rtl_v1_hoist": Knobs(lookahead_steps=None, load_grouping=NEST),
    "gemmini_xdsl_rtl_v1_grp": Knobs(lookahead_steps=None, load_grouping=ROLE,
                                     operand_order=(LHS, WEIGHT)),
    "gemmini_xdsl_rtl_v1_la1": Knobs(lookahead_steps=1, load_grouping=ROLE,
                                     operand_order=(LHS, WEIGHT)),
    "gemmini_xdsl_rtl_v1_la1b": Knobs(lookahead_steps=1, load_grouping=ROLE,
                                      operand_order=(LHS, WEIGHT), placement=BANK_ALIGNED),
    "gemmini_xdsl_rtl_v1_la1_pafter": Knobs(lookahead_steps=1, load_grouping=ROLE,
                                            operand_order=(LHS, WEIGHT), placement=CONTIGUOUS),
}
#: The corpus shapes, by the capsules that carry them. The last two are deep enough that the operand
#: regions overlap, which is where an unchecked knob value goes wrong.
SHAPES = {"A0/A2/A5/C5/C6/GS0": (16, 16, 16), "A3/B1": (16, 32, 16), "C0/C1": (16, 64, 64),
          "C2/C3/C4": (16, 64, 16), "GM0": (16, 6144, 16), "GM1": (16, 8208, 16)}
#: The geometry these packages were generated for, as this target's facts derive it.
GEOMETRY = Geometry(block=16, operand_rows=16384, operand_bank_rows=4096, accumulator_rows=1024,
                    separate_accumulator_space=True)

_ADDR = (1 << 32) - 1
_GARBAGE = (1 << 32) - 1
_ACC_ROW = (1 << 29) - 1          # bits 31/30/29 of an accumulator address are readout flags
_ACC_ACCUMULATE = 1 << 30

_DUMP = '''
import json, sys
from ir_ingest import InterfaceProgram, TensorSpec
from lowering import isa
out = {}
for label, (m, k, n) in json.loads(sys.argv[1]).items():
    program = InterfaceProgram(module=None)
    program.tensors = {"A0": TensorSpec("A0", [m, k], "i8", "input"),
                       "W": TensorSpec("W", [k, n], "i8", "weight"),
                       "Y0": TensorSpec("Y0", [m, n], "i32", "output")}
    encode = lambda v: ({"tensor": v.tensor, "offset": v.offset}
                        if isinstance(v, isa.Address) else v)
    out[label] = [{"name": i.name, "rs1": encode(i.rs1), "rs2": encode(i.rs2)}
                  for i in isa._matmul_trace(program, "A0", "W", "Y0", {"epilogue": []})]
print(json.dumps(out))
'''


@pytest.fixture(scope="module")
def dumped(tmp_path_factory):
    """``{package: {shape label: instruction stream}}`` from each package's own lowering."""
    if not PACKAGES.is_dir():
        pytest.skip(f"no generated packages under {PACKAGES}")
    program = tmp_path_factory.mktemp("dump") / "dump_trace.py"
    program.write_text(_DUMP)
    streams = {}
    for name in VARIANTS:
        package = PACKAGES / name / "mlir_oot"
        if not package.is_dir():
            continue
        done = subprocess.run([sys.executable, str(program), json.dumps(SHAPES)],
                              env={"PYTHONPATH": str(package), "PATH": "/usr/bin:/bin"},
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, f"{name}: {done.stderr[-400:]}"
        streams[name] = json.loads(done.stdout)
    if not streams:
        pytest.skip("none of the measured package variants is present in this checkout")
    return streams


def _package_blocks(stream, m, k, n):
    """Decode one package instruction stream back into block moves, in ELEMENT coordinates."""
    elem, out_elem = 1, 4
    lhs_stride = ((k + 15) // 16) * 16 * elem
    weight_stride = ((n + 15) // 16) * 16 * elem
    out_stride = ((n + 15) // 16) * 16 * out_elem
    blocks = []
    for instruction in stream:
        name, rs1, rs2 = instruction["name"], instruction["rs1"], instruction["rs2"]
        if name == "MVIN":
            row, cols, rows = rs2 & _ADDR, (rs2 >> 32) & 0xFFFF, rs2 >> 48
            stride = lhs_stride if rs1["tensor"] == "A0" else weight_stride
            role = LHS if rs1["tensor"] == "A0" else WEIGHT
            blocks.append(("load", role, rs1["offset"] // stride, (rs1["offset"] % stride) // elem,
                           rows, cols, row))
        elif name == "PRELOAD":
            weight_row, destination = rs1 & _ADDR, rs2
            blocks.append(("preload", None if weight_row == _GARBAGE else weight_row,
                           (destination & _ADDR) & _ACC_ROW, bool(destination & _ACC_ACCUMULATE),
                           destination >> 48, (destination >> 32) & 0xFFFF))
        elif name in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
            blocks.append(("compute", rs1 & _ADDR, rs1 >> 48, (rs1 >> 32) & 0xFFFF,
                           name == "COMPUTE_PRELOADED"))
        elif name == "MVOUT":
            blocks.append(("store", rs1["offset"] // out_stride,
                           (rs1["offset"] % out_stride) // out_elem, rs2 >> 48,
                           (rs2 >> 32) & 0xFFFF, (rs2 & _ADDR) & _ACC_ROW))
    return blocks


def _pass_blocks(schedule):
    blocks = []
    for op in schedule.ops:
        if isinstance(op, Load):
            blocks.append(("load", op.role, op.dram_row, op.dram_col, op.rows, op.cols, op.row))
        elif isinstance(op, Preload):
            blocks.append(("preload", op.weight_row, op.accumulator_row, op.accumulate, op.rows,
                           op.cols))
        elif isinstance(op, Compute):
            blocks.append(("compute", op.input_row, op.rows, op.cols, op.fresh_weights))
        elif isinstance(op, Store):
            blocks.append(("store", op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
    return blocks


#: (package, shape) the pass REFUSES, with the reason. Both are the deep reduction whose operand
#: regions overlap: the package emits a stream that reads an overwritten block.
REFUSED = {("gemmini_xdsl_rtl_v1_hoist", "GM1"), ("gemmini_xdsl_rtl_v1_grp", "GM1")}


@pytest.mark.parametrize("package", sorted(VARIANTS))
@pytest.mark.parametrize("label", sorted(SHAPES))
def test_the_pass_reproduces_each_measured_variant(dumped, package, label):
    if package not in dumped:
        pytest.skip(f"{package} is not in this checkout")
    m, k, n = SHAPES[label]
    want = _package_blocks(dumped[package][label], m, k, n)
    if (package, label) in REFUSED:
        with pytest.raises(BlockScheduleError, match="still live"):
            schedule_contraction(Contraction(m, k, n, "A0", "W", "Y0"), GEOMETRY, VARIANTS[package])
        return
    got = _pass_blocks(schedule_contraction(Contraction(m, k, n, "A0", "W", "Y0"), GEOMETRY,
                                            VARIANTS[package]))
    assert got == want, f"{package} {label}: first difference at " + str(
        next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want))))


def test_the_adapter_consumes_a_real_capsule_command_buffer(dumped):
    """End to end on the artifact a backend actually lowers: a command buffer a graded run emitted.

    The adapter's input is the interface program's own tensors and commands, so a real buffer must
    schedule to exactly what the contraction-level call produces for the same shape.
    """
    if "gemmini_xdsl_rtl_v1_l1" not in dumped:
        pytest.skip("the reference variant is not in this checkout")
    buffers = []
    for index, path in enumerate(runs_dir().glob("*/voyager-h2h/runs/*/*/generated/command_buffer.json")):
        if index >= 200:                      # bounded: the runs tree is large
            break
        buffers.append(path)
    document = None
    for path in buffers:
        candidate = json.loads(path.read_text())
        commits = [c for c in candidate.get("commands", []) if c.get("opcode") == "COMMIT"]
        if len(commits) == 1 and not (commits[0].get("attributes", {}) or {}).get("epilogue"):
            document = candidate
            break
    if document is None:
        pytest.skip("no graded run with a single plain COMMIT in this checkout")
    knobs = VARIANTS["gemmini_xdsl_rtl_v1_l1"]
    schedules = schedule_interface_program(document["tensors"], document["commands"], GEOMETRY, knobs)
    assert len(schedules) == 1
    lhs = document["tensors"][schedules[0].contraction.lhs]["shape"]
    weight = document["tensors"][schedules[0].contraction.weight]["shape"]
    direct = schedule_contraction(Contraction(lhs[0], lhs[1], weight[1]), GEOMETRY, knobs)
    assert _pass_blocks(schedules[0]) == _pass_blocks(direct)


@pytest.mark.parametrize("package", sorted(p for p, _ in REFUSED))
def test_what_the_pass_refuses_is_a_real_hazard_in_the_variant_it_refuses(dumped, package):
    """The refusals are not conservatism: the package's own stream loads an input block into the rows
    a later preload reads as weights, so the arithmetic would run on the wrong bytes."""
    if package not in dumped:
        pytest.skip(f"{package} is not in this checkout")
    m, k, n = SHAPES["GM1"]
    blocks = _package_blocks(dumped[package]["GM1"], m, k, n)
    overlap_row = GEOMETRY.operand_rows - ((k + 15) // 16) * GEOMETRY.block
    overwrite = next(i for i, b in enumerate(blocks)
                     if b[0] == "load" and b[1] == LHS and b[6] == overlap_row)
    read_as_weights = next(i for i, b in enumerate(blocks)
                           if b[0] == "preload" and b[1] == overlap_row)
    assert overwrite < read_as_weights, (
        f"{package}: expected the input load at row {overlap_row} to precede the preload that reads "
        "that row as weights")
