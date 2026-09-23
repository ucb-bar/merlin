"""The library block-schedule pass reproduces the hand-written package variants it was lifted from.

Each variant under ``out/artifacts/targets/gemmini/`` is one measured scheduling policy (the capsule
head-to-head's Verilator and FPGA tables). This test runs each package's OWN lowering in a subprocess
(the packages vendor their dependencies, so they import cleanly with only their ``mlir_oot`` on the
path), decodes its instruction stream back to block moves, and requires the pass with the matching
knob values to produce exactly that stream. The packages are read, never imported into this session
and never modified.

Two properties make that claim hold off this working copy:

* **The geometry is derived, not typed.** The pass is run on ``geometry_from_address_space`` of the
  target's own facts, and that geometry is checked against the constants each package baked in
  (``DIM``, ``SPAD_ROWS``, ``SPAD_BANK_ROWS``, ``ACC_ROWS``). A literal geometry here would let the
  derivation drift away from 18 frozen packages while every cell stayed green.
* **The packages are not tracked**, so a fresh clone has none of them. Each package stream is recorded
  as a digest in ``merlin/tests/data/block_schedule/golden_streams.json`` (regenerate with
  ``.venv/bin/python merlin/tests/gemmini/test_block_schedule_matches_packages.py --regen``), and the
  pass is held to the golden unconditionally; where the packages ARE present they are also held to it,
  so a golden can never go stale silently.

Two variants do NOT match on the deepest shape, and that is the point of the pass: there the two
operand regions overlap, and hoisting every load (or grouping them by operand) overwrites an input
block before the preload that reads it as weights. The pass refuses; the hand-written variant emits
it, and an in-order executor would compute on the wrong bytes without failing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys

import pytest

from merlin.common.paths import artifacts_dir, merlin_dir, runs_dir
from merlin.compile.scheduling import (
    BANK_ALIGNED,
    CONTIGUOUS,
    LHS,
    NEST,
    ROLE,
    WEIGHT,
    BlockScheduleError,
    Compute,
    Contraction,
    ConvContraction,
    Geometry,
    Knobs,
    Load,
    Preload,
    Store,
    geometry_from_address_space,
    schedule_contraction,
    schedule_convolution,
    schedule_interface_program,
)
from merlin.targetgen.address_space import derive_address_space

TARGET = "gemmini"
PACKAGES = artifacts_dir() / "targets" / TARGET
GOLDEN = merlin_dir() / "tests" / "data" / "block_schedule" / "golden_streams.json"

#: Each measured variant and the knob values that should reproduce it.
VARIANTS = {
    "gemmini_xdsl_rtl_v0": Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST),
    "gemmini_xdsl_rtl_v1_l1": Knobs(load_on_index_change=True, lookahead_steps=0, load_grouping=NEST),
    "gemmini_xdsl_rtl_v1_hoist": Knobs(lookahead_steps=None, load_grouping=NEST),
    "gemmini_xdsl_rtl_v1_grp": Knobs(lookahead_steps=None, load_grouping=ROLE, operand_order=(LHS, WEIGHT)),
    "gemmini_xdsl_rtl_v1_la1": Knobs(lookahead_steps=1, load_grouping=ROLE, operand_order=(LHS, WEIGHT)),
    "gemmini_xdsl_rtl_v1_la1b": Knobs(
        lookahead_steps=1, load_grouping=ROLE, operand_order=(LHS, WEIGHT), placement=BANK_ALIGNED
    ),
    "gemmini_xdsl_rtl_v1_la1_pafter": Knobs(
        lookahead_steps=1, load_grouping=ROLE, operand_order=(LHS, WEIGHT), placement=CONTIGUOUS
    ),
}
#: The corpus shapes, by the capsules that carry them. The last two are deep enough that the operand
#: regions overlap, which is where an unchecked knob value goes wrong.
SHAPES = {
    "A0/A2/A5/C5/C6/GS0": (16, 16, 16),
    "A3/B1": (16, 32, 16),
    "C0/C1": (16, 64, 64),
    "C2/C3/C4": (16, 64, 16),
    "GM0": (16, 6144, 16),
    "GM1": (16, 8208, 16),
}

#: (package, shape) the pass REFUSES, with the reason. Both are the deep reduction whose operand
#: regions overlap: the package emits a stream that reads an overwritten block.
REFUSED = {("gemmini_xdsl_rtl_v1_hoist", "GM1"), ("gemmini_xdsl_rtl_v1_grp", "GM1")}

#: The package carrying the correct (zero-path) convolution lowering, and the knob values it embodies:
#: every tap's pixel block is gathered again for every output-channel block, each load in its nest
#: position. The hoisted schedule (``load_on_index_change``) is not in any package; it is checked by
#: execution in the infra suite instead.
CONV_PACKAGE = "gemmini_xdsl_rtl_v1_convzero"
CONV_KNOBS = Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST)
#: Conv shapes: the public capsules' (8x8x4 input, 3x3 kernel, 8 output channels) under padding and
#: stride, plus shapes whose channel slices and output-channel blocks are ragged and multi-block, a
#: dilated one, a batch of two, and a 1x1 -- which is where an ordering or gather error would show.
CONVS = {
    "GC0_k3": ConvContraction(1, 8, 8, 4, 3, 3, 8),
    "GC7_k3_pad1": ConvContraction(1, 8, 8, 4, 3, 3, 8, padding=(1, 1, 1, 1)),
    "GC8_k3_stride2": ConvContraction(1, 8, 8, 4, 3, 3, 8, stride=(2, 2)),
    "ragged_ci20_co40_pad1": ConvContraction(1, 8, 8, 20, 3, 3, 40, padding=(1, 1, 1, 1)),
    "dilated_pad2": ConvContraction(1, 9, 9, 5, 3, 3, 17, padding=(2, 2, 2, 2), dilation=(2, 2)),
    "batch2": ConvContraction(2, 5, 6, 3, 3, 3, 18, padding=(0, 1, 1, 0)),
    "one_by_one": ConvContraction(1, 6, 6, 33, 1, 1, 20),
}

#: Package constant -> the Geometry field it must equal.
_CONSTANTS = {
    "DIM": "block",
    "SPAD_ROWS": "operand_rows",
    "SPAD_BANK_ROWS": "operand_bank_rows",
    "ACC_ROWS": "accumulator_rows",
}

_ADDR = (1 << 32) - 1
_GARBAGE = (1 << 32) - 1
_ACC_ROW = (1 << 29) - 1  # bits 31/30/29 of an accumulator address are readout flags
_ACC_ACCUMULATE = 1 << 30

_DUMP = """
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
"""


_DUMP_CONV = """
import json, sys
from ir_ingest import InterfaceProgram, TensorSpec
from lowering import isa
out = {}
for label, c in json.loads(sys.argv[1]).items():
    oh = (c["in_h"] + c["padding"][0] + c["padding"][2] - (c["dilation"][0] * (c["kh"] - 1) + 1)) // c["stride"][0] + 1
    ow = (c["in_w"] + c["padding"][1] + c["padding"][3] - (c["dilation"][1] * (c["kw"] - 1) + 1)) // c["stride"][1] + 1
    program = InterfaceProgram(module=None)
    program.tensors = {"IFM": TensorSpec("IFM", [c["batch"], c["in_h"], c["in_w"], c["ci"]], "i8", "input"),
                       "W": TensorSpec("W", [c["kh"] * c["kw"] * c["ci"], c["co"]], "i8", "weight"),
                       "Y0": TensorSpec("Y0", [c["batch"] * oh * ow, c["co"]], "i32", "output")}
    command = {"opcode": "CONV2D", "operands": {"ifm": "IFM", "weight": "W_res", "dst": "Y0"},
               "attributes": {"kernel": [c["kh"], c["kw"], c["ci"], c["co"]], "stride": c["stride"],
                              "padding": c["padding"], "dilation": c["dilation"], "layout": "nhwc",
                              "epilogue": []}}
    encode = lambda v: ({"tensor": v.tensor, "offset": v.offset}
                        if isinstance(v, isa.Address) else v)
    out[label] = [{"name": i.name, "rs1": encode(i.rs1), "rs2": encode(i.rs2)}
                  for i in isa._conv_loop_trace(program, command, "W")]
print(json.dumps(out))
"""


def _conv_json() -> str:
    return json.dumps(
        {
            label: {
                "batch": c.batch,
                "in_h": c.in_h,
                "in_w": c.in_w,
                "ci": c.ci,
                "kh": c.kh,
                "kw": c.kw,
                "co": c.co,
                "stride": list(c.stride),
                "padding": list(c.padding),
                "dilation": list(c.dilation),
            }
            for label, c in CONVS.items()
        }
    )


def _geometry() -> Geometry:
    return geometry_from_address_space(derive_address_space(TARGET))


def _golden() -> dict:
    if not GOLDEN.is_file():
        pytest.fail(f"missing {GOLDEN}; regenerate it with this file's --regen")
    return json.loads(GOLDEN.read_text())


def _digest(blocks) -> str:
    return hashlib.sha256(json.dumps(blocks, separators=(",", ":")).encode()).hexdigest()


def _canonical(blocks) -> list:
    return json.loads(json.dumps(blocks))  # tuples -> lists, exactly as the golden stores them


def package_constants(isa_source: str) -> dict[str, int]:
    """The integer module constants a package's ``lowering/isa.py`` bakes, evaluated structurally.

    Only module-level ``NAME = <int expression>`` over ints and earlier names; anything else is skipped
    rather than guessed."""
    env: dict[str, int] = {}
    ops = {
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mult: lambda a, b: a * b,
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.LShift: lambda a, b: a << b,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("not a plain integer expression")

    for stmt in ast.parse(isa_source).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            try:
                env[stmt.targets[0].id] = _eval(stmt.value)
            except (ValueError, ZeroDivisionError):
                continue
    return env


def _run_package(name: str, source: str, argument: str) -> dict | None:
    """Run ``source`` against package ``name``'s own lowering in a subprocess; ``None`` if absent."""
    import tempfile

    package = PACKAGES / name / "mlir_oot"
    if not package.is_dir():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        program = f"{tmp}/dump_trace.py"
        with open(program, "w") as handle:
            handle.write(source)
        done = subprocess.run(
            [sys.executable, program, argument],
            env={"PYTHONPATH": str(package), "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
            timeout=600,
        )
    if done.returncode != 0:
        raise RuntimeError(f"{name}: {done.stderr[-400:]}")
    return json.loads(done.stdout)


def _dump_packages() -> dict:
    """``{package: {shape label: instruction stream}}`` from each present package's own lowering."""
    streams = {}
    for name in VARIANTS:
        dumped = _run_package(name, _DUMP, json.dumps(SHAPES))
        if dumped is not None:
            streams[name] = dumped
    return streams


def _dump_conv() -> dict | None:
    return _run_package(CONV_PACKAGE, _DUMP_CONV, _conv_json())


@pytest.fixture(scope="module")
def dumped():
    if not PACKAGES.is_dir():
        pytest.skip(f"no generated packages under {PACKAGES}")
    streams = _dump_packages()
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
            blocks.append(("load", role, rs1["offset"] // stride, (rs1["offset"] % stride) // elem, rows, cols, row))
        elif name == "PRELOAD":
            weight_row, destination = rs1 & _ADDR, rs2
            blocks.append(
                (
                    "preload",
                    None if weight_row == _GARBAGE else weight_row,
                    (destination & _ADDR) & _ACC_ROW,
                    bool(destination & _ACC_ACCUMULATE),
                    destination >> 48,
                    (destination >> 32) & 0xFFFF,
                )
            )
        elif name in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
            blocks.append(("compute", rs1 & _ADDR, rs1 >> 48, (rs1 >> 32) & 0xFFFF, name == "COMPUTE_PRELOADED"))
        elif name == "MVOUT":
            blocks.append(
                (
                    "store",
                    rs1["offset"] // out_stride,
                    (rs1["offset"] % out_stride) // out_elem,
                    rs2 >> 48,
                    (rs2 >> 32) & 0xFFFF,
                    (rs2 & _ADDR) & _ACC_ROW,
                )
            )
    return blocks


def _pass_blocks(schedule):
    blocks = []
    for op in schedule.ops:
        if isinstance(op, Load):
            blocks.append(("load", op.role, op.dram_row, op.dram_col, op.rows, op.cols, op.row))
        elif isinstance(op, Preload):
            blocks.append(("preload", op.weight_row, op.accumulator_row, op.accumulate, op.rows, op.cols))
        elif isinstance(op, Compute):
            blocks.append(("compute", op.input_row, op.rows, op.cols, op.fresh_weights))
        elif isinstance(op, Store):
            blocks.append(("store", op.dram_row, op.dram_col, op.rows, op.cols, op.accumulator_row))
    return blocks


class DecodeError(AssertionError):
    """A package stream that does not decode into block moves -- e.g. a load with no stride config."""


def _conv_package_blocks(stream, conv: ConvContraction):
    """Decode a package convolution stream into block moves, CHECKING the configs rather than dropping them.

    Every load must be introduced by a ``CONFIG_LD`` carrying its own tensor's stride: the weight MVIN by
    the weight stride, and each gathered run of one-row MVINs by the input stride. A load a config does not
    introduce, a config with the wrong stride, or a gathered run broken by anything else fails the decode --
    so a schedule change that moved the configs is caught here, not after the fact on RTL. An MVIN whose
    source is the integer 0 is the target's zero path and decodes to a ``None`` gather entry.
    """
    ifm_stride = ((conv.ci + 15) // 16) * 16  # i8 input, padded to whole tiles
    weight_stride = ((conv.co + 15) // 16) * 16  # i8 weight
    out_stride = ((conv.co + 15) // 16) * 16 * 4  # i32 output
    names = [i["name"] for i in stream[:2]]
    if names != ["CONFIG_EX", "CONFIG_ST"]:
        raise DecodeError(f"expected the CONFIG_EX, CONFIG_ST prologue, got {names}")
    blocks, stride, run = [], None, None
    for instruction in stream[2:]:
        name, rs1, rs2 = instruction["name"], instruction["rs1"], instruction["rs2"]
        if name != "MVIN" and run is not None:
            blocks.append(("load", LHS, run["gather"], len(run["gather"]), run["cols"], run["row"]))
            run = None
        if name == "CONFIG_LD":
            if stride is not None:
                raise DecodeError("two CONFIG_LD with no load between them")
            stride = rs2
        elif name == "MVIN":
            row, cols, rows = rs2 & _ADDR, (rs2 >> 32) & 0xFFFF, rs2 >> 48
            if isinstance(rs1, dict) and rs1["tensor"] == "W":
                if stride != weight_stride or run is not None:
                    raise DecodeError(f"weight MVIN introduced by stride {stride}, not {weight_stride}")
                blocks.append(
                    ("load", WEIGHT, [rs1["offset"] // weight_stride, rs1["offset"] % weight_stride], rows, cols, row)
                )
                stride = None
                continue
            if rows != 1:
                raise DecodeError(f"a gathered input MVIN spans {rows} rows")
            entry = None if rs1 == 0 else [rs1["offset"] // ifm_stride, rs1["offset"] % ifm_stride]
            if run is None:
                if stride != ifm_stride:
                    raise DecodeError(f"input gather introduced by stride {stride}, not {ifm_stride}")
                run, stride = {"row": row, "cols": cols, "gather": []}, None
            elif row != run["row"] + len(run["gather"]) or cols != run["cols"]:
                raise DecodeError("a gathered run is not contiguous on chip")
            run["gather"].append(entry)
        elif stride is not None:
            raise DecodeError(f"CONFIG_LD not followed by a load (followed by {name})")
        elif name == "PRELOAD":
            weight_row, destination = rs1 & _ADDR, rs2
            blocks.append(
                (
                    "preload",
                    None if weight_row == _GARBAGE else weight_row,
                    (destination & _ADDR) & _ACC_ROW,
                    bool(destination & _ACC_ACCUMULATE),
                    destination >> 48,
                    (destination >> 32) & 0xFFFF,
                )
            )
        elif name in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
            blocks.append(("compute", rs1 & _ADDR, rs1 >> 48, (rs1 >> 32) & 0xFFFF, name == "COMPUTE_PRELOADED"))
        elif name == "MVOUT":
            blocks.append(
                (
                    "store",
                    rs1["offset"] // out_stride,
                    (rs1["offset"] % out_stride) // 4,
                    rs2 >> 48,
                    (rs2 >> 32) & 0xFFFF,
                    (rs2 & _ADDR) & _ACC_ROW,
                )
            )
        else:
            raise DecodeError(f"unexpected {name} inside a convolution stream")
    if run is not None:
        blocks.append(("load", LHS, run["gather"], len(run["gather"]), run["cols"], run["row"]))
    return blocks


def _conv_pass_blocks(schedule):
    blocks = []
    for op in schedule.ops:
        if isinstance(op, Load):
            source = list(op.gather) if op.gather is not None else [op.dram_row, op.dram_col]
            blocks.append(("load", op.role, source, op.rows, op.cols, op.row))
        else:
            blocks.extend(
                _pass_blocks(
                    type(schedule)((op,), schedule.contraction, schedule.geometry, schedule.knobs, schedule.regions)
                )
            )
    return blocks


def _hazard(blocks, k, geometry):
    """(index of the input load into the overlap row, index of the preload reading it as weights)."""
    overlap_row = geometry.operand_rows - ((k + 15) // 16) * geometry.block
    overwrite = next(i for i, b in enumerate(blocks) if b[0] == "load" and b[1] == LHS and b[6] == overlap_row)
    read_as_weights = next(i for i, b in enumerate(blocks) if b[0] == "preload" and b[1] == overlap_row)
    return overwrite, read_as_weights


# ------------------------------------------------------------------------------------ geometry


def test_the_geometry_is_derived_and_agrees_with_what_the_packages_baked():
    """Three copies of the same numbers -- the derivation, the golden record, the packages' constants
    -- held together, so none of them can drift while the stream cells stay green."""
    geometry = _geometry()
    golden = _golden()
    assert golden["geometry"] == {field: getattr(geometry, field) for field in golden["geometry"]}, (
        "the derived geometry moved away from the one the golden streams were recorded against"
    )
    for package, constants in golden["package_constants"].items():
        for name, value in constants.items():
            assert getattr(geometry, _CONSTANTS[name]) == value, (
                f"{package} baked {name} = {value}, but {TARGET}'s facts derive "
                f"{_CONSTANTS[name]} = {getattr(geometry, _CONSTANTS[name])}"
            )


def test_the_recorded_constants_are_still_the_packages_constants():
    present = {name: PACKAGES / name / "mlir_oot" / "lowering" / "isa.py" for name in VARIANTS}
    present = {name: path for name, path in present.items() if path.is_file()}
    if not present:
        pytest.skip("no measured package variant in this checkout")
    recorded = _golden()["package_constants"]
    for name, path in present.items():
        baked = package_constants(path.read_text())
        assert recorded[name] == {c: baked[c] for c in _CONSTANTS if c in baked}, name


# ------------------------------------------------------------------------------ stream cells


@pytest.mark.parametrize("package", sorted(VARIANTS))
@pytest.mark.parametrize("label", sorted(SHAPES))
def test_the_pass_reproduces_each_measured_variant(package, label):
    """Held to the recorded package stream, so it runs on any checkout."""
    m, k, n = SHAPES[label]
    cell = _golden()["cells"][package][label]
    contraction = Contraction(m, k, n, "A0", "W", "Y0")
    if (package, label) in REFUSED:
        assert cell["pass"] == "refused"
        with pytest.raises(BlockScheduleError, match="still live"):
            schedule_contraction(contraction, _geometry(), VARIANTS[package])
        return
    got = _canonical(_pass_blocks(schedule_contraction(contraction, _geometry(), VARIANTS[package])))
    assert (len(got), _digest(got)) == (cell["ops"], cell["sha256"]), (
        f"{package} {label}: the pass no longer reproduces the recorded package stream "
        f"(first ops {got[:3]} vs {cell['head'][:3]})"
    )


@pytest.mark.parametrize("package", sorted(VARIANTS))
def test_each_present_package_still_matches_its_golden(dumped, package):
    """A golden taken from bytes that have since changed is detected, not trusted."""
    if package not in dumped:
        pytest.skip(f"{package} is not in this checkout")
    for label, (m, k, n) in SHAPES.items():
        blocks = _canonical(_package_blocks(dumped[package][label], m, k, n))
        cell = _golden()["cells"][package][label]
        assert (len(blocks), _digest(blocks)) == (cell["package_ops"], cell["package_sha256"]), (
            f"{package} {label}: the package stream moved since the golden was recorded"
        )


@pytest.mark.parametrize("package", sorted(VARIANTS))
def test_where_the_package_is_present_the_first_difference_is_named(dumped, package):
    """Diagnostic companion: on a mismatch, say WHERE, which a digest cannot."""
    if package not in dumped:
        pytest.skip(f"{package} is not in this checkout")
    for label, (m, k, n) in SHAPES.items():
        if (package, label) in REFUSED:
            continue
        want = _package_blocks(dumped[package][label], m, k, n)
        got = _pass_blocks(schedule_contraction(Contraction(m, k, n, "A0", "W", "Y0"), _geometry(), VARIANTS[package]))
        assert got == want, f"{package} {label}: first difference at " + str(
            next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        )


@pytest.mark.parametrize("package", sorted(p for p, _ in REFUSED))
def test_what_the_pass_refuses_is_a_real_hazard_in_the_variant_it_refuses(package):
    """The refusals are not conservatism: the package's own stream loads an input block into the rows
    a later preload reads as weights, so the arithmetic would run on the wrong bytes."""
    hazard = _golden()["cells"][package]["GM1"]["hazard"]
    assert hazard["overwrite_index"] < hazard["read_as_weights_index"], (
        f"{package}: expected the input load at the overlap row to precede the preload that reads that row as weights"
    )


@pytest.mark.parametrize("label", sorted(CONVS))
def test_the_pass_reproduces_the_packaged_convolution(label):
    """The zero-path convolution lowering, reproduced op for op -- including where every stride config
    sits -- from the recorded stream, on any checkout."""
    cell = _golden()["conv_cells"][label]
    got = _canonical(_conv_pass_blocks(schedule_convolution(CONVS[label], _geometry(), CONV_KNOBS)))
    assert (len(got), _digest(got)) == (cell["ops"], cell["sha256"]), (
        f"{label}: the pass no longer reproduces the recorded package convolution stream"
    )


def test_the_present_conv_package_still_matches_its_golden_and_the_pass():
    dumped = _dump_conv()
    if dumped is None:
        pytest.skip(f"{CONV_PACKAGE} is not in this checkout")
    for label, conv in CONVS.items():
        want = _conv_package_blocks(dumped[label], conv)
        cell = _golden()["conv_cells"][label]
        canonical = _canonical(want)
        assert (len(canonical), _digest(canonical)) == (cell["package_ops"], cell["package_sha256"]), label
        got = _canonical(_conv_pass_blocks(schedule_convolution(conv, _geometry(), CONV_KNOBS)))
        assert got == canonical, f"{label}: first difference at " + str(
            next((i for i, (a, b) in enumerate(zip(got, canonical)) if a != b), min(len(got), len(canonical)))
        )


def test_the_conv_decoder_rejects_a_load_its_config_does_not_introduce():
    """The CONFIG_LD placement is part of what is checked: move one and the decode fails."""
    dumped = _dump_conv()
    if dumped is None:
        pytest.skip(f"{CONV_PACKAGE} is not in this checkout")
    stream = list(dumped["GC7_k3_pad1"])
    first_config = next(i for i, x in enumerate(stream) if x["name"] == "CONFIG_LD")
    with pytest.raises(DecodeError):
        _conv_package_blocks(stream[:first_config] + stream[first_config + 1 :], CONVS["GC7_k3_pad1"])


def test_the_adapter_consumes_a_real_capsule_command_buffer():
    """End to end on the artifact a backend actually lowers: a command buffer a graded run emitted.

    The adapter's input is the interface program's own tensors and commands, so a real buffer must
    schedule to exactly what the contraction-level call produces for the same shape.
    """
    buffers = []
    for index, path in enumerate(runs_dir().glob("*/voyager-h2h/runs/*/*/generated/command_buffer.json")):
        if index >= 200:  # bounded: the runs tree is large
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
    geometry = _geometry()
    schedules = schedule_interface_program(document["tensors"], document["commands"], geometry, knobs)
    assert len(schedules) == 1
    lhs = document["tensors"][schedules[0].contraction.lhs]["shape"]
    weight = document["tensors"][schedules[0].contraction.weight]["shape"]
    direct = schedule_contraction(Contraction(lhs[0], lhs[1], weight[1]), geometry, knobs)
    assert _pass_blocks(schedules[0]) == _pass_blocks(direct)


# --------------------------------------------------------------------------------- regenerate


def regenerate() -> None:
    """Re-record the golden from the packages present in this checkout. Refuses when any variant is
    missing, and when any non-refused cell's pass stream disagrees with its package: a golden is a
    record of agreement, never of whatever the pass happens to produce today."""
    streams = _dump_packages()
    missing = sorted(set(VARIANTS) - set(streams))
    if missing:
        raise SystemExit(f"cannot regenerate: packages missing from this checkout: {missing}")
    geometry = _geometry()
    cells: dict = {}
    constants: dict = {}
    sources: dict = {}
    for package in VARIANTS:
        isa = PACKAGES / package / "mlir_oot" / "lowering" / "isa.py"
        baked = package_constants(isa.read_text())
        constants[package] = {c: baked[c] for c in _CONSTANTS if c in baked}
        sources[package] = hashlib.sha256(isa.read_bytes()).hexdigest()
        cells[package] = {}
        for label, (m, k, n) in SHAPES.items():
            blocks = _canonical(_package_blocks(streams[package][label], m, k, n))
            cell = {"package_ops": len(blocks), "package_sha256": _digest(blocks)}
            if (package, label) in REFUSED:
                overwrite, read = _hazard(blocks, k, geometry)
                cell.update(
                    {"pass": "refused", "hazard": {"overwrite_index": overwrite, "read_as_weights_index": read}}
                )
            else:
                got = _canonical(
                    _pass_blocks(
                        schedule_contraction(Contraction(m, k, n, "A0", "W", "Y0"), geometry, VARIANTS[package])
                    )
                )
                if got != blocks:
                    raise SystemExit(f"cannot regenerate: {package} {label} disagrees with its package")
                cell.update({"ops": len(got), "sha256": _digest(got), "head": got[:6]})
            cells[package][label] = cell
    conv_streams = _dump_conv()
    if conv_streams is None:
        raise SystemExit(f"cannot regenerate: {CONV_PACKAGE} missing from this checkout")
    conv_isa = PACKAGES / CONV_PACKAGE / "mlir_oot" / "lowering" / "isa.py"
    baked = package_constants(conv_isa.read_text())
    constants[CONV_PACKAGE] = {c: baked[c] for c in _CONSTANTS if c in baked}
    sources[CONV_PACKAGE] = hashlib.sha256(conv_isa.read_bytes()).hexdigest()
    conv_cells = {}
    for label, conv in CONVS.items():
        want = _canonical(_conv_package_blocks(conv_streams[label], conv))
        got = _canonical(_conv_pass_blocks(schedule_convolution(conv, geometry, CONV_KNOBS)))
        if got != want:
            raise SystemExit(f"cannot regenerate: conv {label} disagrees with {CONV_PACKAGE}")
        conv_cells[label] = {
            "package_ops": len(want),
            "package_sha256": _digest(want),
            "ops": len(got),
            "sha256": _digest(got),
            "head": got[:6],
        }
    document = {
        "schema": "block-schedule-golden/v2",
        "target": TARGET,
        "what": (
            "each measured package variant's matmul block-move stream, decoded by "
            "_package_blocks, per corpus shape; the pass is held to these digests"
        ),
        "geometry": {
            f: getattr(geometry, f)
            for f in ("block", "operand_rows", "operand_bank_rows", "accumulator_rows", "separate_accumulator_space")
        },
        "package_constants": constants,
        "package_isa_sha256": sources,
        "cells": cells,
        "conv_package": CONV_PACKAGE,
        "conv_cells": conv_cells,
    }
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    if "--regen" in sys.argv:
        regenerate()
