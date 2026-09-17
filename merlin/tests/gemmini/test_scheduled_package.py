"""A package minted from the scheduling pass emits the hand-written package's instruction stream.

``scheduled_backend_v0/scripts/mint.py`` copies a measured parent package, vendors the pass, and routes
the parent's two block-moving traces through it. The acceptance is the RAW instruction stream -- every
CONFIG_LD, every MVIN source and tile word, every PRELOAD/COMPUTE/MVOUT -- compared op for op with the
package whose hand edit it replaces, for each measured knob setting and every corpus shape. Where the pass
refuses a hand-written variant's shape (its regions overlap and the variant overwrites a live block), the
minted package must refuse too, rather than emit that stream.

Packages are minted into a temporary root and never into ``out/``; parents are read, never modified. The
whole file skips on a checkout without the (untracked) parent packages.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys

import pytest

from merlin.common.paths import artifacts_dir, merlin_dir
from merlin.compile.scheduling import BANK_ALIGNED, CONTIGUOUS, LHS, NEST, ROLE, WEIGHT, Knobs, block_schedule

TARGET = "gemmini"
PARENTS = artifacts_dir() / "targets" / TARGET
MINT = merlin_dir() / "experiments" / "scheduled_backend_v0" / "scripts" / "mint.py"

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
CONV_PARENT = "gemmini_xdsl_rtl_v1_convzero"
CONV_KNOBS = Knobs(load_on_index_change=False, lookahead_steps=0, load_grouping=NEST)
SHAPES = {
    "A0": (16, 16, 16),
    "A3": (16, 32, 16),
    "C0": (16, 64, 64),
    "C2": (16, 64, 16),
    "GM0": (16, 6144, 16),
    "GM1": (16, 8208, 16),
    "ragged": (37, 21, 45),
}
REFUSED = {("gemmini_xdsl_rtl_v1_hoist", "GM1"), ("gemmini_xdsl_rtl_v1_grp", "GM1")}
CONVS = {
    "k3": dict(batch=1, in_h=8, in_w=8, ci=4, kh=3, kw=3, co=8, stride=[1, 1], padding=[0, 0, 0, 0], dilation=[1, 1]),
    "k3_pad1_relu": dict(
        batch=1,
        in_h=8,
        in_w=8,
        ci=4,
        kh=3,
        kw=3,
        co=8,
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        epilogue=["relu"],
    ),
    "stride2": dict(
        batch=1, in_h=8, in_w=8, ci=4, kh=3, kw=3, co=8, stride=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1]
    ),
    "ragged": dict(
        batch=2, in_h=7, in_w=9, ci=20, kh=3, kw=3, co=40, stride=[1, 1], padding=[1, 1, 1, 1], dilation=[1, 1]
    ),
    "dilated": dict(
        batch=1, in_h=9, in_w=9, ci=5, kh=3, kw=3, co=17, stride=[1, 1], padding=[2, 2, 2, 2], dilation=[2, 2]
    ),
    "pooled": dict(
        batch=1,
        in_h=8,
        in_w=8,
        ci=4,
        kh=3,
        kw=3,
        co=16,
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        epilogue=["maxpool"],
        pool=True,
    ),
}

_DUMP = """
import json, sys
from ir_ingest import InterfaceProgram, TensorSpec
from lowering import isa
spec = json.loads(sys.argv[1])
encode = lambda v: {"tensor": v.tensor, "offset": v.offset} if isinstance(v, isa.Address) else v
def stream(fn):
    try:
        return [[i.name, encode(i.rs1), encode(i.rs2)] for i in fn()]
    except ValueError as e:
        return {"refused": str(e)}
out = {"matmul": {}, "matmul_relu": {}, "matmul_pool": {}, "conv": {}}
for label, (m, k, n) in spec["shapes"].items():
    p = InterfaceProgram(module=None)
    p.tensors = {"A0": TensorSpec("A0", [m, k], "i8", "input"), "W": TensorSpec("W", [k, n], "i8", "weight"),
                 "Y0": TensorSpec("Y0", [m, n], "i32", "output")}
    out["matmul"][label] = stream(lambda: isa._matmul_trace(p, "A0", "W", "Y0", {"epilogue": []}))
    relu = {"epilogue": ["relu"], "acc_scale": 0.5}
    out["matmul_relu"][label] = stream(lambda: isa._matmul_trace(p, "A0", "W", "Y0", relu))
    if m * n <= 64 * 64:
        out["matmul_pool"][label] = stream(lambda: isa._matmul_trace(
            p, "A0", "W", "Y0", {"epilogue": ["maxpool"], "pool_in_dims": [4, 4], "pool_size": [2, 2],
                                 "pool_stride": [2, 2]}))
for label, c in spec["convs"].items():
    oh = (c["in_h"] + c["padding"][0] + c["padding"][2] - (c["dilation"][0] * (c["kh"] - 1) + 1)) // c["stride"][0] + 1
    ow = (c["in_w"] + c["padding"][1] + c["padding"][3] - (c["dilation"][1] * (c["kw"] - 1) + 1)) // c["stride"][1] + 1
    p = InterfaceProgram(module=None)
    out_rows = (oh // 2) * (ow // 2) if c.get("pool") else c["batch"] * oh * ow
    p.tensors = {"IFM": TensorSpec("IFM", [c["batch"], c["in_h"], c["in_w"], c["ci"]], "i8", "input"),
                 "W": TensorSpec("W", [c["kh"] * c["kw"] * c["ci"], c["co"]], "i8", "weight"),
                 "Y0": TensorSpec("Y0", [out_rows, c["co"]], "i8" if c.get("pool") else "i32", "output")}
    attrs = {"kernel": [c["kh"], c["kw"], c["ci"], c["co"]], "stride": c["stride"], "padding": c["padding"],
             "dilation": c["dilation"], "layout": "nhwc", "epilogue": c.get("epilogue", [])}
    if c.get("pool"):
        attrs.update({"pool_in_dims": [oh, ow], "pool_size": [2, 2], "pool_stride": [2, 2]})
    command = {"opcode": "CONV2D", "operands": {"ifm": "IFM", "weight": "W_res", "dst": "Y0"}, "attributes": attrs}
    out["conv"][label] = stream(lambda: isa._conv_loop_trace(p, command, "W"))
print(json.dumps(out))
"""


def _mint_module():
    spec = importlib.util.spec_from_file_location("scheduled_backend_mint", MINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _streams(package):
    done = subprocess.run(
        [sys.executable, "-c", _DUMP, json.dumps({"shapes": SHAPES, "convs": CONVS})],
        env={"PYTHONPATH": str(package / "mlir_oot"), "PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-800:]
    return json.loads(done.stdout)


@pytest.fixture(scope="module")
def minted(tmp_path_factory):
    mint = _mint_module()
    present = [name for name in (*VARIANTS, CONV_PARENT) if (PARENTS / name / "mlir_oot").is_dir()]
    if not present:
        pytest.skip(f"no parent packages under {PARENTS}")
    root = tmp_path_factory.mktemp("minted")
    packages = {}
    for name in present:
        knobs = VARIANTS.get(name, Knobs())
        packages[name] = mint.mint(TARGET, name, knobs, CONV_KNOBS, name.rsplit("_rtl_", 1)[-1], dest_root=root)
    yield mint, packages
    for package in packages.values():
        if package.exists():
            mint.thaw(package)


def _first_difference(got, want):
    if isinstance(got, dict) or isinstance(want, dict):
        left = got if isinstance(got, dict) else "stream"
        right = want if isinstance(want, dict) else "stream"
        return f"refusal mismatch: {left} vs {right}"
    index = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
    return (
        f"first difference at op {index} of {len(got)}/{len(want)}: "
        f"{got[index] if index < len(got) else None} vs {want[index] if index < len(want) else None}"
    )


@pytest.mark.parametrize("parent", sorted(VARIANTS))
def test_a_minted_package_emits_its_parents_matmul_stream_op_for_op(minted, parent):
    mint, packages = minted
    if parent not in packages:
        pytest.skip(f"{parent} is not in this checkout")
    got, want = _streams(packages[parent]), _streams(PARENTS / parent)
    for kind in ("matmul", "matmul_relu", "matmul_pool"):
        for label in want[kind]:
            if kind != "matmul_pool" and (parent, label) in REFUSED:
                assert "refused" in got[kind][label], f"{parent} {kind} {label}: the minted package emitted a stream"
                assert "still live" in got[kind][label]["refused"]
                continue
            assert got[kind][label] == want[kind][label], f"{parent} {kind} {label}: " + _first_difference(
                got[kind][label], want[kind][label]
            )


def test_a_minted_package_emits_the_zero_path_convolution_op_for_op(minted):
    mint, packages = minted
    if CONV_PARENT not in packages:
        pytest.skip(f"{CONV_PARENT} is not in this checkout")
    got, want = _streams(packages[CONV_PARENT]), _streams(PARENTS / CONV_PARENT)
    for label in CONVS:
        assert got["conv"][label] == want["conv"][label], f"conv {label}: " + _first_difference(
            got["conv"][label], want["conv"][label]
        )


def test_the_vendored_pass_is_merlins_bytes_and_the_lineage_says_so(minted):
    import yaml

    mint, packages = minted
    pass_bytes = open(block_schedule.__file__, "rb").read()
    for name, package in packages.items():
        vendored = (package / "mlir_oot" / "lowering" / "block_schedule.py").read_bytes()
        assert vendored == pass_bytes, name
        lineage = yaml.safe_load((package / "manifest.yaml").read_text())["lineage"]
        assert lineage["parent_package_id"] == name
        assert lineage["pass_module_sha256"] == hashlib.sha256(pass_bytes).hexdigest()
        assert lineage["source_digest"] == mint.source_digest(package)
        assert package.name.endswith(lineage["source_digest"][:12])
        mint.verify(package)


def test_a_minted_package_passes_the_integrity_scan(minted):
    """Vendoring the pass must not make the package import merlin."""
    from merlin.targetgen import oot_runner

    mint, packages = minted
    for package in packages.values():
        oot_runner.integrity_scan(oot_runner.load_package(package))


def test_minting_is_idempotent_and_refuses_a_parent_with_a_different_geometry(minted, tmp_path):
    mint, packages = minted
    name = next(iter(packages))
    again = mint.mint(
        TARGET,
        name,
        VARIANTS.get(name, Knobs()),
        CONV_KNOBS,
        name.rsplit("_rtl_", 1)[-1],
        dest_root=packages[name].parent,
    )
    assert again == packages[name]
    fake_root = tmp_path / "parents"
    import shutil

    shutil.copytree(PARENTS / name, fake_root / name, ignore=shutil.ignore_patterns("__pycache__"))
    mint.thaw(fake_root / name)
    isa = fake_root / name / "mlir_oot" / "lowering" / "isa.py"
    isa.write_text(isa.read_text().replace("SPAD_ROWS = 262144 // DIM", "SPAD_ROWS = 131072 // DIM", 1))
    with pytest.raises(SystemExit, match="different geometry"):
        mint.mint(TARGET, name, Knobs(), CONV_KNOBS, "bad", dest_root=tmp_path / "out", parents_root=fake_root)


def _rows_moved(stream, convs_spec):
    """Expand a raw conv stream to (on-chip row, DRAM source) per row plus the non-load instructions, so a
    stream of one-row MVINs and a stream of multi-row runs can be compared for the bytes they move."""
    ci = convs_spec["ci"]
    ifm_stride = ((ci + 15) // 16) * 16
    stride, out = None, []
    for name, rs1, rs2 in stream:
        if name == "CONFIG_LD":
            stride = rs2
        elif name == "MVIN" and (rs1 == 0 or rs1["tensor"] == "IFM"):
            row, cols, rows = rs2 & 0xFFFFFFFF, (rs2 >> 32) & 0xFFFF, rs2 >> 48
            for i in range(rows):
                source = (
                    None
                    if rs1 == 0
                    else (rs1["offset"] // ifm_stride + i * (stride // ifm_stride), rs1["offset"] % ifm_stride)
                )
                out.append(("row", row + i, cols, source))
        elif name != "CONFIG_LD":
            out.append((name, json.dumps(rs1, sort_keys=True), rs2))
    return out


def test_coalesced_gathers_move_the_same_rows_in_fewer_loads(minted, tmp_path):
    """The multi-row rendering changes how many MVINs a gather takes, never which bytes reach which row."""
    mint, packages = minted
    if CONV_PARENT not in packages:
        pytest.skip(f"{CONV_PARENT} is not in this checkout")
    coalesced = mint.mint(
        TARGET, CONV_PARENT, Knobs(), CONV_KNOBS, "coalesced", dest_root=tmp_path, coalesce_gather=True
    )
    try:
        got, want = _streams(coalesced), _streams(PARENTS / CONV_PARENT)
        for label, spec in CONVS.items():
            if spec.get("pool") or spec.get("epilogue"):
                continue
            assert _rows_moved(got["conv"][label], spec) == _rows_moved(want["conv"][label], spec), label
            loads = sum(1 for name, _, _ in got["conv"][label] if name == "MVIN")
            assert loads < sum(1 for name, _, _ in want["conv"][label] if name == "MVIN"), label
    finally:
        mint.thaw(coalesced)
