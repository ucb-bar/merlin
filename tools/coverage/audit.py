#!/usr/bin/env python3
"""audit_compile.py — classify per-dispatch backend routing + opcode coverage.

For each cell under `build/compile_audit/<model>_<hw>[_im2col]/`:

1. Parse `configs/configured_module_*.mlir` to extract per-dispatch:
   - dispatch ordinal + op_kind (matmul/conv/elementwise/pack/...)
   - translation_info pass_pipeline string
   - whether the routing string indicates accelerator lowering

2. Disassemble `binaries/*_embedded_elf_riscv_64.so` with
   riscv64-zephyr-elf-objdump and tally instruction classes:
   - scalar (default)
   - rvv  (vsetvli, vle*, vse*, vmul, vmacc, vadd.vv, ...)
   - rocc custom-3 (opcode bits[6:0] == 0x7b)
     broken down by funct7 (k_CONFIG=0, k_MVIN=2, k_MVOUT=3,
     k_COMPUTE_PRELOADED=4, k_COMPUTE_ACCUMULATE=5, k_PRELOAD=6,
     k_FLUSH=7, k_LOOP_WS=8, ...).

3. Classify each dispatch:
   - accel_routed_and_emitted   — translation_info had accel pipeline, ELF has
                                  expected accelerator opcodes
   - accel_routed_but_fell_back — translation_info had accel pipeline, ELF
                                  has NONE of those opcodes (real bug)
   - not_routed_eligible_op     — op is matmul/conv, expected accel, plugin
                                  did NOT route it (coverage gap)
   - not_routed_ineligible_op   — op is e.g. softmax/elementwise (correctly
                                  skipped)
   - scalar_baseline            — backend=scalar, only scalar opcodes expected

4. Emit per-cell `audit/coverage.json` and a top-level rollup table at
   `tmp/firesim_shuttle_compile_audit.{json,md}`.

Designed to run AFTER compile_audit.sh has populated the artifact tree.
No build dependencies beyond Python stdlib + the chipyard riscv64-zephyr-elf-
objdump being present.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path


# ---- toolchain locations -------------------------------------------------
# Resolution goes through tools/utils.py:find_toolchain_binary so we don't
# bake per-developer absolute paths into this file. Override via env vars:
#   MERLIN_RISCV_OBJDUMP        a riscv64 objdump binary
def resolve_objdump() -> Path:
    """Pick whichever riscv64 objdump is reachable."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import utils

    try:
        return utils.find_toolchain_binary(
            "riscv64-zephyr-elf-objdump",
            env_var="MERLIN_RISCV_OBJDUMP",
            aliases=("riscv64-unknown-elf-objdump",),
        )
    except FileNotFoundError:
        sys.exit(
            "[audit] no riscv64 objdump found; install zephyr-sdk or "
            "chipyard riscv-tools, or set $MERLIN_RISCV_OBJDUMP"
        )


# ---- accelerator detection heuristics ------------------------------------
# Strings in the configured-dispatch translation_info that indicate this
# dispatch was routed through a non-default codegen pipeline.
ACCEL_PIPELINE_MARKERS = {
    "gemmini": [
        "gemmini-lower-tile-to-isa",
        "gemmini-legalize-for-llvm-export",
    ],
    "gemmini_mx": [
        "gemmini-lower-tile-to-isa",  # same pipeline; differs only in mx-format option
    ],
    "opu": [
        # OPU's translation_info pipeline name is the generic
        # `DoubleTilingExpert` (a CPU pipeline) — the OPU-specific signal
        # lives in the dispatch BODY, not the pipeline string. See
        # ACCEL_BODY_MARKERS["opu"] below for the real routing test.
        # Kept here as defence-in-depth so legacy pipelines (e.g.
        # `Mmt4dTilingExpert`) still register.
        "Mmt4dTilingExpert",
    ],
    "rvv": [
        # RVV doesn't go through a plugin pipeline — it's pure LLVM
        # auto-vec. So at the MLIR level it's indistinguishable from
        # scalar. We rely on opcode-class detection only.
    ],
    "scalar": [],
}

# Substrings whose presence anywhere in a configured-dispatch's MLIR body
# means this dispatch was routed to the accelerator. This is checked in
# addition to ACCEL_PIPELINE_MARKERS because OPU's pipeline string is
# generic (DoubleTilingExpert) and Gemmini's dialect ops may not always
# be reflected in the pipeline name.
ACCEL_BODY_MARKERS = {
    "gemmini": ["gemmini.", "@llvm.riscv.gemmini", "llvm.intr.riscv.gemmini"],
    "gemmini_mx": ["gemmini.", "@llvm.riscv.gemmini", "llvm.intr.riscv.gemmini"],
    "opu": [
        'iree_codegen.ukernel.generic "iree_uk_opu_matmul"',
        '"iree_uk_opu_matmul"',
    ],
    "rvv": [],
    "scalar": [],
}

# Function-name patterns -> op_kind classification.
OP_KIND_PATTERNS = [
    (re.compile(r"matmul"), "matmul"),
    (re.compile(r"conv2d|conv_"), "conv2d"),
    (re.compile(r"pack|pack_"), "pack"),
    (re.compile(r"unpack"), "unpack"),
    (re.compile(r"depthwise"), "depthwise_conv"),
    (re.compile(r"reduce|softmax|argmax"), "reduce"),
    (re.compile(r"transpose"), "transpose"),
    (re.compile(r"reshape|broadcast"), "reshape"),
    (re.compile(r"elementwise|elwise"), "elementwise"),
    (re.compile(r"slice|extract|gather|scatter"), "data_move"),
    (re.compile(r"fill|copy|cast"), "memref_op"),
]


def classify_op_kind(func_name: str) -> str:
    """Map dispatch function name suffix to a coarse op kind."""
    for pat, kind in OP_KIND_PATTERNS:
        if pat.search(func_name):
            return kind
    return "other"


# Whether a given op_kind is *eligible* for accelerator lowering on
# Gemmini / OPU (i.e. whether we EXPECT a routed dispatch).
GEMMINI_ELIGIBLE = {"matmul"}  # plus conv2d if im2col is enabled
OPU_ELIGIBLE = {"matmul"}  # plus conv2d if im2col is enabled


def is_eligible(op_kind: str, hw: str, im2col: bool) -> bool:
    if hw == "scalar" or hw == "rvv":
        return False
    if hw in ("gemmini", "gemmini_mx"):
        eligible = set(GEMMINI_ELIGIBLE)
        if im2col:
            eligible.add("conv2d")
            eligible.add("depthwise_conv")
        return op_kind in eligible
    if hw == "opu":
        eligible = set(OPU_ELIGIBLE)
        if im2col:
            eligible.add("conv2d")
            eligible.add("depthwise_conv")
        return op_kind in eligible
    return False


# ---- parsing dispatch configs --------------------------------------------
@dataclasses.dataclass
class DispatchInfo:
    ordinal: int
    func_name: str
    op_kind: str
    pipeline_str: str
    has_accel_pipeline: bool


CONFIG_FUNC_RE = re.compile(
    r"func\.func\s+@(?P<name>[A-Za-z0-9_$]+).*?translation_info\s*=\s*"
    r'#iree_codegen\.translation_info<\s*pipeline\s*=\s*[^"]*"(?P<pipe>[^"]+)"',
    re.DOTALL,
)


def parse_configs(cell_dir: Path, hw: str) -> list[DispatchInfo]:
    configs_dir = cell_dir / "configs"
    out: list[DispatchInfo] = []
    if not configs_dir.is_dir():
        return out
    accel_keys = ACCEL_PIPELINE_MARKERS.get(hw, [])
    body_keys = ACCEL_BODY_MARKERS.get(hw, [])
    for f in sorted(configs_dir.glob("configured_*.mlir")):
        text = f.read_text(errors="replace")
        body_hit = any(k in text for k in body_keys) if body_keys else False
        for m in CONFIG_FUNC_RE.finditer(text):
            name = m.group("name")
            pipe = m.group("pipe")
            # Pull ordinal from filename: configured_module_*_async_dispatch_<N>.mlir
            mord = re.search(r"async_dispatch_(\d+)", f.name)
            ord_i = int(mord.group(1)) if mord else len(out)
            pipe_hit = any(k in pipe for k in accel_keys) if accel_keys else False
            out.append(
                DispatchInfo(
                    ordinal=ord_i,
                    func_name=name,
                    op_kind=classify_op_kind(name),
                    pipeline_str=pipe,
                    has_accel_pipeline=pipe_hit or body_hit,
                )
            )
    return out


# ---- ELF opcode histogram ------------------------------------------------
RVV_INSN_PATTERNS = [
    re.compile(r"\bvsetv"),
    re.compile(r"\bvle\d"),
    re.compile(r"\bvse\d"),
    re.compile(r"\bvadd"),
    re.compile(r"\bvsub"),
    re.compile(r"\bvmul"),
    re.compile(r"\bvmacc"),
    re.compile(r"\bvfm[a-z]"),
    re.compile(r"\bvredsum"),
    re.compile(r"\bvslide"),
    re.compile(r"\bvfwm[a-z]"),
]


def looks_rvv(disasm_line: str) -> bool:
    return any(p.search(disasm_line) for p in RVV_INSN_PATTERNS)


# OPU custom OP-V (opcode 0x57) funct6 slots observed post-ucb-bar/main
# switch on the linked .s files. These are vopacc / opmvinbcast / opmvin /
# opmvout variants emitted by `iree_uk_opu_matmul` bitcode. llvm-objdump
# without the xopu disassembler tables prints them as `<unknown>` so we
# must decode the raw 32-bit word ourselves.
#
# Encoding layout (RISC-V V-extension OP-V):
#   bits[6:0]   = 0x57 (OP-V)
#   bits[14:12] = funct3 (OPIVV=0b000, OPMVV=0b010, OPMVX=0b110, ...)
#   bits[31:26] = funct6
# OPU lives in custom funct6 slots 0x28, 0x2C, 0x2E (and adjacents). The
# stock RVV funct6 range is 0x00..0x27 and 0x30..0x3F for arithmetic;
# 0x28..0x2F is the custom space upstream sets aside for vendor exts.
OPU_OPCODE = 0x57
OPU_FUNCT6_RANGE = range(0x28, 0x30)


def looks_opu_raw(raw_word: int, disasm_line: str) -> bool:
    """Return True for OPU custom OP-V instructions.

    OPU shares funct6 ranges with standard RVV (vsrl.vi has funct6=0x28,
    vnsrl.wi has 0x2C, vmacc.vv has 0x2D, etc.) — so we cannot identify
    OPU by funct6 alone. The reliable test is: opcode == 0x57 (OP-V)
    AND the disassembler failed to decode it as a standard RVV op.

    `riscv64-unknown-elf-objdump` emits `<unknown>` for any opcode word
    that has no entry in its decode table. Standard RVV (V 1.0) is
    covered; OPU's custom funct6/funct7 slots fall through to <unknown>.
    Custom OPU mnemonic strings (if a future objdump learns them) are
    also accepted.
    """
    ll = disasm_line.lower()
    if "opmvin" in ll or "vopacc" in ll or ll.startswith("opu_") or " opu_" in ll:
        return True
    if (raw_word & 0x7F) == OPU_OPCODE:
        # GNU binutils objdump emits ".insn 4, 0x..." for opcodes with no
        # decode table entry; llvm-objdump emits "<unknown>". Either is
        # a positive signal in OP-V opcode space.
        if "<unknown>" in ll or ll.lstrip().startswith(".insn"):
            return True
    return False


def histogram_elf(objdump: Path, elf_path: Path) -> dict:
    """Run objdump and bucket each instruction. Returns a histogram."""
    proc = subprocess.run(
        [str(objdump), "-d", "-M", "no-aliases", str(elf_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    hist = {
        "total_insns": 0,
        "scalar": 0,
        "rvv": 0,
        "rocc_custom3": 0,  # Gemmini: opcode 0x7B
        "rocc_funct7": Counter(),  # Gemmini funct7 breakdown
        "opu_marked": 0,  # OPU: opcode 0x57 with custom funct6
        "opu_funct6": Counter(),  # OPU funct6 breakdown
    }
    insn_re = re.compile(r"^\s+([0-9a-f]+):\s+([0-9a-f]+)\s+(.+)$")
    for line in proc.stdout.splitlines():
        m = insn_re.match(line)
        if not m:
            continue
        word_hex = m.group(2)
        mnem_field = m.group(3)
        hist["total_insns"] += 1

        # Try to interpret as a 32-bit RV insn. Compressed insns are 4 hex
        # chars; full insns are 8 hex chars.
        if len(word_hex) == 8:
            try:
                w = int(word_hex, 16)
                opcode = w & 0x7F
                if opcode == 0x7B:  # Gemmini RoCC custom-3
                    hist["rocc_custom3"] += 1
                    f7 = (w >> 25) & 0x7F
                    hist["rocc_funct7"][f7] += 1
                    continue
                if looks_opu_raw(w, mnem_field):
                    # OPU custom OP-V (opcode 0x57 + custom funct6) —
                    # disjoint from Gemmini custom-3.
                    hist["opu_marked"] += 1
                    f6 = (w >> 26) & 0x3F
                    hist["opu_funct6"][f6] += 1
                    continue
            except ValueError:
                pass

        if looks_rvv(mnem_field):
            hist["rvv"] += 1
        else:
            hist["scalar"] += 1
    # Counters are not JSON-serializable as-is.
    hist["rocc_funct7"] = dict(hist["rocc_funct7"])
    hist["opu_funct6"] = dict(hist["opu_funct6"])
    return hist


# ---- per-cell audit ------------------------------------------------------
def audit_cell(cell_dir: Path) -> dict:
    name = cell_dir.name
    # Parse name → model, hw, im2col flag.
    parts = name.split("_")
    im2col = parts[-1] == "im2col"
    if im2col:
        parts = parts[:-1]
    # hw can be one of {scalar, rvv, opu, gemmini, gemmini_mx}
    if parts[-2:] == ["gemmini", "mx"]:
        hw = "gemmini_mx"
        model = "_".join(parts[:-2])
    else:
        hw = parts[-1]
        model = "_".join(parts[:-1])

    dispatches = parse_configs(cell_dir, hw)

    bin_dir = cell_dir / "binaries"
    elf_paths = sorted(bin_dir.glob("*_embedded_elf_riscv_64.so")) if bin_dir.is_dir() else []
    histogram = {}
    if elf_paths:
        objdump = resolve_objdump()
        # Most cells have one linked .so; if multiple, hist them all
        # separately keyed by basename.
        for e in elf_paths:
            histogram[e.name] = histogram_elf(objdump, e)

    # Roll up: total accel opcodes across all ELFs in this cell.
    total_rocc = sum(h["rocc_custom3"] for h in histogram.values())
    total_rvv = sum(h["rvv"] for h in histogram.values())
    total_opu = sum(h.get("opu_marked", 0) for h in histogram.values())
    total_scalar = sum(h["scalar"] for h in histogram.values())

    classifications = []
    cov_counters = Counter()
    for d in dispatches:
        eligible = is_eligible(d.op_kind, hw, im2col)
        # The "did we emit" check is at cell level (we can't easily
        # attribute opcodes to a specific dispatch without per-dispatch
        # ELFs from breakdown_vmfb.py). For accuracy at the dispatch
        # level run breakdown_vmfb after compile_audit.
        if hw == "scalar":
            cat = "scalar_baseline"
        elif hw == "rvv":
            cat = "rvv_lane_compiled"
        elif d.has_accel_pipeline and total_rocc > 0 and hw in ("gemmini", "gemmini_mx"):
            cat = "accel_routed_and_emitted"
        elif d.has_accel_pipeline and hw in ("opu",) and total_opu > 0:
            cat = "accel_routed_and_emitted"
        elif d.has_accel_pipeline:
            cat = "accel_routed_but_fell_back"
        elif eligible:
            cat = "not_routed_eligible_op"
        else:
            cat = "not_routed_ineligible_op"
        cov_counters[cat] += 1
        classifications.append(
            {
                "ordinal": d.ordinal,
                "func": d.func_name,
                "op_kind": d.op_kind,
                "category": cat,
                "has_accel_pipeline": d.has_accel_pipeline,
            }
        )

    summary = {
        "cell": name,
        "model": model,
        "hw": hw,
        "im2col": im2col,
        "n_dispatches": len(dispatches),
        "n_matmul_or_conv": sum(1 for d in dispatches if d.op_kind in ("matmul", "conv2d", "depthwise_conv")),
        "categories": dict(cov_counters),
        "elf_histograms": histogram,
        "totals": {
            "rocc_custom3": total_rocc,
            "rvv": total_rvv,
            "opu_marked": total_opu,
            "scalar": total_scalar,
        },
        "dispatches": classifications,
    }
    return summary


def render_markdown_summary(rows: list[dict]) -> str:
    """One-line per cell, sorted by (model, hw)."""
    rows_sorted = sorted(rows, key=lambda r: (r["model"], r["hw"], r["im2col"]))
    out = [
        "# FireSim Shuttle compile audit",
        "",
        "| Model | HW | im2col | dispatches | matmul/conv | routed | rocc-c3 | rvv | opu-funct6 | scalar |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_sorted:
        cats = r["categories"]
        routed = cats.get("accel_routed_and_emitted", 0) + cats.get("accel_routed_but_fell_back", 0)
        out.append(
            f"| {r['model']} | {r['hw']} | "
            f"{'yes' if r['im2col'] else 'no'} | "
            f"{r['n_dispatches']} | {r['n_matmul_or_conv']} | "
            f"{routed} | "
            f"{r['totals']['rocc_custom3']} | "
            f"{r['totals']['rvv']} | "
            f"{r['totals']['opu_marked']} | "
            f"{r['totals']['scalar']} |"
        )
    out.append("")
    out.append("Category legend:")
    out.append("- `accel_routed_and_emitted` — translation_info routed AND ELF contains expected accel opcodes")
    out.append("- `accel_routed_but_fell_back` — translation_info routed but ELF has NO accel opcodes (bug)")
    out.append("- `not_routed_eligible_op` — matmul/conv that the plugin did NOT route (coverage gap)")
    out.append("- `not_routed_ineligible_op` — op kind correctly skipped (softmax, reshape, ...)")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        required=True,
        help="build/compile_audit root directory",
    )
    ap.add_argument(
        "--out-md",
        default="/scratch2/agustin/merlin/tmp/firesim_shuttle_compile_audit.md",
    )
    ap.add_argument(
        "--out-json",
        default="/scratch2/agustin/merlin/tmp/firesim_shuttle_compile_audit.json",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"[audit] root not found: {root}")

    cells = sorted([d for d in root.iterdir() if d.is_dir()])
    if not cells:
        sys.exit(f"[audit] no cells under {root}")

    rows: list[dict] = []
    for c in cells:
        try:
            s = audit_cell(c)
            (c / "audit").mkdir(exist_ok=True)
            (c / "audit" / "coverage.json").write_text(json.dumps(s, indent=2, sort_keys=True) + "\n")
            rows.append(s)
            print(
                f"[audit] {c.name}: {s['n_dispatches']} dispatches, "
                f"matmul/conv={s['n_matmul_or_conv']}, "
                f"rocc={s['totals']['rocc_custom3']}, "
                f"opu={s['totals']['opu_marked']}, "
                f"rvv={s['totals']['rvv']}"
            )
        except Exception as e:  # noqa: BLE001
            print(f"[audit] {c.name}: ERROR {e}")
            continue

    Path(args.out_json).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    Path(args.out_md).write_text(render_markdown_summary(rows))
    print(f"\n[audit] wrote {args.out_md}")
    print(f"[audit] wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
