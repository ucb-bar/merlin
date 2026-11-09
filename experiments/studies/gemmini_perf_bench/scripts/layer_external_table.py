#!/usr/bin/env python3
"""Measure a THIRD-PARTY kernel beside this target's library kernel, on FPGA, per layer.

The published comparison between this target's stock library and the two autoscheduled/expert
ResNet-50 variants is a WHOLE-MODEL number taken on another platform. It cannot be reproduced here:
the external bundle ships kernels only, and the three data headers its program includes
(`resnet50_params_spec.h`, `images_spec.h`, `resnet50_gold_logits.h`) are not in it, while its golden
argmax belongs to an image set we do not have. What IS reproducible is the kernel comparison, and
that is the one the whole-model number cannot answer anyway: whether OUR kernel stands up.

So each arm runs the same packed operand blob, writes the same output buffer, prints the same
`LB_RECORD <label> cycles= digest=` line, and is admitted only when its digest equals the digest of
the contract's expected output. Only the call differs.

Three properties this bench exists to keep, each of them a trap the external bundle documents:

* **Isolated flatters embedded.** The bundle's own log records the same optimized kernel at 2.61M
  cycles in isolation against 5.56M embedded in the model, the difference being a per-invocation
  `memset` of an 861,632-byte host padding buffer. Both arms therefore run warm-then-measured, and
  the measured window includes every host transform the kernel needs to compute the function from
  the operands it was handed -- the external kernel's input zero-padding included. A bench that
  timed only the accelerator call would report a kernel that does not exist.
* **A specialised kernel can silently become the library.** The external kernels guard their fixed
  scratchpad footprint with a compile-time `..._FITS` test and fall back to the library's auto-tiled
  path when it fails, so on a smaller design "their kernel" IS the library under another name. The
  footprint is asserted against this design's own capacity before anything is built.
* **A shape it did not declare is not that kernel.** Enforced by the renderer
  (`render_external_layer`), which refuses a spec that differs from the descriptor's declared shape.

Shapes and liveness come from the external bundle's own descriptor table, never from literals here:
a kernel its table marks as never called is not measured, because measuring dead code would report a
speedup the model never receives.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_root = os.environ.get("MERLIN_REPO_ROOT", "").strip()
REPO = Path(_root).expanduser().resolve() if _root else _HERE.parents[4]
sys.path.insert(0, str(REPO / "merlin/python"))

#: Columns the descriptor table must carry for a row to be measurable.
_REQUIRED_COLUMNS = ("kernel", "op", "filter", "in_ch", "out_ch", "stride", "spatial", "call_sites", "status")
#: The status a row uses to say its kernel is never called. Anything else is measured.
_DEAD = "dead_code_never_called"


class ExternalBenchError(RuntimeError):
    pass


def read_descriptor(path: Path) -> list[dict]:
    """The external bundle's own kernel table. Refuses a table missing a column it needs."""
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ExternalBenchError(f"{path} declares no kernels")
    missing = [c for c in _REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise ExternalBenchError(f"{path} is missing column(s) {missing}")
    return rows


def spec_from_row(row: dict, *, batch: int, scale: float, relu: bool) -> tuple[dict, dict]:
    """``(layer spec, declared shape)`` for one descriptor row.

    The declared shape is what the renderer holds the spec to. Padding is derived from the filter so
    the output keeps the input's spatial extent, which is what a `stride 1` ResNet block does; a row
    whose geometry does not admit that is refused rather than reshaped.
    """
    if row["op"] != "conv2d":
        raise ExternalBenchError(f"{row['kernel']}: no external renderer for op {row['op']!r}")
    fh, _, fw = row["filter"].partition("x")
    sh, _, sw = row["spatial"].partition("x")
    if fh != fw or sh != sw:
        raise ExternalBenchError(f"{row['kernel']}: non-square filter/spatial {row['filter']}/{row['spatial']}")
    k, n, stride = int(fh), int(sh), int(row["stride"])
    if k % 2 != 1:
        raise ExternalBenchError(f"{row['kernel']}: even filter {k} has no symmetric same-padding")
    declared = {
        "kernel": k,
        "in_channels": int(row["in_ch"]),
        "out_channels": int(row["out_ch"]),
        "stride": stride,
        "in_dim": n,
    }
    spec = {
        "op": "conv2d",
        "label": "X" + row["kernel"][:20],
        "batch": batch,
        "padding": k // 2,
        "scale": scale,
        "relu": relu,
        "seed": 1,
        "protocol": "warm_then_measured",
        **declared,
    }
    return spec, declared


def assert_footprint_fits(row: dict, *, spad_rows: int) -> int:
    """Refuse a kernel whose declared scratchpad footprint this design cannot hold.

    The external kernels compile to the library's auto-tiled path when their fixed footprint does not
    fit, so a run on a too-small design would measure the library and report it under their name.
    """
    need = (row.get("spad_rows_needed") or "").strip()
    if not need:
        raise ExternalBenchError(f"{row['kernel']}: descriptor states no scratchpad footprint to check")
    if int(need) > spad_rows:
        raise ExternalBenchError(
            f"{row['kernel']}: needs {need} scratchpad rows, this design holds {spad_rows}; the kernel "
            "would compile to the library's fallback path and be measured under its own name"
        )
    return int(need)


def design_spad_rows(target: str) -> int:
    """Rows of operand scratchpad this target declares, from its own derived address space.

    Derived rather than read off a header path, so a target whose capacity comes from somewhere else
    answers the same question. A capacity this cannot resolve is a refusal, never a default: the whole
    point of the check is that a too-small design silently turns the external kernel into the library.
    """
    from merlin.compile.capacity import _operand_store_capacity_elems
    from merlin.compile.mesh import _mesh_tile_binding

    binding = _mesh_tile_binding(target, "i8", "i32")
    dim = int(getattr(binding, "tile_dim", 0) or 0)
    elems = _operand_store_capacity_elems(target, binding.operand_dtype)
    if not dim or not elems:
        raise ExternalBenchError(
            f"{target} declares no operand-store capacity this can resolve; the external kernel's fixed "
            "footprint cannot be checked, and an unchecked footprint may compile to the library fallback"
        )
    return int(elems) // dim


def build_arm(spec, declared, *, arm, target, workroot, kernel_header, symbol, support_headers=()):
    """Build one ELF for one arm. Returns the payload row (no execution)."""
    from merlin.perf.layer_bench import build_program
    from merlin.perf.layer_bench.reference import pack_operands
    from merlin.runtime.backends import base
    from merlin.sched.contract import contract as _contract

    backend = base.get_backend(target)
    # The same numerics contract the library arm is measured under, built from this target's own
    # readout facts -- so both arms are admitted against one expected output, not two.
    contract = _contract("per_tensor_readout_v1", backend.readout_facts())
    blob, offsets = pack_operands(spec, accumulator_dtype=contract.accumulator_dtype)
    if arm == "library":
        source = backend.render_library_layer(spec, offsets=offsets)
    elif arm == "external":
        source = backend.render_external_layer(
            spec, offsets=offsets, symbol=symbol, header=Path(kernel_header).name, declared_shape=declared
        )
    else:
        raise ExternalBenchError(f"unknown arm {arm!r}")
    wd = Path(workroot) / f"{spec['label']}_{arm}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / backend.LIBRARY_LAYER_OPERAND_BLOB).write_bytes(blob)
    (wd / "layer.c").write_text(source, encoding="utf-8")
    if arm == "external":
        (wd / Path(kernel_header).name).write_bytes(Path(kernel_header).read_bytes())
        # The kernel header may include siblings from its own bundle (a macro shim bridging the short
        # primitive names its search used to this target's real gemmini_extended_* macros). They are
        # staged under include/ because that is the prefix the header spells, and they are the
        # BUNDLE's copies: substituting ours would change what is being measured.
        for extra in support_headers:
            dest = wd / "include" / Path(extra).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(extra).read_bytes())
    built = build_program([wd / "layer.c"], wd, target=target, max_loaded_bytes=None)
    return {
        "arm": arm,
        "label": spec["label"],
        "kernel_symbol": symbol if arm == "external" else "tiled_conv_auto",
        "spec": spec,
        "workdir": str(wd),
        "elf": str(built.elf),
        "elf_sha256": built.elf_sha256,
        "loaded_bytes": built.loaded_bytes,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--descriptor", required=True, type=Path, help="the external bundle's kernels.csv")
    ap.add_argument("--kernel-header", required=True, type=Path, help="the external bundle's kernel header")
    ap.add_argument("--symbol-column", default="kernel", help="descriptor column holding the C symbol")
    ap.add_argument(
        "--support-header",
        type=Path,
        action="append",
        default=[],
        help="a sibling header the kernel header includes, staged under include/",
    )
    ap.add_argument("--target", default="gemmini")
    ap.add_argument("--workroot", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--scale", type=float, default=0.03125, help="requant multiplier, identical in both arms")
    ap.add_argument("--no-relu", action="store_true")
    ap.add_argument("--out", type=Path, help="write the build table here")
    args = ap.parse_args(argv)

    spad_rows = design_spad_rows(args.target)
    rows, built, skipped = read_descriptor(args.descriptor), [], []
    for row in rows:
        if row["status"].strip() == _DEAD:
            skipped.append({"kernel": row["kernel"], "why": _DEAD, "call_sites": row["call_sites"]})
            continue
        spec, declared = spec_from_row(row, batch=args.batch, scale=args.scale, relu=not args.no_relu)
        footprint = assert_footprint_fits(row, spad_rows=spad_rows)
        for arm in ("library", "external"):
            payload = build_arm(
                spec,
                declared,
                arm=arm,
                target=args.target,
                workroot=args.workroot,
                kernel_header=args.kernel_header,
                symbol=row[args.symbol_column],
                support_headers=args.support_header,
            )
            payload["spad_rows_needed"] = footprint
            payload["call_sites"] = int(row["call_sites"])
            built.append(payload)

    table = {
        "schema": "external_kernel_layer_build_v1",
        "target": args.target,
        "design_spad_rows": spad_rows,
        "descriptor": str(args.descriptor),
        "kernel_header_sha256": __import__("hashlib").sha256(args.kernel_header.read_bytes()).hexdigest(),
        "built": built,
        "skipped": skipped,
    }
    text = json.dumps(table, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
