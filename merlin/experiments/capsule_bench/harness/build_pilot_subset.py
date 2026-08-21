#!/usr/bin/env python3
"""Build the capsule_bench_v0 PILOT capsule subset (operator-side, full goldens).

The pilot validates the harness + raw_baseline workflow on a reduced set before the full
A0-A7/B0-B4/C0-C6 suite. We assemble a curated capsules_root containing ONLY:

  public/dev:  A0_config_smoke, A2_single_tile_matmul, A4_acc_scale_i8, B0_quantized_linear_i8
  holdouts:    the first N_HOLDOUTS of the held-out set, DERIVED by listing it (post-freeze phase)

The holdouts are derived rather than named: this file is tracked in a public repo, and a hardcoded
list of held-out capsule names publishes the very set the hidden phase is meant to hold back.

Each pilot entry is a REAL directory whose files are symlinks into the frozen
merlin/contract/capsules tree (so `capsule_runner.discover_capsules`, which rglobs for
`capsule.yaml`, finds them — pathlib does not necessarily descend into symlinked dirs, so we
symlink files, not dirs). This root is used ONLY by the operator-side grader (qa_check / the
hidden phase); it includes golden.yaml. The AGENT never sees this root — its workspace is
golden-masked separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import _common as C

CAPS = C.REPO / "merlin/contract" / "capsules"
OUT = C.EXP / "scripts" / "pilot_capsules"

# (subset_name, source_capsule_dir) — the public half is named because it IS public.
N_HOLDOUTS = 3
PUBLIC = [
    ("A0_config_smoke",        CAPS / "isa" / "A0_config_smoke"),
    ("A2_single_tile_matmul",  CAPS / "isa" / "A2_single_tile_matmul"),
    ("A4_acc_scale_i8",        CAPS / "isa" / "A4_acc_scale_i8"),
    ("B0_quantized_linear_i8", CAPS / "layers" / "B0_quantized_linear_i8"),
]


def holdout_subset(n: int = N_HOLDOUTS) -> list[tuple[str, "Path"]]:
    """The first ``n`` holdouts of the shared corpus, listed off disk in a stable order.

    Derived, never named. The operator-side grader running this script can see the holdout store;
    this tracked file must not carry its contents. An empty holdout store yields an empty list --
    a public clone builds the public half of the pilot and says so, rather than failing."""
    store = CAPS / "hidden"
    if not store.is_dir():
        return []
    dirs = sorted(d for d in store.iterdir() if (d / "capsule.yaml").is_file())
    return [(d.name, d) for d in dirs[:n]]


PILOT = PUBLIC + holdout_subset()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    for name, src in PILOT:
        if not (src / "capsule.yaml").exists():
            print(f"MISSING source capsule: {src}", file=sys.stderr)
            return 2
        dst = OUT / name
        dst.mkdir(exist_ok=True)
        for f in sorted(src.iterdir()):
            if not f.is_file():
                continue
            link = dst / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(f.resolve())
        made.append(name)
    print(f"pilot subset @ {OUT}")
    for n in made:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
