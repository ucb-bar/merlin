#!/usr/bin/env python
"""Board A/B for `vectorize_amax_reduction`: build twice, run on the K1, compare BY DIGEST.

WHY THIS SCRIPT EXISTS RATHER THAN A NUMBER IN A REPORT. The feature is EXACT by construction (see
`llvmlower/reduce_vec`), so its correctness claim is bit-identity of the model OUTPUT, not a cosine.
Bit-identity is only worth asserting if it survives a changed process image: a lever that reads
uninitialized memory, or whose emitted code depends on a stack or heap offset, prints the same
number in a quiet loop and a different one under a real workload. So every configuration is run
under several ENVIRONMENT PADDINGS -- a variable of increasing length, which shifts the initial
stack pointer and therefore every stack-derived address in the process -- and the digest has to be
the same across all of them AND equal between the two builds.

WALL TIME IS REPORTED AND IS NOT A RESULT. On this repo, levers that removed instructions and shrank
the object have measured 1.09x and 1.28x SLOWER, and a five-lever stack regressed 1.815 -> 1.943. The
`--iters`/`--n` sampling here exists so the number is a MEASUREMENT rather than a single sample; it
still has to clear the board's own noise floor (>= 1.9%, band 2.6%) before it means anything, and
this script prints that comparison rather than a speedup verdict.

USAGE (the board is the only thing here that is not host-side):

    MERLIN_K1_HOST=root@<board-ip> PYTHONPATH=merlin/python .venv/bin/python \\
        build_tools/scripts/k1_amax_reduction_ab.py \\
        --bundle out/artifacts/recaptures/lstmnetvit_int8_w8a8_consistent \\
        --package out/artifacts/targets/rvv/hand_v0_int8 \\
        --paddings 6 --n 5

Add `--build-only` to do everything except touch the board: it still builds both ELFs and prints the
linked-ELF census, which is the whole static half of the evidence and needs no hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from merlin.common.artifacts import new_product
from merlin.common.paths import repo_root
from merlin.llvmlower import toolchain
from merlin.llvmlower.reduce_vec import FEATURE
from merlin.mining import k1
from merlin.mining.registry import load_rvv_package

#: libm symbols the quantization path can reach. `fabsf` is the one this feature removes; the others
#: are listed so the report shows whether removing it moved anything else (it should not).
LIBM_SYMBOLS = ("fabsf", "fabs", "roundevenf", "expf", "erff", "tanhf",
                "sqrtf", "logf", "powf", "truncf", "floorf", "ceilf")


def _objdump(elf: Path) -> str:
    """Disassemble the LINKED ELF. Never an unrelocated `.o`: there `forward` is split at the
    `.Lpcrel_hi*` labels the assembler emits for every symbol reference, so a span-based metric
    reads a tiny prefix of the function and silently reports a fraction of the truth."""
    tool = toolchain.clang().parent / "llvm-objdump"
    if not tool.is_file():
        raise SystemExit(f"llvm-objdump not found next to {toolchain.clang()}")
    return subprocess.run([str(tool), "-d", str(elf)],
                          capture_output=True, text=True, check=True).stdout


def elf_census(elf: Path) -> dict:
    """Per-symbol vector/scalar instruction split and libm call sites, from the linked ELF.

    The vector test is the RISC-V vector mnemonic prefix, read off the disassembly's own mnemonic
    column -- structural, so an instruction this repo has never seen is still classified.
    """
    per: dict[str, dict] = {}
    calls: dict[str, int] = {}
    cur = None
    for line in _objdump(elf).splitlines():
        s = line.strip()
        if s.endswith(":") and "<" in s and ">" in s:
            cur = s[s.find("<") + 1:s.rfind(">")]
            per.setdefault(cur, {"vector": 0, "scalar": 0, "total": 0})
            continue
        if cur is None:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        mnemonic = parts[2].strip().split(" ")[0]
        if not mnemonic:
            continue
        d = per[cur]
        d["total"] += 1
        d["vector" if mnemonic.startswith("v") else "scalar"] += 1
        for sym in LIBM_SYMBOLS:
            if f"<{sym}>" in line:
                calls[sym] = calls.get(sym, 0) + 1
    fwd = per.get("forward", {"vector": 0, "scalar": 0, "total": 0})
    frac = (fwd["vector"] / fwd["total"]) if fwd["total"] else None
    return {"elf_bytes": elf.stat().st_size, "symbols": len(per),
            "forward": fwd, "forward_vector_fraction": frac, "libm_call_sites": calls}


def build(bundle: Path, pkg, features: list[str], work: Path) -> Path:
    """Cross-compile + LINK, host-side. `build_k1_binary` never contacts the board.

    `fallback_policy="forbid"` on purpose: the default silently falls back to a SCALAR whole-model
    build on a PipelineError, and a scalar arm compared against a vectorized one measures the
    fallback, not the feature. A build that cannot lower must fail loudly here.
    """
    work.mkdir(parents=True, exist_ok=True)
    return k1.build_k1_binary(bundle, work, replace(pkg, compiler_features=sorted(features)),
                              fallback_policy="forbid")


def run_paddings(bundle: Path, bwork: Path, pkg, elf: Path, n_paddings: int,
                 iters: int, timeout: int) -> list[dict]:
    """Run the ALREADY-BUILT ELF on the board once per environment padding; a row per padding.

    Through `k1.run_binary_on_k1`, which deploys and runs a given binary under an explicit
    environment and takes the board lock around deploy+run only. Building once and running many
    times is the point: rebuilding per padding would put a different object under each measurement
    and make the comparison unattributable.

    The padding is a single environment variable whose length doubles each step. It changes nothing
    the program reads -- only the size of the environment block the kernel copies above the initial
    stack pointer, and therefore the alignment and absolute address of every stack object. A digest
    that moves across these has an address dependence, which for a rewrite claimed EXACT is a defect
    no cosine gate would catch.
    """
    rows = []
    for i in range(n_paddings):
        pad = "X" * (1 << (6 + i))                 # 64 B, 128 B, ... doubling per padding
        env = {"MERLIN_AB_PAD": pad, "MERLIN_ITERS": str(iters)}
        try:
            res = k1.run_binary_on_k1(bundle, bwork, pkg, elf, env=env, timeout=timeout)
        except Exception as e:                                       # noqa: BLE001
            rows.append({"padding_bytes": len(pad), "error": f"{type(e).__name__}: {e}"})
            continue
        # The harness prints its own digest over the output bytes; recompute host-side over the
        # parsed values as an independent check, and REPORT BOTH. A single digest that the same code
        # both produces and checks cannot detect a harness-side bug.
        outputs = res.get("outputs")
        host = (hashlib.sha256(
            b"".join(float(v).hex().encode() for v in outputs)).hexdigest()
            if outputs else None)
        walls = res.get("iter_wall_ns") or []
        rows.append({"padding_bytes": len(pad),
                     "board_out_hash": res.get("out_hash"),
                     "host_digest_over_parsed_outputs": host,
                     "n_outputs": len(outputs) if outputs else 0,
                     "wall_ns_min": min(walls) if walls else None})
    return rows


def verdict(base_rows: list[dict], feat_rows: list[dict]) -> dict:
    """Bit-identity across paddings AND between arms, or the exact reason it could not be decided."""
    def digests(rows):
        return [r.get("board_out_hash") for r in rows]
    b, f = digests(base_rows), digests(feat_rows)
    if any(d is None for d in b + f) or not b or not f:
        return {"decided": False,
                "reason": "at least one run produced no output digest; a missing digest is refused "
                          "as firmly as a mismatch -- it cannot support a bit-identity claim"}
    return {"decided": True,
            "baseline_stable_across_paddings": len(set(b)) == 1,
            "feature_stable_across_paddings": len(set(f)) == 1,
            "arms_bit_identical": set(b) == set(f) and len(set(b)) == 1,
            "baseline_digests": sorted(set(b)), "feature_digests": sorted(set(f))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--paddings", type=int, default=6,
                    help="environment paddings per arm (>= 6 is the standard here)")
    ap.add_argument("--n", type=int, default=5, help="repeats per padding (min wall is taken)")
    ap.add_argument("--build-only", action="store_true",
                    help="build both ELFs and print the linked-ELF census; never touch the board")
    ap.add_argument("--timeout", type=int, default=900, help="per-run board timeout (s)")
    ap.add_argument("--work", type=Path, default=None)
    a = ap.parse_args(argv)

    work = a.work or Path(tempfile.mkdtemp(prefix="amax_ab_"))
    pkg = load_rvv_package(a.package)
    base_features = sorted(frozenset(pkg.compiler_features or []))
    arms = {"baseline": base_features,
            "with_amax_reduction": sorted(set(base_features) | {FEATURE})}

    report: dict = {"bundle": str(a.bundle), "package": str(a.package),
                    "baseline_features": base_features, "feature": FEATURE,
                    "paddings": a.paddings, "n": a.n, "work": str(work), "arms": {}}

    elves: dict[str, Path] = {}
    for tag, feats in arms.items():
        try:
            elf = build(a.bundle, pkg, feats, work / tag)
        except Exception as e:                                       # noqa: BLE001
            report["arms"][tag] = {"features": feats, "build_error": f"{type(e).__name__}: {e}"}
            continue
        elves[tag] = elf
        report["arms"][tag] = {"features": feats, "elf": str(elf), "static": elf_census(elf)}

    if len(elves) == 2:
        b, f = (report["arms"]["baseline"]["static"],
                report["arms"]["with_amax_reduction"]["static"])
        report["static_delta"] = {
            "forward_total": f["forward"]["total"] - b["forward"]["total"],
            "forward_vector": f["forward"]["vector"] - b["forward"]["vector"],
            "forward_scalar": f["forward"]["scalar"] - b["forward"]["scalar"],
            "forward_vector_fraction": [b["forward_vector_fraction"],
                                        f["forward_vector_fraction"]],
            "libm_call_sites": {"baseline": b["libm_call_sites"],
                                "with_feature": f["libm_call_sites"]},
            "elf_bytes": [b["elf_bytes"], f["elf_bytes"]]}

    if not a.build_only and len(elves) == 2:
        rows = {tag: run_paddings(a.bundle, work / tag, pkg, elf, a.paddings, a.n, a.timeout)
                for tag, elf in elves.items()}
        for tag, r in rows.items():
            report["arms"][tag]["runs"] = r
        report["correctness"] = verdict(rows["baseline"], rows["with_amax_reduction"])
        walls = {tag: [x.get("wall_ns_min") for x in r if x.get("wall_ns_min")]
                 for tag, r in rows.items()}
        if all(walls.values()):
            lo, hi = min(walls["baseline"]), min(walls["with_amax_reduction"])
            report["wall"] = {
                "baseline_ns_min": lo, "feature_ns_min": hi,
                "ratio_baseline_over_feature": lo / hi if hi else None,
                "board_noise_floor_pct": 1.9,
                "note": "NOT a speed claim. A ratio inside the board's own noise floor is not a "
                        "result; see memory k1-measurement-noise-floor."}
    elif not a.build_only:
        report["correctness"] = {"decided": False,
                                 "reason": "both arms did not build; nothing was run"}

    product = new_product("amax-reduction-ab", target="k1_spacemit", version=1)
    out = Path(product) / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
