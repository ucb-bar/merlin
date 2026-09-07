#!/usr/bin/env python
"""Board A/B for HONOURING the package's declared cflags on the K1 model object.

WHAT THIS MEASURES, AND WHY IT RECALIBRATES OTHER RESULTS. `out/artifacts/targets/rvv/hand_v0_int8`
declares ``cflags: [-march=rv64gcv, -fno-vectorize, -fno-slp-vectorize]`` and
`mining.registry.RvvPackage` documents them as feeding the model-object build. `build_k1_binary`
never read them. So on this path clang's OWN loop and SLP vectorizers have been running on our
emitted ``model.ll``, and every static claim of the form "our transform schedule vectorized this op"
was measuring two vectorizers and attributing the result to one of them. This script builds the two
arms and prints the split: how much of the binary's vector code survives when clang's vectorizers
are turned off, i.e. how much of it is OURS.

THE ARM IS NOT "THE PACKAGE'S CFLAGS VERBATIM". The package spells ``-march=rv64gcv``, which
promises only the RVV MINIMUM VLEN of 128 bits; the K1 path derives ``-march`` from the board
(`codegen_march`, VLEN=256 plus ``zfh``/``zvfh``). Applying the package's spelling would DOUBLE every
register group and drop the f16 extensions while looking like a change of vectorizer flags -- a
confound that would swamp the thing being measured. `k1.merge_package_cflags` therefore drops the
flag classes the build derives for itself and applies the rest, and this script's delta is
attributable to ``-fno-vectorize -fno-slp-vectorize`` alone.

NOT A SPEED CLAIM. Turning clang's vectorizers off may well be SLOWER; that is a legitimate outcome
and is why the wall numbers here are sampled and printed against the board's own noise floor rather
than reduced to a verdict. The deliverable is the STATIC split.

    PYTHONPATH=merlin/python .venv/bin/python build_tools/scripts/k1_pkg_cflags_ab.py \\
        --bundle out/artifacts/recaptures/lstmnetvit_int8_w8a8_consistent \\
        --package out/artifacts/targets/rvv/hand_v0_int8 --build-only

Drop ``--build-only`` and set ``MERLIN_K1_HOST=root@<board-ip>`` to add the board half (output
digests across environment paddings + sampled wall).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from merlin.common.artifacts import new_product
from merlin.mining import k1
from merlin.mining.registry import load_rvv_package
from merlin.llvmlower import toolchain

#: Instruction families the report breaks out by name. Chosen because each answers a different
#: question about WHOSE vectorizer produced the code: the reduce family is the max-reduction idiom
#: clang recognises, the widening-multiply family is the int8 datapath our schedule emits, and the
#: load/store families move the operands. Counted by MNEMONIC PREFIX off the disassembly's own
#: mnemonic column, so a mnemonic this repo has never seen is still attributed to its family.
FAMILIES = ("vfredmax", "vfredusum", "vredmax", "vwmacc", "vmacc", "vmul", "vle", "vse",
            "vsetvli", "vsetivli", "vfmacc", "vsext", "vzext", "vand", "vfabs", "vslide")

#: Matched longest-first so a shorter family cannot shadow a longer one.
_FAMILIES_BY_LEN = tuple(sorted(FAMILIES, key=len, reverse=True))

LIBM_SYMBOLS = ("fabsf", "fabs", "roundevenf", "expf", "erff", "tanhf", "sqrtf", "logf", "powf")


def _objdump(elf: Path) -> str:
    """Disassemble the LINKED ELF -- never an unrelocated ``.o``, where `forward` is split at the
    ``.Lpcrel_hi*`` labels the assembler emits for every symbol reference, so a span-based metric
    reads a tiny prefix of the function and silently reports a fraction of the truth."""
    tool = toolchain.clang().parent / "llvm-objdump"
    if not tool.is_file():
        raise SystemExit(f"llvm-objdump not found next to {toolchain.clang()}")
    import subprocess
    return subprocess.run([str(tool), "-d", str(elf)],
                          capture_output=True, text=True, check=True).stdout


def elf_census(elf: Path) -> dict:
    """Per-symbol vector/scalar split, per-family mnemonic counts and libm call sites."""
    per: dict[str, dict] = {}
    fam_fwd: dict[str, int] = {}
    fam_all: dict[str, int] = {}
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
        # LLVM objdump lays a disassembly line out as
        #     "   11992: 020cf407     \tvle64.v\tv8, (s9)"
        # i.e. tab-field 0 is address+encoding, field 1 is the MNEMONIC and field 2 the operands.
        # Reading field 2 instead yields the operand list, whose first token is a register name --
        # which still "works" for a vector/scalar test (`v8,` starts with a v) and so produces a
        # plausible but WRONG split: every instruction that writes a SCALAR register from vector
        # state (`vsetvli a0, ...`, `vmv.x.s`, `vcpop`) is then counted scalar, and no mnemonic
        # family ever matches, so the family breakdown silently comes back empty. That empty
        # breakdown is the tell. An operand-only line (a bare `ret`) has no field 2 at all.
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        mnemonic = parts[1].strip().split(" ")[0]
        if not mnemonic:
            continue
        d = per[cur]
        d["total"] += 1
        d["vector" if mnemonic.startswith("v") else "scalar"] += 1
        # LONGEST PREFIX FIRST. `vsetvli` and `vsext.vf4` both start with `vse`, so a
        # declaration-order scan attributes them to the store family and the two real families
        # report zero -- a breakdown that looks complete and is wrong. Sorting by length makes the
        # match independent of how the tuple happens to be written.
        for fam in _FAMILIES_BY_LEN:
            if mnemonic.startswith(fam):
                fam_all[fam] = fam_all.get(fam, 0) + 1
                if cur == "forward":
                    fam_fwd[fam] = fam_fwd.get(fam, 0) + 1
                break
        for sym in LIBM_SYMBOLS:
            if f"<{sym}>" in line:
                calls[sym] = calls.get(sym, 0) + 1
    fwd = per.get("forward", {"vector": 0, "scalar": 0, "total": 0})
    frac = (fwd["vector"] / fwd["total"]) if fwd["total"] else None
    return {"elf_bytes": elf.stat().st_size, "symbols": len(per), "forward": fwd,
            "forward_vector_fraction": frac, "forward_families": fam_fwd,
            "whole_elf_families": fam_all, "libm_call_sites": calls}


def build(bundle: Path, pkg, work: Path, honor: bool, max_session_steps: int | None) -> Path:
    """Cross-compile + LINK, host-side; `build_k1_binary` never contacts the board.

    ``fallback_policy="forbid"``: the default silently falls back to a SCALAR whole-model build on a
    PipelineError, and a scalar arm compared against a vectorized one measures the fallback rather
    than the flags.
    """
    work.mkdir(parents=True, exist_ok=True)
    return k1.build_k1_binary(bundle, work, pkg, fallback_policy="forbid",
                              max_session_steps=max_session_steps,
                              honor_pkg_cflags=honor)


def run_paddings(bundle: Path, bwork: Path, pkg, elf: Path, n_paddings: int,
                 iters: int, timeout: int) -> list[dict]:
    """Run the ALREADY-BUILT ELF once per environment padding. The padding changes nothing the
    program reads -- only the size of the environment block above the initial stack pointer, hence
    every stack-derived address. A digest that moves across paddings has an address dependence,
    which no cosine gate would catch."""
    rows = []
    for i in range(n_paddings):
        pad = "X" * (1 << (6 + i))
        env = {"MERLIN_AB_PAD": pad, "MERLIN_ITERS": str(iters)}
        try:
            res = k1.run_binary_on_k1(bundle, bwork, pkg, elf, env=env, timeout=timeout)
        except Exception as e:                                       # noqa: BLE001
            rows.append({"padding_bytes": len(pad), "error": f"{type(e).__name__}: {e}"})
            continue
        outputs = res.get("outputs")
        host = (hashlib.sha256(b"".join(float(v).hex().encode() for v in outputs)).hexdigest()
                if outputs else None)
        walls = res.get("iter_wall_ns") or []
        rows.append({"padding_bytes": len(pad), "board_out_hash": res.get("out_hash"),
                     "host_digest_over_parsed_outputs": host,
                     "n_outputs": len(outputs) if outputs else 0,
                     "wall_ns_min": min(walls) if walls else None})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--paddings", type=int, default=6)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--build-only", action="store_true",
                    help="build both ELFs and print the linked-ELF census; never touch the board")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--work", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    # A 256-step, 154 MB session corpus becomes a 770 MB `model_io.h` costing ~7 GB of RSS to
    # compile. The corpus size does not affect the emitted MODEL code this script censuses, so a
    # static-only comparison caps it; both arms get the SAME cap, so the delta stays attributable.
    ap.add_argument("--max-session-steps", type=int, default=None)
    a = ap.parse_args(argv)

    work = a.work or Path(tempfile.mkdtemp(prefix="pkgcflags_ab_"))
    pkg = load_rvv_package(a.package)
    arms = {"clang_vectorizers_on": False, "package_cflags_honoured": True}

    report: dict = {"bundle": str(a.bundle), "package": str(a.package),
                    "declared_pkg_cflags": list(pkg.cflags),
                    "applied_pkg_cflags": k1.merge_package_cflags([], pkg.cflags),
                    "work": str(work), "arms": {}}

    elves: dict[str, Path] = {}
    for tag, honor in arms.items():
        try:
            elf = build(a.bundle, pkg, work / tag, honor, a.max_session_steps)
        except Exception as e:                                       # noqa: BLE001
            report["arms"][tag] = {"honor_pkg_cflags": honor,
                                   "build_error": f"{type(e).__name__}: {e}"}
            continue
        elves[tag] = elf
        report["arms"][tag] = {"honor_pkg_cflags": honor, "elf": str(elf),
                               "static": elf_census(elf)}

    if len(elves) == 2:
        b = report["arms"]["clang_vectorizers_on"]["static"]
        f = report["arms"]["package_cflags_honoured"]["static"]
        bv, fv = b["forward"]["vector"], f["forward"]["vector"]
        report["static_delta"] = {
            "forward_total": [b["forward"]["total"], f["forward"]["total"]],
            "forward_vector": [bv, fv],
            "forward_scalar": [b["forward"]["scalar"], f["forward"]["scalar"]],
            "forward_vector_fraction": [b["forward_vector_fraction"],
                                        f["forward_vector_fraction"]],
            # THE DELIVERABLE. With clang's vectorizers off, the vector instructions that REMAIN in
            # `forward` are the ones our transform schedule put there; the ones that disappear were
            # clang's. Reported as a share of the baseline's vector population.
            "ours_vector_instructions": fv,
            "clang_vector_instructions": bv - fv,
            "ours_share_of_baseline_vector_pct":
                round(100.0 * fv / bv, 2) if bv else None,
            "clang_share_of_baseline_vector_pct":
                round(100.0 * (bv - fv) / bv, 2) if bv else None,
            "forward_families": {"clang_on": b["forward_families"],
                                 "honoured": f["forward_families"]},
            "libm_call_sites": {"clang_on": b["libm_call_sites"],
                                "honoured": f["libm_call_sites"]},
            "elf_bytes": [b["elf_bytes"], f["elf_bytes"]]}

    if not a.build_only and len(elves) == 2:
        rows = {tag: run_paddings(a.bundle, work / tag, pkg, elf, a.paddings, a.n, a.timeout)
                for tag, elf in elves.items()}
        for tag, r in rows.items():
            report["arms"][tag]["runs"] = r
        def digs(t):
            return [x.get("board_out_hash") for x in rows[t]]
        b_d, f_d = digs("clang_vectorizers_on"), digs("package_cflags_honoured")
        if any(d is None for d in b_d + f_d) or not b_d or not f_d:
            report["correctness"] = {"decided": False,
                                     "reason": "a run produced no output digest; a missing digest "
                                               "is refused as firmly as a mismatch"}
        else:
            report["correctness"] = {
                "decided": True,
                "baseline_stable_across_paddings": len(set(b_d)) == 1,
                "honoured_stable_across_paddings": len(set(f_d)) == 1,
                # Vectorizer flags change the SCHEDULE of f32 arithmetic, so the two arms are NOT
                # required to be bit-identical -- only each stable across paddings. A difference
                # between arms is reported, never asserted away.
                "arms_bit_identical": set(b_d) == set(f_d) and len(set(b_d)) == 1,
                "baseline_digests": sorted(set(b_d)), "honoured_digests": sorted(set(f_d))}
        walls = {t: [x.get("wall_ns_min") for x in rows[t] if x.get("wall_ns_min")] for t in rows}
        if all(walls.values()):
            lo, hi = min(walls["clang_vectorizers_on"]), min(walls["package_cflags_honoured"])
            report["wall"] = {"clang_on_ns_min": lo, "honoured_ns_min": hi,
                              "ratio_clang_on_over_honoured": lo / hi if hi else None,
                              "board_noise_floor_pct": 1.9,
                              "note": "NOT a speed claim. A ratio inside the board's own noise "
                                      "floor is not a result."}

    product = new_product("pkg-cflags-ab", target="k1_spacemit", version=1)
    out = product.add_artifact("report.json")
    out.write_text(json.dumps(report, indent=2))
    if a.json:
        a.json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report.get("static_delta", report), indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
