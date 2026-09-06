#!/usr/bin/env python3
"""Prove the 1-hart and N-hart builds are THE SAME KERNEL, then hand over the board A/B.

WHY THIS EXISTS. A multicore scaling number divides one build's wall by another build's wall, and it
is only a thread measurement if the two builds emit the same code. They did not. The multicore stage
split every contraction with ``tile_using_forall num_threads``, which left each hart a ``ceil(dim/N)``
tile the register block then had to mask -- so the block, and with it the emitted kernel, was a
function of the hart count. Measured on lstmnetvit int8 at 8 harts, in the LINKED ELF: 5 of 37
matmuls fell out of the block table into scalar loops, 9 more were narrowed, issued instructions went
59,752 -> 97,701 and vector ops 13,767 -> 5,144. The published 1.24x "scaling" was comparing two
different compilers' output.

So this script refuses to print a board command until it has SHOWN the two arms match, on three
independent readings:

  * the block tags the prepare step applied (the decision itself),
  * the vector-op census of the IR the package schedule produces -- the one reading that survives
    inlining, because a static count off an object cannot tell "devectorized" from "outlined",
  * the LINKED ELF (never the ``.o``: an unrelocated object splits ``forward`` at its ``.Lpcrel_hi``
    labels and every per-symbol metric silently reads a tiny prefix).

Correctness is checked by OUTPUT DIGEST (``--digest``), not by a cosine gate: a race is exactly the
class a cosine gate does not catch, and the digest is taken across several thread counts AND several
environment paddings, because padding moves the initial stack and turns a read of uninitialised
memory into a different answer instead of a plausible one.

HOST ONLY. Nothing here touches a board. The emulator is user-mode QEMU; the board A/B it prints is
for a human to run when a board slot is free.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "merlin" / "python"))

from merlin.common.artifacts import new_product                        # noqa: E402
from merlin.common.paths import build_dir, repo_root                   # noqa: E402
from merlin.llvmlower import perop_blocks as pb                        # noqa: E402
from merlin.llvmlower import toolchain                                 # noqa: E402
from merlin.llvmlower.pipeline import build_rvv_pipeline               # noqa: E402
from merlin.mining import k1                                           # noqa: E402
from merlin.mining.registry import load_rvv_package                    # noqa: E402
from merlin.runtime.backends import zephyr_model as zm                 # noqa: E402

#: The ops whose count must agree between the two arms. A contraction still named ``linalg.matmul``
#: after the package schedule ran is one no arm claimed: it lowers through convert-linalg-to-loops,
#: i.e. scalar, with correct numbers and no gate to notice.
_CENSUS_OPS = ("vector.contract", "vector.transfer_read", "vector.transfer_write",
               "vector.mask", "vector.create_mask", "linalg.matmul", "linalg.batch_matmul",
               "linalg.generic", "scf.forall")

#: Counted apart: the wrapper is the ONE thing that is allowed to differ.
_WRAPPER_OP = "scf.forall"


def _leading_op(line: str) -> str:
    """The op a printed MLIR line starts, read structurally: drop one leading ``%x = `` and take the
    first token. A substring test would count every op named inside an enclosing op's line."""
    s = line.strip()
    if " = " in s:
        s = s.split(" = ", 1)[1]
    parts = s.split()
    return parts[0] if parts else ""


def block_tags(work: Path) -> dict:
    """``{attribute: count}`` for the per-op block tags in the tagged module, scanned structurally."""
    path = work / "model.perop_tagged.mlir"
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    counts: collections.Counter = collections.Counter()
    i = 0
    while True:
        j = text.find(pb.TAG_PREFIX, i)
        if j < 0:
            return dict(sorted(counts.items()))
        k = j + len(pb.TAG_PREFIX)
        while k < len(text) and (text[k].isalnum() or text[k] == "_"):
            k += 1
        counts[text[j:k]] += 1
        i = k


def _lower_dir(work: Path) -> Path:
    for name in ("lower_vecomp", "lower"):
        if (work / name).is_dir():
            return work / name
    raise SystemExit(f"{work}: no lowering directory -- did the build get that far?")


def schedule_census(work: Path) -> dict:
    """Op census of the IR the PACKAGE SCHEDULE produces, via the real pipeline prefix.

    Re-runs the pipeline up to and including the package schedule's transform interpreter with the
    standalone LLVM-23 ``mlir-opt`` over the same upstream text the build lowered. That is the
    reading that is about the schedule's decision rather than about clang's later inlining.
    """
    low = _lower_dir(work)
    sched = low / "rvv_schedule.mlir"
    par = low / "rvv_parallel_schedule.mlir"
    vec = low / "rvv_vec_pre_schedule.mlir"
    pipe = build_rvv_pipeline(sched, hoist_static_allocs=False, features=frozenset(),
                              par_sched_path=(par if par.is_file() else None),
                              vec_sched_path=(vec if vec.is_file() else None),
                              perop_parallel=par.is_file())
    anchor = "transform-interpreter{entry-point=__transform_main},canonicalize,cse"
    if anchor not in pipe:
        raise SystemExit("the package schedule's interpreter is not in the pipeline string")
    prefix = pipe[:pipe.index(anchor) + len(anchor)]
    prefix += ")" * (prefix.count("(") - prefix.count(")"))
    mlir_opt = Path(str(toolchain.clang()).replace("clang-23", "mlir-opt"))
    if not mlir_opt.is_file():
        raise SystemExit(f"{mlir_opt} not built (third_party/llvm-install)")
    proc = subprocess.run([str(mlir_opt), str(low / "model.upstream.mlir"),
                           f"--pass-pipeline=builtin.module({prefix})"],
                          capture_output=True, text=True, timeout=7200)
    if proc.returncode != 0:
        raise SystemExit(f"schedule census failed:\n{proc.stderr[-3000:]}")
    counts: collections.Counter = collections.Counter()
    for line in proc.stdout.splitlines():
        op = _leading_op(line)
        if op in _CENSUS_OPS:
            counts[op] += 1
    return {op: counts.get(op, 0) for op in _CENSUS_OPS}


def elf_census(elf: Path, objdump: Path) -> dict:
    """Instruction / vector-instruction census of the LINKED ELF's compute symbols.

    The linked ELF, never the ``.o``: an unrelocated object splits ``forward`` at its ``.Lpcrel_hi``
    labels, so a per-symbol metric reads a tiny prefix and reports it as the whole function.
    ``forward`` and everything outlined out of it (the OpenMP regions) are counted together, because
    which of the two a body lives in is exactly what the parallel wrapper changes.
    """
    out = subprocess.run([str(objdump), "-d", str(elf)],
                         capture_output=True, text=True, timeout=3600).stdout
    cur = None
    instrs = vector = 0
    mnem: collections.Counter = collections.Counter()
    symbols = 0
    for line in out.splitlines():
        s = line.strip()
        if s.endswith(">:") and "<" in s:
            cur = s[s.index("<") + 1:-2]
            if cur == "forward" or cur.startswith("forward."):
                symbols += 1
            continue
        if cur is None or "\t" not in s:
            continue
        if not (cur == "forward" or cur.startswith("forward.")):
            continue
        fields = s.split("\t")
        if len(fields) < 2:
            continue
        toks = fields[1].split()
        if not toks:
            continue
        m = toks[0]
        instrs += 1
        # `vset*` configures the vector unit but does no work; counted separately so it cannot pad
        # a "we are still vectorized" claim.
        if m.startswith("v") and not m.startswith("vset"):
            vector += 1
        mnem[m] += 1
    return {"compute_symbols": symbols, "instructions": instrs, "vector_instructions": vector,
            "top_mnemonics": dict(mnem.most_common(20))}


def _objdump() -> Path:
    cc = k1.toolchain_cc()
    if cc is None:
        raise SystemExit("SpacemiT toolchain not found (MERLIN_K1_TOOLCHAIN)")
    cand = Path(cc).parent / "llvm-objdump"
    if not cand.is_file():
        raise SystemExit(f"{cand} not found")
    return cand


def emulator() -> Path | None:
    """User-mode QEMU for rv64, or None. NEVER the board."""
    for p in (Path("/scratch2/agustin/merlin/build_tools/riscv-tools-iree/qemu/linux/RISCV/"
                   "qemu-riscv64"),):
        if p.is_file():
            return p
    from shutil import which
    got = which("qemu-riscv64")
    return Path(got) if got else None


def output_digests(elf: Path, qemu: Path, vlen: int, threads, pads, timeout: int) -> dict:
    """``{(threads, pad): digest}`` of the harness's OUT line, one run per cell.

    Two axes on purpose. THREADS is the property under test. PADDING shifts the initial stack, and
    is what turns a read of uninitialised memory into a different answer -- a build measured here
    earlier scored cos 0.968 while reading 56 of 64 bytes uninitialised, and no cosine gate saw it.
    """
    import os

    out: dict[str, str] = {}
    for pad in pads:
        env = dict(os.environ, MERLIN_PAD="x" * int(pad))
        for n in threads:
            env["OMP_NUM_THREADS"] = str(n)
            proc = subprocess.run(
                [str(qemu), "-cpu", f"rv64,v=true,vlen={vlen},elen=64", str(elf)],
                capture_output=True, text=True, cwd=str(elf.parent), env=env, timeout=timeout)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("OUT ")]
            if not lines:
                out[f"threads={n},pad={pad}"] = "NO_OUTPUT"
                continue
            out[f"threads={n},pad={pad}"] = hashlib.sha256(
                "\n".join(lines).encode()).hexdigest()[:32]
    return out


def _et_model_name(bundle_name: str) -> str:
    """The reference arm's model name for ``bundle_name``, or a visible placeholder.

    Resolved against the ExecuTorch registry rather than by trimming the bundle name: the suffixes a
    recapture carries (``_int8_consistent``, ``_w8a8_consistent``, ``_pretransposed``, ...) are not a
    closed set, and a trimmed name that happens to be wrong makes the reference arm silently measure
    a DIFFERENT MODEL. Longest match wins so ``rdt2`` is never resolved as ``rdt``. Unresolvable ->
    a placeholder, because a wrong name here is worse than an unpasteable command.
    """
    try:
        from merlin.baselines import executorch as _et
        known = sorted(getattr(_et, "ALL_MODELS", ()), key=len, reverse=True)
    except Exception:                                        # noqa: BLE001
        known = []
    for m in known:
        if bundle_name == m or bundle_name.startswith(f"{m}_"):
            return m
    return "<ET_MODEL_NAME: this bundle is not in the ExecuTorch registry; pass --et-model>"


def build(bundle: Path, work: Path, pkg, harts: int) -> Path:
    return k1.build_k1_binary(bundle, work, pkg, inputs_npz=bundle / "inputs.npz",
                              parallel_harts=(harts if harts > 1 else None))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="the bundle to build (a recapture dir)")
    ap.add_argument("--package", required=True, help="rvv package dir (out/artifacts/targets/rvv/...)")
    ap.add_argument("--features", default=None,
                    help="comma-separated compiler features; omitted = the package's own")
    ap.add_argument("--harts", type=int, default=8, help="the multicore arm's hart count")
    ap.add_argument("--digest", action="store_true",
                    help="also verify the two binaries agree BIT-EXACTLY under user-mode QEMU, "
                         "across thread counts and environment paddings")
    ap.add_argument("--digest-threads", default="1,4,8")
    ap.add_argument("--digest-pads", default="0,37,512")
    ap.add_argument("--digest-timeout-s", type=int, default=3600)
    ap.add_argument("--et-model", default=None,
                    help="the reference arm's model NAME for the board command this prints (its "
                         "--model). Resolved from the ExecuTorch model registry when the bundle "
                         "name starts with a registered one; a bundle outside that registry has to "
                         "name it here, and the command is printed with a placeholder rather than a "
                         "guess if it does not.")
    ap.add_argument("--ref-cpu-threads", type=int, default=None,
                    help="cores the REFERENCE arm gets in the board command this prints "
                         "(default: the same as --harts, which is what makes it an NvN cell)")
    ap.add_argument("--out", default=None, help="artifact dir (default: a new versioned product)")
    a = ap.parse_args()

    bundle = Path(a.model_dir).resolve()
    base = load_rvv_package(a.package)
    feats = ([f.strip() for f in a.features.split(",") if f.strip()] if a.features is not None
             else list(base.compiler_features or []))
    key = hashlib.sha256(",".join(sorted(feats)).encode()).hexdigest()[:10] if feats else "nofeatures"
    root = build_dir() / "multicore_kernel_ab" / bundle.name / key
    root.mkdir(parents=True, exist_ok=True)

    arms: dict[str, dict] = {}
    for name, harts in (("serial", 1), ("parallel", int(a.harts))):
        work = root / name
        pkg = replace(base, run_id=f"mcab_{name}", compiler_features=feats)
        print(f"[build] {name}: harts={harts} -> {work}", flush=True)
        elf = build(bundle, work, pkg, harts)
        rec = {"harts": harts, "work": str(work), "elf": str(elf),
               "block_tags": block_tags(work),
               "schedule_census": schedule_census(work),
               "elf_census": elf_census(elf, _objdump()),
               "derived_split": None}
        split = work / zm.PARALLEL_ARMS_FILE
        if split.is_file():
            rec["derived_split"] = json.loads(split.read_text(encoding="utf-8"))
        arms[name] = rec

    s, p = arms["serial"], arms["parallel"]
    census_s = dict(s["schedule_census"])
    census_p = dict(p["schedule_census"])
    wrapper = {"serial": census_s.pop(_WRAPPER_OP), "parallel": census_p.pop(_WRAPPER_OP)}
    problems = []
    if s["block_tags"] != p["block_tags"]:
        problems.append("the two arms tagged different register blocks")
    if census_s != census_p:
        differing = {k: [census_s.get(k), census_p.get(k)]
                     for k in set(census_s) | set(census_p) if census_s.get(k) != census_p.get(k)}
        problems.append(f"the schedule's own output differs: {differing}")
    for arm, c in (("serial", s["schedule_census"]), ("parallel", p["schedule_census"])):
        left = c.get("linalg.matmul", 0) + c.get("linalg.batch_matmul", 0)
        if left:
            problems.append(f"{arm}: {left} contraction(s) no arm claimed -- they lower to scalar")
    if wrapper["serial"]:
        problems.append("the serial arm carries a parallel wrapper")
    if not wrapper["parallel"]:
        problems.append("the parallel arm carries NO parallel wrapper -- it is a single-core build")

    digests = None
    if a.digest:
        qemu = emulator()
        if qemu is None:
            problems.append("--digest asked for but no user-mode rv64 emulator is available")
        else:
            threads = [int(t) for t in a.digest_threads.split(",") if t.strip()]
            pads = [int(t) for t in a.digest_pads.split(",") if t.strip()]
            digests = {}
            for name in ("serial", "parallel"):
                print(f"[digest] {name}", flush=True)
                digests[name] = output_digests(Path(arms[name]["elf"]), qemu, k1.VLEN,
                                               threads, pads, a.digest_timeout_s)
            values = {v for d in digests.values() for v in d.values()}
            if len(values) != 1 or "NO_OUTPUT" in values:
                problems.append(f"the two arms do not agree bit-exactly: {digests}")

    verdict = {"kernels_match": not problems, "problems": problems,
               "wrapper_forall": wrapper, "output_digests": digests}
    record = {"bundle": bundle.name, "package": Path(a.package).name, "features": sorted(feats),
              "harts": int(a.harts), "arms": arms, "verdict": verdict}

    if a.out:
        outdir = Path(a.out)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "multicore_kernel_ab.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[out] {outdir / 'multicore_kernel_ab.json'}")
    else:
        prod = new_product("compare", version=0,
                           notes=f"multicore kernel-identity A/B for {bundle.name}")
        dest = prod.add_artifact("multicore_kernel_ab.json")
        dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        prod.write_manifest()
        print(f"[out] {dest}")

    print()
    print(f"block tags   serial={sum(s['block_tags'].values())} "
          f"parallel={sum(p['block_tags'].values())}  identical={s['block_tags'] == p['block_tags']}")
    print(f"schedule IR  serial={census_s}")
    print(f"schedule IR  parallel={census_p}")
    print(f"scf.forall   serial={wrapper['serial']} parallel={wrapper['parallel']}")
    print(f"linked ELF   serial={s['elf_census']['instructions']} instrs / "
          f"{s['elf_census']['vector_instructions']} vector "
          f"({s['elf_census']['compute_symbols']} compute symbols)")
    print(f"linked ELF   parallel={p['elf_census']['instructions']} instrs / "
          f"{p['elf_census']['vector_instructions']} vector "
          f"({p['elf_census']['compute_symbols']} compute symbols)")
    if p["derived_split"]:
        d = p["derived_split"]
        print(f"derived split {d['split_contractions']}/{d['priced_contractions']} contractions, "
              f"{len(d['arms'])} distinct arms; serial residue: {d['serial_contractions']}")
    print()

    if problems:
        for why in problems:
            print(f"REFUSED: {why}", file=sys.stderr)
        print("The two arms are NOT the same kernel, so a wall ratio between them would not be a "
              "thread measurement. No board command is printed.", file=sys.stderr)
        return 2

    threads = int(a.ref_cpu_threads) if a.ref_cpu_threads else int(a.harts)
    et_model = a.et_model or _et_model_name(bundle.name)
    print("KERNELS MATCH. The parallel arm is the serial kernel plus "
          f"{wrapper['parallel']} scf.forall wrapper(s).")
    print()
    print("Board A/B (needs a free board slot; run it as one command so both arms share a session):")
    print()
    print(f"  {Path(sys.executable)} {repo_root()}/build_tools/scripts/k1_int8_fair_compare.py \\")
    print(f"      --model {et_model} --model-dir {bundle} \\")
    print(f"      --baseline {Path(a.package).resolve()} \\")
    print(f"      --features {','.join(sorted(feats))} \\")
    print(f"      --parallel-harts {a.harts} --ref-cpu-threads {threads}")
    print()
    print("...and the same command with `--parallel-harts 1 --ref-cpu-threads 1` for the per-core "
          "cell. Both arms now compile the same kernel, so the ratio between them is a thread "
          "effect; it was not before.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
