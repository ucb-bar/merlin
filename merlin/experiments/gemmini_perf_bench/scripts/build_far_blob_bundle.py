#!/usr/bin/env python3
"""Link a whole-model bare-metal ELF whose CONSTANT BLOB lives at a fixed absolute address.

WHY THIS EXISTS AS A SCRIPT. `targetgen.contract.compile.compile_lowered_to_elf` cannot build this
shape: `HarnessBuildRecipe` carries no extra-flag or extra-source field, `link_elf` fixes its object
list, and the gemmini renderer emits C initializer lists in which a 1.2 GiB blob is not expressible
at all. The mechanism itself was proven once, by hand, in
`out/artifacts/perf-bench/gemmini/tiny_llama_bundle_20260908/` -- but the script that produced it
was never kept, so the evidence was reusable and the recipe was not. This is that recipe.

WHEN IT IS NEEDED. Under `-mcmodel=medany` every reference is PC-relative within +/-2 GiB. An image
larger than that does not fail to link, it links and MIS-ADDRESSES: correct arithmetic on the wrong
bytes. tiny_llama projects 2.26 GiB (1.21 GiB of weights beside a 1.00 GiB arena) and FAULTS the
window; with the blob placed far, only the ~1.03 GiB near region has to be reachable. Decide with
`PackPlan.projected_image_bytes(const_is_far=True)` and `liveness.preconditions.medany_span` BEFORE
spending the minutes to link something nobody should run.

FIVE THINGS THAT ARE EASY TO GET WRONG, each of which cost a build here:

1. **Support objects link FIRST.** `crt.S` reaches `_init` with a JAL (+/-1 MiB) and a whole-model
   kernel's `.text` is ~1 MiB on its own, so putting the kernel ahead of crt puts `_init` out of
   branch range. The shipped bundle's link receipt records exactly this ordering.
2. **Pass the ARGUMENT LIST, not the whole call.** `harness_abi.warm_profile_invocation` builds
   `<entry>(<args>);` *and* appends the target's completion fence. Handing it a finished call loses
   the fence, and a profile that closes its cycle window with accelerator work outstanding measures
   the wrong window while printing a plausible number.
3. **`declarations` already contains the gate** as `merlin_gate_check()` -- concatenating the `gate`
   fragment as well redefines it.
4. **The harness needs the target's own includes**: `stdint.h` for the cycle counters' `uint64_t`,
   `stdio.h`, and the target testutils header, which is where `counter_read` /
   `counter_snapshot_take` are declared and the `MAIN_*` event codes defined.
5. **`merlin_reference` and `merlin_mutable_blob` are declared `extern`** by the fragments, so the
   BUNDLE owns their definitions: the reference as read-only bytes, the arena in `.bss` (crt zeroes
   it and every element is written before it is read, so it costs no image bytes).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def compose_harness(fragments: Path, out: Path, *, target: str) -> Path:
    """Write `harness.c` from bundle_harness fragments plus the warm-then-measure profile main."""
    from merlin.perf import hw_counters as H
    from merlin.perf.warm_profile_harness import (WarmProfileContract,
                                                  render_target_warm_then_measure_main)

    def frag(name: str) -> str:
        return (fragments / f"harness_{name}.c.frag").read_text(encoding="utf-8")

    call = frag("call").strip()
    entry = call.split("(", 1)[0]
    args = call[len(entry) + 1:].rstrip().rstrip(";").rstrip()
    if not args.endswith(")"):
        raise SystemExit(f"the call fragment does not close its argument list: {args[-40:]!r}")
    args = args[:-1]

    counters = H.counters_for_target(target)
    if counters.get("status") != "derived":
        raise SystemExit(f"counters not derived for {target}: {counters}")
    bracket = H.occupancy_partition_bracket(
        Path(counters["header"]).read_text(encoding="utf-8"), slots=8)
    if not bracket["closes_accelerator_busy"]:
        raise SystemExit("refusing a partial counter partition: busy would be a lower bound "
                         "reported as a total")

    main = render_target_warm_then_measure_main(
        target=target, arguments=args, prepare_input=frag("reseed"),
        validate_outputs=frag("validate").strip(), contract=WarmProfileContract(),
        counter_bracket=bracket)
    prologue = "\n".join(["#include <stdint.h>", "#include <stdio.h>",
                          f'#include "include/{target}_testutils.h"', ""])
    out.write_text("\n".join([prologue, frag("declarations"), main]) + "\n", encoding="utf-8")
    return out


def write_bundle_data(bundle: Path, *, arena_bytes: int, reference_bin: Path) -> Path:
    """Define the two symbols the harness fragments leave `extern`."""
    path = bundle / "bundle_data.S"
    path.write_text("\n".join([
        "/* Definitions for the symbols the harness declarations leave extern. */",
        '    .section .rodata, "a"',
        "    .balign 64",
        "    .global merlin_reference",
        "merlin_reference:",
        f'    .incbin "{reference_bin}"',
        "",
        '    .section .bss, "aw", @nobits',
        "    .balign 64",
        "    .global merlin_mutable_blob",
        "merlin_mutable_blob:",
        f"    .zero {int(arena_bytes)}",
        "",
    ]), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True,
                    help="dir holding the harness_*.c.frag set, const_blob.bin, plan.json")
    ap.add_argument("--llvm", type=Path, required=True, help="package-emitted llvm-dialect MLIR")
    ap.add_argument("--reference", type=Path, required=True,
                    help="raw reference bytes for merlin_reference (no container header)")
    ap.add_argument("--const-blob-base", required=True, help="absolute address, e.g. 0x200000000")
    ap.add_argument("--target", default="gemmini")
    ap.add_argument("--elf-name", default=None)
    args = ap.parse_args()

    from merlin.runtime.backends import base as backends
    from merlin.targetgen import bundle_harness as BH
    from merlin.targetgen.contract.compile import llvm_mlir_to_object

    bundle = args.bundle.resolve()
    base = int(args.const_blob_base, 16)
    plan = json.loads((bundle / "plan.json").read_text(encoding="utf-8"))
    recipe = backends.harness_build_recipe(args.target)
    work = bundle / ".build"
    work.mkdir(parents=True, exist_ok=True)
    cc = recipe.compiler
    includes = [f"-I{root}" for root in recipe.include_roots]
    cflags = list(recipe.cflags) + list(BH.far_blob_compile_flags(base))
    ldflags = list(recipe.ldflags) + list(BH.far_blob_link_flags(base))

    def run(step: str, cmd: list[str]) -> None:
        started = time.time()
        done = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  {step}: rc={done.returncode} {time.time()-started:.1f}s", flush=True)
        if done.returncode != 0:
            print((done.stderr or "")[-3000:])
            raise SystemExit(f"{step} failed")

    compose_harness(bundle, bundle / "harness.c", target=args.target)
    (bundle / "const_blob.S").write_text(
        BH.render_far_blob_assembly(blob_path=str(bundle / "const_blob.bin")), encoding="utf-8")
    data_s = write_bundle_data(bundle, arena_bytes=plan["mutable_bytes"],
                               reference_bin=args.reference.resolve())

    obj = llvm_mlir_to_object(args.llvm.read_text(encoding="utf-8"), work, target=args.target)
    march = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany"]
    run("assemble const_blob.S", [cc, *march, "-c", str(bundle / "const_blob.S"),
                                  "-o", str(work / "const_blob.o")])
    run("assemble bundle_data.S", [cc, *march, "-c", str(data_s),
                                   "-o", str(work / "bundle_data.o")])
    run("compile harness.c", [cc, *cflags, *includes, "-c", str(bundle / "harness.c"),
                              "-o", str(work / "harness.o")])
    support: list[Path] = []
    for source in recipe.support_sources:
        source = Path(source)
        if source.suffix in (".c", ".S", ".s"):
            unit = work / f"{source.stem}.o"
            run(f"compile {source.name}", [cc, *cflags, *includes, "-c", str(source),
                                           "-o", str(unit)])
            support.append(unit)
        else:
            support.append(source)

    elf = bundle / (args.elf_name or "model_warm1_measure1.elf")
    # Support objects FIRST -- see (1) in the module docstring.
    run("link", [cc, *cflags, *includes, "-T", str(recipe.link_script), "-o", str(elf),
                 *[str(o) for o in support], str(work / "harness.o"), str(obj),
                 str(work / "const_blob.o"), str(work / "bundle_data.o"), *ldflags])
    print(f"ELF: {elf} ({elf.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
