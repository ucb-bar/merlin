---
title: Reproducing the whole-model gemmini result
kind: guide
status: current
owner: compiler
last_verified: 2026-08-29
related: [whole_model_on_accelerator, generated_target_repos, gemmini_experiment, reproducibility, adding_a_target]
code_refs: [merlin/python/merlin/runtime/backends/spike_model.py, merlin/python/merlin/llvmlower/device_build.py, merlin/python/merlin/targetgen/oot_runner.py, merlin/python/merlin/targetgen/publish.py]
---

# Reproducing the whole-model gemmini result

A runbook for a fresh machine: build one artifact that holds host **and** device code, run a whole
model with the matmuls on gemmini, and check the answer on both a functional simulator and
cycle-accurate RTL. Concepts live in [whole_model_on_accelerator](whole_model_on_accelerator.md);
this page is the operational path and the traps.

## What you will and will not have shown

Be precise about this before quoting anything, because the two runs make different claims.

| result | claim | not a claim |
|---|---|---|
| bit-exact on spike + `libgemmini` | the lowering computes the right numbers | anything about the hardware |
| completes on cycle-accurate Verilator | the artifact executes on the RTL | that it matches spike |
| kernels certified at L3 (`rtl_verilator`) | those kernels are right on the hardware | that the whole model is |

**Measured, and unexplained:** the same artifact is bit-identical on spike and diverges on RTL
(cos 0.973, all elements differing). Do not present the spike number as an RTL result.

## Prerequisites

Set these in `.env` (copy `.env.example`); every path below is read from there, never hardcoded.

- `MERLIN_EXT_CHIPYARD` — a chipyard checkout with a **built** Verilator simulator for the config you
  intend to run, plus its `riscv-tools` (gcc, spike, `libgemmini.so`).
- LLVM/MLIR **23** at `third_party/llvm-install` (version *and* commit must match the `llvm:` block of
  any codegen package you build — the out-of-tree C++ API moves between them).
- A model capture under `out/artifacts/recaptures/<model>/`. These are multi-GB and untracked, so a
  fresh clone has none; see [model2mlir](model2mlir.md) to produce one.

Everything below uses `small_llama_int8_consistent` and the `GemminiRocketConfig` simulator. Both are
parameters, not requirements.

## 1. Find out what the SoC actually is

Two facts decide whether the image can run at all, and **both must come from the target, not from a
default**. The generated device tree is the authority:

```sh
DTS=$(ls "$MERLIN_EXT_CHIPYARD"/sims/verilator/generated-src/*"$CONFIG"*/*.dts | head -1)
grep -A2 'memory@' "$DTS"     # -> reg = <0x80000000 0x10000000>  i.e. base and SIZE
grep 'riscv,isa' "$DTS"       # -> rv64imafdcbzicsr_..._xrocket   note: no `v`
```

For `GemminiRocketConfig` that reads **256 MB at 0x80000000**, and an ISA string with **no vector
extension**. Both matter, and getting either wrong costs an hour of simulation before it shows.

## 2. Build the artifact

```python
from merlin.llvmlower.device_build import DeviceRouting
from merlin.runtime.backends import spike_model

# Build for the ISA the DTS declared. The default is -march=rv64gcv, and this core has no `v`.
SCALAR = ["-march=rv64gc_zba_zbb_zbs_zfh", "-mabi=lp64d", "-mcmodel=medany",
          "-O2", "-ffreestanding", "-fno-builtin"]

routing = DeviceRouting(
    device="gemmini",
    package_dir="out/artifacts/targets/gemmini/<a certified OOT package>",
    operand_dtype="int8", accum_dtype="i32",
    select=lambda shape: True,          # the placement decision, passed in — None moves nothing
)

r = spike_model.build(
    "out/artifacts/recaptures/small_llama_int8_consistent", "<workdir>",
    arena_mb=128,
    dram_base=0x80000000,
    dram_bytes=0x10000000,              # from the DTS — see the trap below
    int8_compute=True,
    cflags_override=SCALAR,
    device=routing,                     # omit for the host-only control
)
```

Expect `routed 15 contraction(s) to gemmini across 4 signature(s)` and `linked 4 kernel(s) + shim`.

Verify before spending simulator time — this is seconds, and catches both traps:

```sh
readelf -lW <workdir>/model.elf | awk '/LOAD/{print $3, $5}'   # every vaddr inside DRAM
"$RISCV"/bin/riscv64-unknown-elf-objdump -d <workdir>/model.elf \
  | grep -cE '^\s+[0-9a-f]+:\s+[0-9a-f]{8}\s+v(set|le|se|add|mul|mac)'   # must be 0
```

## 3. Run on spike (minutes)

The stock chipyard spike has **no gemmini built in**; it comes from an extlib:

```sh
spike --isa=rv64gc_zba_zbb_zbs_zfh \
      --extlib="$RISCV"/lib/libgemmini.so --extension=gemmini \
      -p1 -m0x80000000:0x10000000 <workdir>/model.elf
```

It prints `Gemmini extension configured with: dim = 16`, then `OUT <n> <bit patterns>` and
`METRIC cycles`. Build the host-only variant too and compare the two `OUT` lines: they should be
**bit-identical**. That is the compiler check — offload must not change the answer.

An **all-zero** output is the `libgemmini` / `gemmini_params.h` header-skew signature, not a compiler
bug. Check that first.

Spike's cycle counter is not a timing model. The drop from ~336M (host-only) to ~13.5M cycles is
evidence the contractions *left the host*, not a speedup measurement.

## 4. Run on cycle-accurate RTL (hours)

```sh
cd "$MERLIN_EXT_CHIPYARD"/sims/verilator
stdbuf -o0 ./simulator-chipyard.harness-$CONFIG <workdir>/model.elf
```

`stdbuf -o0` is not optional: the simulator's stdout is block-buffered, so without it a running job
shows an empty log and looks hung.

Budget the run **before** launching it. Divide the spike cycle count by the observed Verilator rate
(order 10^3–10^4 cycles/s, and it drops with load). The offloaded artifact is ~22M cycles ≈ hours; the
host-only control is ~25× that, i.e. **days** — it is not a practical control, and estimating that
first saves a wasted run.

Compare the `OUT` line against spike's. Ours agrees on neither: cos 0.973 with every element
differing — a systematic arithmetic difference between `libgemmini` and the gemmini RTL, not a
localized defect.

## 5. Certify and publish a package

Certification is per-rung and must name the oracle it was earned on:

```sh
P=out/artifacts/targets/gemmini/<package>
for r in $P/rungs/*.interface.mlir; do
  python -m merlin.targetgen.oot_runner --package "$P" --input "$r" \
      --run-id "vcert_$(basename "$r" .interface.mlir)" --simulator verilator
done
python -m merlin.targetgen.publish record-cert --target gemmini --champion <package> \
      --results out/runs/gemmini_contract/runs/gemmini-contract/vcert_*/results.yaml
python -m merlin.targetgen.publish promote --target gemmini --champion <package>
python -m merlin.targetgen.publish publish --target gemmini --dry-run
```

`--simulator verilator` gives `derived_from_rtl: true, cycle_accurate: true`; `spike` gives a
functional pass. `record-cert` is what carries the verdict onto the package — without it nothing can
be promoted, because `promote` asks the gate and the gate asks for the certification. See
[generated_target_repos](generated_target_repos.md).

## Traps

Each of these cost real time, and each one *looks* like a different bug than it is.

| symptom | cause |
|---|---|
| dies mid-run on a TileLink `PutPartial` monitor assertion | image built with `dram_bytes=None`, so the arena sits at `0xC0000000` — memory the chip does not have. Pass `dram_bytes` from the DTS. |
| `TRAP mcause=2`, `mtval` low bits `0x57` | an RVV instruction on a core with no `v`. Build for the DTS's ISA string. |
| empty log on a job that is clearly running | block-buffered stdout — use `stdbuf -o0` |
| all-zero output | `libgemmini` vs `gemmini_params.h` skew |
| a spike run traps on a valid-looking instruction | the `--isa` string omits an extension the image was compiled with (e.g. Zbs `bseti`) |
| a test hangs for minutes in a git worktree | worktrees carry no `.env`, no `third_party/llvm-install`, and no derived-facts cache, so a lookup re-derives from RTL. Set `MERLIN_OUT_ROOT` at the main checkout and symlink the LLVM install. |

## What to record with a result

A number without its tier and its tree is not citable. Quote `tier_reached`, not a bare score; say
which oracle produced it (`rtl_verilator` vs a functional model); and keep the provenance block the
run emits. A result attributed to the wrong hardware revision is worse than no result, because it
gets cited — see [hardware pins](../../.claude/skills/hardware-pins/SKILL.md).
