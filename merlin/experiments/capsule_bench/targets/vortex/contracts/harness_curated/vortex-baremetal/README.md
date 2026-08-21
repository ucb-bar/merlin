# Curated Vortex bare-metal harness

Board support the capsule-bench arms link against — the Vortex analog of
`gemmini_capsule_bench_v0/contracts/harness_curated/gemmini-rocc-tests` (linker/crt/headers, **no
kernels**). Everything here is runner-owned; the agent supplies exactly one device symbol.

## The contract

The agent's backend emits **`merlin_kernel_body`** (see `include/merlin_vortex_abi.h`). That is all.

- **Work distribution is hardware.** Vortex's **KMU** launches every `(block, thread)` coordinate at
  the kernel entry, so no software spawn is needed. Identity comes from the CTA CSRs
  (`0xCD3` thread_id.x / `0xCD6` block_id.x / `0xCD9` block_dim.x — the ISA spec sheet has the full
  map), read with a plain `csrr`. **Verified:** a kernel using zero Vortex headers runs correctly on
  simx.
- **Stock LLVM only.** No Vortex fork on the agent's side.

## Why each piece is board support and not an answer

| Staged | What | Why it is not an answer |
|---|---|---|
| `lib/libvortex_curated.a` | Vortex's `libvortex2.a` **minus `vx_spawn`** | Seven objects (start / syscalls / print×2 / tinyprintf / serial / spawn); only `vx_spawn` distributes work, and it is removed. The rest is startup and I/O. |
| `lib/vx_entry.o` | KMU entry stub | See below — the one unavoidable Vortex-toolchain dependency, isolated here. |
| `lib/vx_start_min.o` | minimal startup | KMU dispatch only. **Must** be linked ahead of the archive — see "Startup" below. |
| `kernel_startup.sh` | link-time feature check | Fails the link if a kernel needs gp/TLS/init_array the minimal startup omits. |
| `link/link*.ld` | linker script | Memory map. |
| `include/` | the ABI header only | No programming-model header (`vx_spawn2.h`) is staged. |
| `vxbin.py` | ELF → `.vxbin` | Image format. |
| `host/merlin_vx_host` | generic capsule host driver | Allocates one buffer per operand, fills inputs from a fixed LCG, launches over the declared grid, prints `OUT`/`METRIC`/`DONE`. Workload-agnostic: it knows sizes and dtypes, never the operation. |

## The one Vortex-toolchain dependency, and how it is isolated

`__kernel` expands to `__attribute__((annotate("vortex.kernel"), used, retain))`, which the **Vortex
LLVM backend** turns into the `.vx_entry` stub / VXSYMTAB footer the KMU launches from. An image
without that metadata **hangs** — verified twice: an un-annotated kernel built with the Vortex
toolchain, and a stock-LLVM build where the attribute is present in source but no backend pass acts on
it. Both hang; both are fixed by this stub.

So `src/vx_entry.cpp` is prebuilt **once** with the Vortex toolchain and does nothing but dispatch to
`merlin_kernel_body`. It contains no work distribution, no identity computation, and no compute. The
agent's object is then built with stock LLVM and linked against it.

**Validated end to end on simx:** harness entry (Vortex-built) + `libvortex_curated.a` + a body built
with stock clang carrying no Vortex headers → `PASSED`.

## What the harness does NOT solve: divergence

`+xvortex` makes the Vortex fork auto-insert `vx_split_n`/`vx_join` around divergent branches; stock
clang emits a bare branch and **zero** reconvergence, which is silently wrong on hardware whenever
threads in a warp disagree. The entry-stub trick does not help here — divergence lives inside the
agent's own code. Handling it (or emitting predicated, divergence-free code) is a real compiler
obligation of the backend under test, and is what the arm-3 `SPLIT_JOIN_BALANCED` check looks for.

## Determinism contract

The host driver fills inputs from an LCG (`s = s*1664525 + 1013904223`) that
`targetgen/vortex_oracle.fill_f32/​fill_i8/​fill_i32` reproduce exactly, so goldens can be computed
offline without running the device. **Verified bit-for-bit** (64/64 f32 elements) against a real simx
run. `merlin/tests/targetgen/test_vortex_oracle.py` reads the C source and fails if the recurrence or
any per-dtype derivation drifts from the Python — changing one without the other would silently grade
every capsule against the wrong inputs.

## Both oracle tiers (simx L2 and rtlsim L3)

The same staged harness and the same `.vxbin` run under either driver — only `VORTEX_DRIVER` changes.
Measured on the pilot capsules:

| tier | driver | wall / capsule | simulated-cycle rate |
|---|---|---|---|
| L2 | simx (cycle-approximate) | ~0.1 s | ~129 kHz |
| L3 | rtlsim (Verilator, cycle-exact) | ~0.4 s | ~24 kHz |

rtlsim is only ~4x the wall time at this scale, so every pilot capsule declares **both** tiers —
cycle-exact grading everywhere is affordable. Budget roughly **1 minute of rtlsim per 1.4M cycles**
as capsules grow. The fidelity difference is real and is why L3 exists: the same 2144-instruction
kernel reports **12620 cycles on simx and 9773 on rtlsim**.

### Geometry parity is enforced, and you should not skip it

simx and rtlsim are separately-built simulators with no structural link, so they drift. When they do,
the failure is *silent*: an rtlsim left over from an unrelated sweep was built 2 clusters x 2 cores,
and the mismatched launch grid left output buffers **unwritten** — the host read back `0xBAADF00D`
poison, which grades as a wrong answer from the backend rather than a broken rig.

So the host driver reports what it actually ran on:

```
METRIC geometry clusters=1 cores=1 warps=8 threads=8
```

and `vortex_oracle.run_image` raises `VortexUnavailable` if that does not match `FROZEN_GEOMETRY`.
Rebuild a drifted driver with:

```bash
cd $VORTEX_HOME/build
CONFIGS="-DVX_CFG_NUM_CLUSTERS=1 -DVX_CFG_NUM_CORES=1 -DVX_CFG_SOCKET_SIZE=1 \
         -DVX_CFG_NUM_WARPS=8 -DVX_CFG_NUM_THREADS=8" \
  make -C sw/runtime/simx   DESTDIR=$PWD/sw/runtime
CONFIGS="...same..." make -C sw/runtime/rtlsim DESTDIR=$PWD/sw/runtime
```

## Build

```bash
scripts/build_harness.sh --vortex $VORTEX_HOME --out <staging-dir> [--xlen 64]
```

Run by the experiment owner (it reads the Vortex tree, a denied surface for the arms); the staged
output is self-contained and carries no pointer back to it.

## Startup: why the archive's `vx_start` must not be used

`libvortex*.a` is built once for every app, so its `vx_start.S` is compiled with **all** startup
features on (`NEED_GP`, `NEED_TLS`, `NEED_INITFINI`) — the library cannot know what a given kernel
needs. Linking it means every launched hart runs `__init_tls` and `__libc_init_array` at CTA entry.

At this geometry that is **256 harts doing libc init concurrently across 4 non-coherent L1 dcaches**.
The result is corrupted state and output buffers that are never written — the host reads back
`0xBAADF00D`. Critically it is **invisible until you run multi-core RTL**: simx passes (it does not
model the caches faithfully) and a 1-core rtlsim build passes; the 4-core build fails 64/64.

So the harness stages `vx_start_min.o` (KMU dispatch only) and the link puts it **ahead of the
archive**. After linking, `kernel_startup.sh` is run against the image and the link fails if the
kernel genuinely needs any of those features — compiler-emitted kernels should need none.

This also dominated the cost model: with the archive's startup, V0 measured 11367 cycles on simx;
with the minimal one, **1086**. Roughly 80% of every earlier measurement was startup.

## Measured cost (minimal startup, 2x2+L2)

| capsule | simx cycles | rtlsim cycles |
|---|---|---|
| V0 elementwise f32 | 1086 | 3069 |
| V2 matmul i8 (K=16) | 2529 | 5114 |

rtlsim runs in well under a second per capsule at this size, which is why every capsule declares both
tiers. Budget roughly 1 minute of rtlsim per 1.4M cycles as capsules grow.

## Known rough edges

- Only `xlen=64` is validated.
