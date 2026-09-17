---
title: Building the pinned LLVM/MLIR toolchain (third_party/llvm-install)
kind: guide
status: current
owner: runtime
last_verified: 2026-08-10
related: [getting_started, llvm_integration, reproducibility, model2mlir, zephyr, targetgen]
code_refs: [merlin/python/merlin/llvmlower/toolchain.py, merlin/python/merlin/llvmlower/pipeline.py,
            merlin/python/merlin/targetgen/contract/toolchain.py, merlin/python/merlin/runtime/backends/zephyr_model.py]
---

# Building the pinned LLVM/MLIR toolchain

`third_party/llvm-install/` is an LLVM/MLIR **23** install that the whole-model path compiles through.
It is **gitignored** (`.gitignore:193-195` covers `llvm-install/`, `llvm-build/` and `llvm-build.log`), so a
fresh clone has none of it — only the `third_party/llvm-project` submodule *pointer*. Nothing warns you at
clone time; you find out when a build asks for `clang-23`. This page is how to produce it.

If you already have an LLVM 23 install elsewhere, skip to [Using an LLVM you already
have](#using-an-llvm-you-already-have) — three environment variables, no build.

## What consumes it

Worth reading before you spend an hour of CPU: the tools you need depend on what you are doing, and the
examples need two of them.

| tool | resolved by | needed for |
|---|---|---|
| `bin/clang-23` | `llvmlower.toolchain.clang()` — override `MERLIN_CLANG` | Every compile to a RISC-V object. The Zephyr backend takes it directly (`runtime/backends/zephyr_model.py:1199`), as do the K1 and spike backends. |
| `bin/mlir-translate` | `llvmlower.toolchain.mlir_translate()` — override `MERLIN_MLIR_TRANSLATE` | The **OpenMP/multicore** path only. `llvmlower/pipeline.py:572` runs it `--mlir-to-llvmir` out-of-process, because the torch-mlir wheel's in-process bridge segfaults on whole-model `omp` IR. Single-hart images do not reach it; every multi-hart image does. |
| `bin/llvm-objdump` | `kernels/decode/objdump.py:18` | Kernel mining and the ELF audits that read back emitted code. |
| `bin/llvm-mc`, `bin/llvm-objcopy` | `targetgen/program_oracle.py` | Assembling target-ISA directives for the targetgen oracle. |
| `lib/`, `include/`, MLIR CMake config | `MERLIN_MLIR_INSTALL` (`targetgen/contract/toolchain.py:19`) | Building out-of-tree C++ MLIR packages (`gemmini-opt`). This is why the install is 4 GB and cannot be pruned to a `bin/` directory. |
| `FileCheck` | `targetgen/rtl_check_runner.py:47` | RTL-check screening. Note it looks in the **build** tree, not the install — see [What is deliberately absent](#what-is-deliberately-absent). |

**For the [`examples/`](../../examples/) (Kodiak, gemmelos) you need exactly two: `clang-23` and
`mlir-translate`.** The RISC-V *linker*, `ar` and `objcopy` in those builds come from the chipyard
riscv-tools and the Zephyr SDK, not from here — so you do not need `lld`, the LLVM runtimes, or a full
cross-toolchain out of this build.

## The pin

```
third_party/llvm-project @ a47bddccec30255619bb8c37fa59700e661d4e66     # upstream main, 2026-06-03
LLVM version 23.0.0git, Optimized build with assertions
```

23 is upstream `main`, not a release: **no distribution packages it**, which is the reason this is built
from source rather than installed. The pin matters for the C++ path — an out-of-tree MLIR package links
against these static libraries and will not load against a different LLVM.

## Build it

```bash
cd <repo>

# 1. the source. The pin is a specific main commit, not a branch tip; GitHub serves it shallowly, so
#    one commit (~2.6 GB) is enough. Drop --depth 1 if your mirror refuses a by-SHA fetch.
git submodule update --init --depth 1 third_party/llvm-project

# 2. configure. This is the flag set that produced the current install, read back from its own
#    CMakeCache.txt -- see "Why these flags" below.
cmake -G Ninja \
  -S third_party/llvm-project/llvm \
  -B third_party/llvm-build \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="X86;RISCV" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=OFF \
  -DMLIR_ENABLE_EXECUTION_ENGINE=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/third_party/llvm-install"

# 3. build and install (ninja picks its own job count; -j is for capping, not raising)
ninja -C third_party/llvm-build
ninja -C third_party/llvm-build install
```

Nothing needs to be exported afterwards: `llvmlower.toolchain` defaults to this path, and the repo's own
install deliberately wins over anything on `PATH` so a checkout stays self-contained.

### The cheap version, if you only want the examples to build

A full `install` is most of the cost. The two tools the examples need can be built and used straight out of
the build tree:

```bash
ninja -C third_party/llvm-build clang mlir-translate
export MERLIN_CLANG=$PWD/third_party/llvm-build/bin/clang-23
export MERLIN_MLIR_TRANSLATE=$PWD/third_party/llvm-build/bin/mlir-translate
```

That skips installing 4 GB of headers and static libraries you only need for the out-of-tree C++ path.
Everything else in this repo that wants LLVM will still report honest-unavailable rather than break.

### Cost

Measured on this machine:

| | size |
|---|---|
| `third_party/llvm-project` (shallow, 1 commit) | 2.6 GB |
| `third_party/llvm-build` | 5.8 GB |
| `third_party/llvm-install` | 4.1 GB |

Roughly **13 GB** for all three. The build tree is disposable *except* for `bin/FileCheck` (below); if you
keep it, incremental rebuilds after a pin bump are cheap.

Wall-clock is not measured here and depends almost entirely on core count — this was built on a 48-core
host. Plan for tens of minutes on a many-core server and a few hours on a laptop, and note that link steps
are memory-hungry: add `-DLLVM_PARALLEL_LINK_JOBS=2` if you have less than ~2 GB of RAM per core.

## Why these flags

Each one is load-bearing; the list is short because everything optional is off.

- **`LLVM_ENABLE_PROJECTS="mlir;clang"`** — MLIR for `mlir-translate`/`mlir-opt`, clang for `clang-23`.
  Nothing else in the monorepo is used.
- **`LLVM_TARGETS_TO_BUILD="X86;RISCV"`** — RISCV is what the boards are; X86 is the host, needed for the
  MLIR execution engine and for host-side reference builds. Dropping either breaks a real path.
- **`CMAKE_BUILD_TYPE=Release` + `LLVM_ENABLE_ASSERTIONS=ON`** — optimized, but assertions kept. A
  miscompile in a lowering pipeline is far cheaper to find as an assertion than as a wrong answer on a
  board you cannot attach a debugger to. `mlir-opt --version` reports `Optimized build with assertions`,
  which is how you confirm you got this rather than a plain Release.
- **`MLIR_ENABLE_EXECUTION_ENGINE=ON`** — the host-side JIT path used to evaluate references.
- **`LLVM_ENABLE_RTTI=OFF`** — the upstream default. It has to *match* in any out-of-tree package linking
  these libraries; a mismatch is a link error, not a runtime surprise.
- **`LLVM_INCLUDE_TESTS=OFF`** — the LLVM test suite is not something this repo runs.
- Static libraries throughout (`BUILD_SHARED_LIBS=OFF`, `LLVM_LINK_LLVM_DYLIB=OFF`, both defaults) —
  the out-of-tree packages link statically.

The current install was configured with host `gcc` 13.3.0. Building it with clang instead is fine and
faster if you have one; nothing downstream depends on which host compiler produced these binaries.

## What is deliberately absent

- **`lld`.** Not in `LLVM_ENABLE_PROJECTS`, on purpose. Where a stock `ld.lld` is needed — the Muon/SIMT
  link — `targetgen/fixed_format/link.py` finds one on `PATH` (your distro's is fine). That is the
  "no forked toolchain" rule: we link with stock tools rather than shipping our own linker.
- **`FileCheck` and `lit`.** `LLVM_INSTALL_UTILS` is `OFF`, so they exist only in the build tree.
  `targetgen/rtl_check_runner.py:47` looks for `third_party/llvm-build/bin/FileCheck` first and falls back
  to `PATH`, then runs a Python-only screen with a warning if it finds neither. **If you use RTL checks,
  keep the build tree** (or add `-DLLVM_INSTALL_UTILS=ON`).
- **MLIR Python bindings** (`MLIR_ENABLE_BINDINGS_PYTHON=0`). The Python-side IR work uses xDSL and the
  torch-mlir wheel inside model2MLIR's own venv, not these bindings.
- **compiler-rt, libcxx and the other runtimes.** The RISC-V C library and startup files come from the
  chipyard riscv-tools and the Zephyr SDK.

## Using an LLVM you already have

Three overrides, each read from the process environment or a repo-root `.env` (so you set them once per
clone, not once per shell):

```bash
MERLIN_MLIR_INSTALL=/path/to/llvm-install      # the install prefix: lib/, include/, CMake config
MERLIN_CLANG=/path/to/bin/clang-23             # must target riscv64 AND x86-64
MERLIN_MLIR_TRANSLATE=/path/to/bin/mlir-translate
```

Two conditions on a substitute. It must be **LLVM 23** if you build any out-of-tree C++ package against
it — those link static libraries and the ABI is not stable across versions. And its clang must have the
RISCV target compiled in; a host-only clang configures fine and then fails at the first cross-compile.
The single-model Python paths are more forgiving, but there is no version check that will tell you politely.

## Verify it

```bash
# what merlin will actually resolve, which is the only question that matters
.venv/bin/python -c "
from merlin.llvmlower import toolchain as tc
print('clang         ', tc.clang(), tc.clang().is_file())
print('mlir-translate', tc.mlir_translate(), tc.mlir_translate().is_file())"

# assertions really are on
third_party/llvm-install/bin/mlir-opt --version | head -3

# the cross-compile works (the thing a host-only clang fails)
echo 'int f(int x){return x+1;}' > /tmp/t.c
third_party/llvm-install/bin/clang-23 --target=riscv64-unknown-elf \
    -march=rv64gcv_zvl256b -c /tmp/t.c -o /tmp/t.o
third_party/llvm-install/bin/llvm-readelf -h /tmp/t.o | grep Machine   # -> RISC-V
```

Expected from the first command: both paths under `third_party/llvm-install/bin` and both `True`.
`build_tools/scripts/check_repro_env.py` covers this alongside every other capability, and the examples'
own `./run.sh preflight` reports these two tools per stage.

## Troubleshooting

- **`git submodule update --depth 1` fails to find the commit.** Some mirrors refuse a fetch by SHA. Drop
  `--depth 1` for a full clone (much larger), or fetch the SHA explicitly into the submodule.
- **A link step is OOM-killed.** Linking clang and the MLIR tools is the memory peak, and ninja's default
  job count assumes compiles. `-DLLVM_PARALLEL_LINK_JOBS=2`.
- **`clang-23: No such file or directory`.** Either the install is absent (this page) or `MERLIN_CLANG`
  points somewhere stale. Resolution order is `MERLIN_CLANG` → this install → a legacy IREE build →
  `PATH`, so a stale override silently wins over a good install: unset it and re-run the verify snippet.
- **`mlir-translate ... failed` on a multi-hart build only.** The single-hart path never calls it, so this
  is `mlir-translate` missing or version-skewed rather than anything about your model.
- **`WARNING: FileCheck binary not found`.** Expected if you deleted the build tree; RTL checks fall back
  to the Python screen. See [What is deliberately absent](#what-is-deliberately-absent).

## See also

- [Getting started](getting_started.md) — every other external dependency and its variable
- [LLVM integration](llvm_integration.md) — when to *modify* LLVM (out-of-tree extension vs fork), a
  separate question from building this install
- [`examples/`](../../examples/) — Kodiak and gemmelos, the two walkthroughs that need `clang-23` and
  `mlir-translate`
- [Reproducibility](reproducibility.md) — what each result rests on
