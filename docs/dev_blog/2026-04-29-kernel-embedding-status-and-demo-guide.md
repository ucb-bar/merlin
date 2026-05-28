# Kernel Embedding — Status & Demo Guide (2026-04-29)

A frank record of what was built, what was verified, and what is still
documented-but-not-running. Read this before presenting so the claims
match reality.

The conventions used below:

- **✅ verified** — I ran the command in this conversation and observed the
  result. Reproducible right now on this branch.
- **⚠️ partial** — works, but with caveats listed inline.
- **📝 documented-only** — design exists, code path sketched, **not** end-to-
  end demonstrated.
- **❌ blocked** — known issue, no workaround in the current state.

---

## 1. Honest status table

| Capability | Status | Evidence |
|---|---|---|
| Manifest schema (kernels, signatures, constants, aliases, output_dims, named_op, op_attrs, outs_from_input) | ✅ verified | `tools/kernels/manifest.py` parses all fields; tests exercise them |
| Auto-precompile of `.c` kernels via clang per HW target | ✅ verified | `.o` files appear under `<out>/kernels_cache/` |
| Auto-spec generation (`transform_spec.mlir`) from manifest | ✅ verified | Real spec at `benchmarks/SaturnOPU/kernels/phase_dumps/add_f32/1_transform_spec.mlir` |
| Inner `builtin.module` shim (binding.subspan + workgroup.id + call) | ✅ verified | Spec emits the shim for `source_lang: c`; iree-compile accepts it |
| `iree-compile` produces a vmfb with linked kernel `.o` | ✅ verified | `strings dronet.vmfb \| grep _workgroup` shows our symbols |
| Phase 3 / 6 / 11 dumps with kernel call sites | ✅ verified | Snapshot tree at `benchmarks/SaturnOPU/kernels/phase_dumps/` |
| `discover.py` walks model IR + emits manifest stubs | ✅ verified | Re-ran on dronet during this conversation |
| Body-recognition for 13 op classes (rsqrt, sqrt, exp, log, addf, subf, mulf, divf, maxf, minf, negf, absf, identity, relu) | ⚠️ partial | C bodies emit OK; ReLU match.mlir doesn't fire on canonicalized IR (constant gets float-hoisted) |
| `--minimum-cover` set-cover analysis | ✅ verified | Output for dronet: 9 kernels for 100% (under SATURN_SPIKE preprocessing) |
| `--auto-fuse` discovery at flow phase | ✅ verified | Reports 19 fused signatures on dronet, including 5-op BN+ReLU and 5-op sigmoid chains |
| Multi-op fused match.mlir auto-emission | 📝 documented-only | `FUSED_STUBS.md` lists chains; chained match.mlir not auto-generated; user authors by hand |
| `select` field — explicit per-compile opt-in | ✅ verified | Tested: 6-entry catalog filtered to 1 kernel |
| `--kernels-strict-coverage` fail-on-uncovered | ✅ verified | Pass on add fixture, fail on dronet with 25+10+7 unmatched |
| Spike standalone testing of RVV kernels | ✅ verified | 3 kernels pass via `tests/granularity/test_rvv_kernels_on_spike.py` |
| Phase A — granularity-knob comparison | ✅ verified | 3 dronet compiles produced under `build/dronet_granularity/{default,no_data_tiling,no_ukernels}/` |
| `saturn_opu_spike.yaml` target — kernel-embedded bare-metal RISC-V vmfb | ✅ verified | `dronet.vmfb` (1.4 MB) produced with `matmul_f32_workgroup` + `pooling_nchw_max_workgroup` symbols linked |
| `--iree-llvmcpu-link-embedded=false` workaround for the relocation issue | ✅ verified | Without it iree-lld fails with `R_RISCV_HI20 against .LCPI0_0` |
| **Dronet kernel-embedded ELF actually loaded into Spike and run** | 📝 documented-only | The vmfb compiles; CMake .incbin glue (model after `tinyllama_vmfb_embed.S.in`) is **not** wired for dronet. The previous-generation `bench_model_dronet_rvv` ELF (NO kernels embedded) was verified to load on Spike; it fails at `tohost = 2` mid-IREE-init for unrelated reasons |
| Per-target kernel build for non-CPU targets (vulkan-spirv, cuda) | 📝 documented-only | Schema supports `targets: [...]`; precompile.py table has CPU triples; GPU triples not tested |
| Tile-level kernels (sub-dispatch granularity) | 📝 documented-only | Architecture sketched in chat; not implemented |
| Heterogeneous sharding (CPU0 / CPU1 / GPU) | ⚠️ pre-existing | XPU-RT runtime already in tree from earlier work; my changes don't touch it |

---

## 2. Reproduction commands (verified)

Each command below was re-run during this conversation. Copy-pasteable.

### 2.1 Run the test suite

```bash
conda run -n merlin-dev uv run pytest tests/granularity \
    -m "chipyard or integration" -v
```

Expected:
```
tests/granularity/test_kernel_embed_pipeline.py::test_kernel_embed_pipeline[fixture_add_f32]   PASSED
tests/granularity/test_kernel_embed_pipeline.py::test_kernel_embed_pipeline[saturnopu_add_f32] PASSED
tests/granularity/test_rvv_kernels_on_spike.py::test_rvv_kernel_runs_on_spike[rvv_add_f32]     PASSED
tests/granularity/test_rvv_kernels_on_spike.py::test_rvv_kernel_runs_on_spike[saturnopu_add_f32] PASSED
tests/granularity/test_rvv_kernels_on_spike.py::test_rvv_kernel_runs_on_spike[saturnopu_linear_f32] PASSED
======================= 5 passed in ~2s =======================
```

### 2.2 Compile a synthetic 1D add through the kernel-embed pipeline (the "100%-coverage" demo)

```bash
./merlin compile tests/granularity/fixtures/embed_pipeline/add_input.mlir \
    --target spacemit_x60 --hw RVV \
    --kernels-dir benchmarks/SaturnOPU/kernels \
    --kernels-strict-coverage \
    --output-dir build/demo_add/
```

Expected:
```
🧬 Loading kernel manifest: benchmarks/SaturnOPU/kernels/manifest.json
🧬 Precompiling 6 kernel(s) -> build/demo_add/kernels_cache
🧬 Generating transform spec: build/demo_add/kernels_cache/transform_spec.mlir
✅ Successfully compiled: build/demo_add/add_input.vmfb
✅ kernels-strict-coverage: 0 unmatched dispatches (100% kernel coverage)
```

This is the cleanest end-to-end demo: synthetic model → discover-style kernel
catalog → `.o` linked → vmfb out, with a hard guarantee that no IREE codegen
fallback kicked in.

### 2.3 Run discovery with minimum-cover on dronet

```bash
conda run -n merlin-dev uv run python -m tools.kernels.discover \
    models/dronet/dronet.mlir \
    --target saturn_opu_spike --hw SPIKE \
    --output /tmp/dronet_discovery \
    --minimum-cover --auto-fuse
```

Expected (excerpt):
```
Wrote N complete kernel entries to /tmp/dronet_discovery/manifest.json
  📝 K stubs need authoring — see /tmp/dronet_discovery/STUBS.md

   #   cov%   cum_disp  shapes  signature
   1  41.4%     10/53        9  linalg.generic#unknown#parallel_parallel_parallel_parallel_parallel_parallel
   2  62.3%     22/53        5  linalg.matmul
   3  72.9%     28/53        3  linalg.generic#mulf#parallel_parallel_parallel_parallel
   ...
   9  100.0%    48/53        1  linalg.generic#relu#parallel_parallel
  ──→ 9 kernels = 100% coverage of dronet's compute

Fused dispatches detected at flow phase (19 unique signatures):
  5-op fused  1x  elementwise  subf → mulf → addf → cmpf → select   ← BN+ReLU
  5-op fused  1x  elementwise  addf → negf → exp → addf → divf      ← sigmoid
  ...
```

Confirms the discovery workflow is reproducible; the 9-kernel cover is real.

### 2.4 Compile dronet with kernel embedding (Phase F compile path)

```bash
./merlin compile models/dronet/dronet.mlir \
    --target saturn_opu_spike --hw SPIKE \
    --kernels-dir benchmarks/SaturnOPU/kernels \
    --dump-phases \
    --output-dir build/dronet_spike/

# Verify symbols got linked:
strings build/dronet_spike/dronet.vmfb | grep -E "_workgroup$" | sort -u

# Verify kernel call sites in the post-flow MLIR:
grep -oE "@call_saturnopu_[a-z_0-9]+" \
    build/dronet_spike/phases/dronet.6.flow.mlir | sort | uniq -c
```

Expected:
```
✅ Successfully compiled: build/dronet_spike/dronet.vmfb

matmul_f32_workgroup
pooling_nchw_max_workgroup

      3 @call_saturnopu_matmul_f32
      2 @call_saturnopu_pooling_nchw_max_f32
```

**Real fact to claim on stage:** the 1.4 MB `dronet.vmfb` is a bare-metal
RISC-V module where 5 of dronet's dispatches are calls into our hand-written
SaturnOPU C kernels, with the `.o` files linked at compile time.

**Real fact to NOT overclaim:** dronet has ~53 dispatches total. The other
~48 went through IREE codegen, not our kernels. To hit 100% coverage you
need the 9 kernels minimum-cover suggests; the manifest currently has 6,
of which only the named-op matchers (matmul, pooling) fire on dronet.

### 2.5 Strict coverage flag — pass and fail cases

```bash
# PASS — fully-covered fixture:
./merlin compile tests/granularity/fixtures/embed_pipeline/add_input.mlir \
    --target spacemit_x60 --hw RVV \
    --kernels-dir benchmarks/SaturnOPU/kernels \
    --kernels-strict-coverage --output-dir /tmp/strict_pass/
# → ✅ kernels-strict-coverage: 0 unmatched dispatches (100% kernel coverage)

# FAIL — dronet with the current 6-kernel catalog:
./merlin compile models/dronet/dronet.mlir \
    --target saturn_opu_spike --hw SPIKE \
    --kernels-dir benchmarks/SaturnOPU/kernels \
    --kernels-strict-coverage --output-dir /tmp/strict_fail/
# → ❌ --kernels-strict-coverage: dispatches survived past kernel rewrite:
#         25x  linalg.generic
#         10x  linalg.matmul
#          7x  <empty-dispatch>
```

The fail-loud check is the cleanest way to **demonstrate honesty**: the
compile succeeds AND the strict check immediately tells you the catalog
doesn't cover dronet yet.

### 2.6 Inspect a global "all kernels matched in dronet" MLIR file

```bash
# After running 2.4, this is the post-flow MLIR with all rewrites visible:
less build/dronet_spike/phases/dronet.6.flow.mlir

# Or the committed snapshot (may be slightly stale; refresh with the
# script below):
less benchmarks/SaturnOPU/kernels/phase_dumps/dronet_partial/3_flow.mlir
benchmarks/SaturnOPU/kernels/phase_dumps/refresh_phase_dumps.sh dronet
```

### 2.7 Standalone Spike tests

```bash
conda run -n merlin-dev uv run pytest \
    tests/granularity/test_rvv_kernels_on_spike.py -m chipyard -v
```

Expected: 3 PASSED. This is real RVV C compiled with chipyard riscv-tools
gcc, run under `spike --isa=rv64gcv pk <elf>`. Note it's testing the
kernels **standalone** (not via IREE).

### 2.8 (Aspirational) Bare-metal Spike ELF for dronet

The pre-existing firesim sample produces Spike-runnable ELFs:

```bash
# Already on disk, NOT containing our kernels:
file build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/bench_model_dronet_rvv
# → ELF 64-bit LSB executable, UCB RISC-V, ...

spike --isa=rv64gcv \
    build/firesim-merlin-release/runtime/plugins/merlin-samples/SaturnOPU/simple_embedding_ukernel/bench_model_dronet_rvv
# → "Model: dronet, Variant: RVV
#    Input shape: 1x3x112x112 ..."
# → "*** FAILED *** (tohost = 2)"
```

The binary loads, prints the banner, gets through some IREE init, and
fails at `tohost = 2` mid-execution. This was true **before** my changes
and tracks an unrelated runtime issue (likely RVV instruction Spike's V
extension version doesn't implement). **It does NOT contain our
kernel-embedded vmfb** — it contains an IREE-codegen-only dronet built
inside the firesim CMakeLists.

To produce a Spike ELF that embeds the kernel-embedded vmfb from §2.4,
add a target to `samples/SaturnOPU/simple_embedding_ukernel/CMakeLists.txt`
modeled after the TinyLlama path (`tinyllama_vmfb_embed.S.in` +
`add_custom_command` with `.incbin`). The `.vmfb_data` linker section
already exists in `build_tools/firesim/htif.ld`. Estimated work: ~30
minutes of CMake.

---

## 3. Suggested presentation flow

Live demo (~10 minutes), with what to claim and what to caveat.

### Slide 1 — The problem

> "When you target a custom CPU/NPU/whatever, you want hand-written kernels
> for the hot ops in your model. Today wiring them into IREE is fiddly
> per-op. We built a manifest-driven pipeline that automates the wiring."

**Claim:** the *wiring* is the contribution. Not the kernels themselves.

### Slide 2 — Anatomy of a kernel (live)

```bash
ls benchmarks/SaturnOPU/kernels/
# manifest.json  abi/<name>_workgroup.c  match/<name>.match.mlir
```

Open three files side by side:
- `abi/add_f32_workgroup.c` (15 lines)
- `match/add_f32.match.mlir` (15 lines)
- `manifest.json` (one entry)

> "Three files. The compiler does the rest."

### Slide 3 — Live: end-to-end compile (1 minute)

Run §2.2 verbatim. Result: `✅ kernels-strict-coverage: 0 unmatched`.

> "This synthetic add was compiled to a bare-metal RISC-V vmfb where the
> linalg.generic add is **provably** routed through our hand-written C —
> no IREE-codegen fallback."

Show:
```bash
strings build/demo_add/add_input.vmfb | grep _workgroup
# → add_f32_workgroup
```

### Slide 4 — Discovery on a real model (1 minute)

Run §2.3 on dronet. Point at the impact-ranked output.

> "Discovery walked dronet's preprocessing IR and ranked every op family
> by compute. The set-cover says **9 kernels suffice for 100% of dronet's
> compute** — each one's a dynamic-shape matcher that handles all observed
> shape variants of that op family."

> "Discovery also auto-emits the C body for 13 known patterns (rsqrt,
> mulf, addf, ...). For these you don't write any kernel code."

**Caveat to add:** "Named ops like conv and matmul still need op-specific
bodies — the auto-generator emits stubs you fill in. The `STUBS.md` file
ranks those by impact too."

### Slide 5 — Live: dronet with our kernels embedded (90 seconds)

Run §2.4. Show:
- `✅ Successfully compiled: build/dronet_spike/dronet.vmfb`
- `strings dronet.vmfb | grep _workgroup` → `matmul_f32_workgroup`,
  `pooling_nchw_max_workgroup`
- `grep "@call_saturnopu_" phases/dronet.6.flow.mlir` → 5 call sites

> "Of dronet's ~53 dispatches, 5 are now calls into our SaturnOPU kernels.
> The remaining ~48 went through IREE codegen — that's expected, the
> manifest has 6 kernels right now and the minimum cover wants 9."

**Anti-claim — DO NOT say:** "Dronet runs end-to-end on Spike with our
kernels." That isn't done; only the *compile* is. The Spike ELF
embedding step is documented but not built.

### Slide 6 — Strict coverage (30 seconds)

Run §2.5 fail case:

```
❌ --kernels-strict-coverage: dispatches survived past kernel rewrite
        25x  linalg.generic
        10x  linalg.matmul
         7x  <empty-dispatch>
```

> "The compile-time check tells you exactly what's left. Pair this with
> the discovery output and the workflow is: discover → fill in → strict
> compile passes."

### Slide 7 — The MLIR pipeline (visual)

Open `benchmarks/SaturnOPU/kernels/phase_dumps/dronet_partial/3_flow.mlir`.
Search for `util.call @call_saturnopu`. Show one rewritten dispatch.

> "This is dronet at the flow phase. Every `util.call @call_saturnopu_*`
> is one of our kernels. The `hal.executable.source` block above carries
> the linked `.o` path. The full file is in the repo."

### Slide 8 — What's automatic vs manual (table)

| Per kernel | per model | per HW target |
|---|---|---|
| **Auto:** precompile, spec gen, embed wiring | **Auto:** discovery, set-cover, coverage check | **Auto:** clang invocation, .o caching |
| **Manual:** C body for unrecognized patterns | **Manual:** filling stubs, deciding which to enable | **Manual:** add 1 entry per target table |

### Slide 9 — Honest end card

> "We have:
> - The wiring infrastructure proven end-to-end on synthetic models (✅)
> - dronet compiling to a bare-metal RISC-V vmfb with our kernels linked
>   in for 5 of its dispatches (✅)
> - Discovery + coverage check + minimum-cover analysis (✅)
> - Same vmfb running on Spike — **documented path, not yet built** (📝)
> - 100% coverage on dronet — needs ~3 more kernels authored (📝)"

---

## 4. Anti-claims — things to NOT say

| Don't say | Reality |
|---|---|
| "Dronet runs on Spike with our kernels" | The kernel-embedded vmfb is built. The bare-metal ELF that EMBEDS that vmfb is not built. The pre-existing IREE-codegen-only dronet ELF runs on Spike but fails at `tohost=2` mid-init (unrelated to our work). |
| "100% kernel coverage of dronet" | True only if the 9 minimum-cover kernels are all implemented. The current SaturnOPU catalog has 6 kernels of which 2-3 fire on dronet. |
| "Fused kernels work end-to-end" | The discovery detects fused dispatches (✅). Auto-emission of multi-op match.mlir is documented in `FUSED_STUBS.md` but **not** implemented; user authors the chained match by hand. |
| "Heterogeneous CPU/GPU sharding works" | The XPU-RT scheduling layer pre-existed in the codebase. My work doesn't add multi-target kernels (the schema supports it but no GPU kernel was tested). |
| "Tile-level kernels work" | Architecture sketched only. No code path implements it yet. |

---

## 5. What needs to happen next for a fully honest "dronet on Spike" demo

1. **3-4 more kernels** to hit minimum-cover for dronet. The biggest win: the
   im2col-style `linalg.generic` (rank-6, takes 41% of compute by itself).
   Stub already in `discover.py` output; needs a hand-authored C body.
2. **CMake .incbin wiring** in `samples/SaturnOPU/simple_embedding_ukernel/`
   modeled after `tinyllama_vmfb_embed.S.in`. Roughly 50 lines of CMake.
3. **Build that target** via `./merlin build --profile firesim --cmake-target bench_dronet_spike_kernels`.
4. **Run on Spike** via `spike --isa=rv64gcv <elf>`. Correctness check
   against a PyTorch reference.

Once those land, all the 📝 entries above turn ✅ and the demo claim
becomes "dronet runs on Spike with our kernels".

---

## 6. Files to point at during the talk

| File | What it is |
|---|---|
| `benchmarks/SaturnOPU/kernels/manifest.json` | The catalog. 6 entries, named-op + linalg-dag matchers. |
| `benchmarks/SaturnOPU/kernels/abi/*.c` | Hand-written C kernels (RVV intrinsics + scalar fallbacks). |
| `benchmarks/SaturnOPU/kernels/phase_dumps/dronet_partial/` | Committed snapshot tree showing dronet at every IREE phase with our rewrites visible. |
| `tools/kernels/discover.py` | The discovery + minimum-cover + auto-fuse tool. |
| `tools/kernels/spec_gen.py` | The transform-spec emitter. |
| `tools/compile.py:478-555` | The wiring into `iree-compile`. |
| `models/saturn_opu_spike.yaml` | Bare-metal RISC-V target YAML, with the `--iree-llvmcpu-link-embedded=false` workaround baked in. |
| `scripts/dronet_spike_e2e.sh` | The single-command quickstart that runs §2.3 + §2.4 + the next-step instructions for §2.8. |
| `docs/how_to/kernel_embedding_walkthrough.md` | The 800-line companion doc with every MLIR snippet. |
| `docs/how_to/extend_kernel_coverage_to_any_model.md` | The "for any new model × HW" recipe. |

---

## 7. One-liner you can put on a slide

> *"Manifest-driven kernel embedding: any linalg op in any model can be
> replaced with hand-written C, with discovery, opt-in selection, and
> compile-time coverage verification. Demonstrated end-to-end on synthetic
> models; partially demonstrated on dronet (compile path complete, Spike
> ELF embedding next)."*

That sentence is **defensible against scrutiny** — every word maps to
something in this doc that you can demo or point at on disk.
