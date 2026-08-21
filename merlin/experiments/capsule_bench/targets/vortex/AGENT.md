# AGENT.md — merlin/experiments/capsule_bench/targets/vortex

The **agentic A/B capsule-bench harness** for the Vortex RISC-V SIMT GPGPU compiler: the Vortex sibling
of `gemmini_capsule_bench_v0`. Same contract, same arms (baseline / merlin-assisted), same redacted QA
loop, same integrity boundary; only the target, the input dialect, and the oracle ladder differ.

- **Tracked source**: `task/`, `target_experiment.yaml`, `contracts/`, `input_bundles/`, `scripts/`.
- **Generated output**: runs route to `out/runs/vortex/capsule-bench/`.
- **Status**: BRING-UP. Written and validated **on real hardware, both oracle tiers**: the curated
  harness (`contracts/harness_curated/vortex-baremetal/`), the ISA spec sheet
  (`contracts/hwbringup_vortex_v0/VORTEX_ISA_SPEC.md`, encodings cross-checked), the capsule corpus
  (`merlin/contract/capsules/vortex/` — **38 capsules** across `isa/`, `layers/`, `model_slices/`,
  `hidden/`), the oracle adapters (`targetgen/vortex_oracle.py`),
  `capability_manifests.vortex_manifest()`, `task/TASK_full.md` and `target_experiment.yaml`.
  the three input bundles (`input_bundles/`, generated), and the `simt_coverage` gate
  (`targetgen/vortex_coverage.py`). **Still to build**: the arm-3 RTL-check compiler and the QA gate —
  see "What still needs building".
- **The frozen machine**: 2 clusters x 2 cores/cluster (**4 cores total**) x 8 warps x 8 threads =
  **256 threads**, L2 enabled, **XLEN=64** (`rv64imafd` / `lp64d`). Single source of truth is
  `target_experiment.yaml` -> `hardware_spec.config.macro`; `vortex_oracle.FROZEN_GEOMETRY` asserts it
  against what the device reports and raises rather than grading a mismatched machine. Two traps:
  `NUM_CORES` is **per cluster** while `VX_CAPS_NUM_CORES` reports the total, and XLEN=64 is Vortex's
  own build default (`build/config.mk`) — it is what the installed simx/rtlsim drivers and
  `libvortex2.a` were built with, so the kernel ABI is rv64, not rv32.

## Why this target is shaped differently from gemmini

Gemmini is a fixed-function systolic MXU driven by a RoCC command stream, so its bench is built around a
tensor-resident interface dialect (`merlin_iface`: `pack/matmul/commit/evict`), a `command_buffer.json`,
and RoCC `.insn` emission. Vortex is a **programmable SIMT core**. The deltas that follow from that:

| Axis | gemmini | vortex |
|---|---|---|
| Compute-unit kind | `systolic` | `simt` (`families.py` already has the profile) |
| Input the agent compiles | `merlin_iface` interface dialect | **linalg-on-tensors + arith** (stock upstream, as torch export emits) |
| Target dialect shape | tensor-resident pack/matmul/commit/evict | **SIMT kernel grid**: warps/threads, divergence, barriers, memory spaces — agent-designed |
| 4th artifact | RoCC `.insn` module + command buffer | LLVM-dialect module defining `merlin_kernel_body` + `merlin.grid` / `merlin.arg_table` |
| `emit_command_buffer` | required | **N/A** — no command ISA |
| Extra entrypoint | — | **`optimize_interface`** (global opt, still linalg/arith) |
| Coverage gate | RoCC instruction classes, all required | **CUSTOM0 SIMT classes** in three forms: all-required, `any_of` groups (divergence, where the ISA permits either split/join or predication), and non-gating `advisory` (shared memory) |
| Input control flow | straight-line only | a `linalg.generic` body may contain an **unbounded `scf.while`** (the divergence capsules) |
| Oracle ladder | spike L2 + verilator L3 (chipyard) | **simx L2** (default, all capsules) + **rtlsim L3** (declared subset) |
| Target facts | derived via mlc RTL discovery | **mlc RTL discovery** over the Vortex HW-dialect import + an experiment-authored spec sheet |
| Vendor ISA text granted | `isa_include/` (`gemmini.h`) | **none** — no Vortex software at all (see below) |
| Numerics | `exact_int` only | `exact_int` **and** `tolerance_float` |

Two facts make this cheaper than it looks:

1. **No forked toolchain.** Vortex's SIMT ops are plain `.insn r CUSTOM0` instructions — two families
   keyed by `funct7` (0 = tmc/wspawn/split/join/barrier/pred/wsync, 1 = the vote/shfl collectives; see
   the spec sheet §3.1-3.2) — and its ids are CSR reads, so stock LLVM +
   `llvm.inline_asm` on an upstream RISC-V target is sufficient — the same fork-free principle
   `families.py` records for the `simt` kind, and the mechanism `llvmlower/custom_isa.py` implements.
   (Those encodings are stated in the experiment-authored spec sheet; the vendor header they can also be
   read from is denied — see below.) These are R-type encodings and so are **XLEN-independent**; the
   verification below was run on rv32, while the graded build is rv64 (frozen `XLEN=64`) — the words are
   the same either way. **VERIFIED** on stock Ubuntu clang 18.1.3,
   `--target=riscv32 -march=rv32imaf -mabi=ilp32f`: all six CUSTOM0 forms assemble to the correct
   encodings (`tmc` 0x0005000b, `wspawn` 0x00b5100b, `split` 0x0005250b, `join` 0x0005300b,
   `barrier` 0x00b5400b, `pred` 0x00b5500b — opcode 0x0b, funct3 0-5 as specified), and `csrr a0, 0xcc0`
   round-trips. No fork required.

   **Caveat for the coverage gate:** `llvm-objdump` prints `<unknown>` for CUSTOM0 — stock LLVM's
   *assembler* knows `.insn`, its *disassembler* has no CUSTOM0 decode table. So the SIMT coverage
   checker decodes raw instruction words itself (`word & 0x7f == 0x0b`, funct3 = bits [14:12],
   **funct7 = bits [31:25]** — both are needed, see item 2 below); it cannot grep objdump mnemonics.
2. **Float grading needs no schema change.** `contract/schemas/capsule.schema.json` already accepts
   `numeric_policy.compare: tolerance_float` with `rtol`/`atol`.

## Why no Vortex software is granted (divergence from gemmini)

Gemmini's bring-up contract (`contracts/hwbringup_gemmini_v0`) grants RTL + an architecture README +
`isa_include/` (`gemmini.h`, `gemmini_params.h`), and its ablation axis is the *worked example kernel*
(`hwbringup_nokernel_v0` removes it). Copying that grant to Vortex would be a mistake:

- **Gemmini is fixed-function.** Knowing the RoCC encodings tells you nothing about tiling, dataflow, or
  scratchpad management — the compiler work lives entirely above the ISA, so the header is cheap.
- **Vortex is programmable, and ships a mature SDK.** `vx_spawn.h` is an NDRange launch runtime: it *is*
  the iteration-space→warp/thread mapping, which is the bulk of what this benchmark measures.
  `vx_intrinsics.h` pre-wraps every CUSTOM0 op as ready-to-call inline asm. And `tools/llvm-vortex`,
  `tools/pocl`, `tools/chipstar` are entire alternative compilers.

A target under bring-up has none of that. Granting it would turn "bring up a compiler from the hardware"
into "use the vendor SDK" and measure the wrong thing. So **all Vortex software is denied to every arm**
(`answer_surfaces.denied_target_software`), and the irreducible ISA facts — you cannot write a code
generator for an ISA you have no statement of — are supplied by a spec sheet **we author**:
`contracts/hwbringup_vortex_v0/VORTEX_ISA_SPEC.md`. Prose and encoding tables, not compilable code.

Note the spec sheet is *always* granted — it is the shared floor, not a rung. What varies is Merlin
tooling (arms 1 vs 2) and, for arm 3, additional RTL-derived facts on top.

## Arms (two info tiers, three arms)

Arms 1 and 2 share one info tier and differ **only** in the Merlin tooling granted. Arm 3 is a *second*
info tier: it adds the RTL-facts pin and the RTL generators, so it can derive facts itself in addition to
receiving the harness's advisory checks. Same allow/deny structure as gemmini otherwise — same
oracle-route exclusions (`generate/runtime_adapter.py`, `xdsl_dialects/lowering/`), same integrity scan
on the shipped package. `tests/targetgen/test_vortex_bundles.py` pins arms 1-2 to an identical hardware
grant and arm 3's extra to exactly the facts pin.

1. **`raw_baseline_v0`** — no Merlin tooling. The control.
2. **`merlin_assisted_v0`** — read access to Merlin's *authoring* aids: `targetgen/synthesize/`,
   `targetgen/generate/` (minus `runtime_adapter.py`), `xdsl_dialects/` (minus `lowering/`),
   `targetgen/contract/interface_emit.py`, `targetgen/rtl_backend.py`, and the CCA spine
   (`kernels/{cca,cca_compare,cca_contract,action_catalog,microkernel}.py`).
3. **`merlin_assisted_rtlchecks_v0`** — arm 2 plus the per-round RTL-derived advisory checks.

### How the RTL checks work (they are OURS to write, not the agent's)

The agent never authors these — it only *receives* their results each round, alongside the redacted QA
verdict, as `expected` / `got` / `fix_hint`. They are **advisory and never gate pass/fail**. The
generation pipeline is entirely harness-side, mirroring gemmini's:

`rtl/circt_introspect.py` (RTL facts) + the capsule's **declared** shape → `rtl_check_compiler.py`
(bakes concrete RTL-grounded literals into FileCheck directives at generation time) →
`rtl_check_runner.py` (runs FileCheck over the agent's emitted artifacts) → the QA loop.

**The non-overfit invariant is load-bearing**: every `expected` literal must derive ONLY from (i) RTL
facts, (ii) the capsule's declared shapes/modes, and (iii) ISA structural rules — never from a golden,
never a per-capsule magic number. Where a value cannot be derived generally, the check degrades to a
lower bound or is omitted, honestly. Violate this and the checks become a backdoor oracle, which would
invalidate the arm's comparability.

Vortex check candidates (the analog of gemmini's `MVOUT_COUNT` / `ILLEGAL_FUNCT_COUNT` set), all
derivable from the decoder's legal CUSTOM0 set + declared shape + the frozen geometry:

| Check | Derived from | Catches |
|---|---|---|
| `ILLEGAL_CUSTOM0_COUNT 0` | decoder's legal funct3 set | an encoding the hardware rejects |
| `TMC_PRESENT yes` | ISA rule | **scalar collapse** — the kernel never enables threads |
| `WSPAWN_PRESENT yes` (multi-warp capsules) | declared shape vs `NUM_WARPS` | work never distributed across warps |
| `BARRIER_COUNT ≥ n` | declared cross-warp dependences | missing sync before a cross-warp read |
| `SPLIT_JOIN_BALANCED` | IPDOM stack discipline | unbalanced reconvergence |
| grid coverage ≥ ⌈elems / (warps·threads)⌉ | declared shape + geometry | mapping doesn't cover the output |
| shared-memory footprint < RTL capacity | module params | resident footprint exceeds hardware |

Note `TMC_PRESENT` / `WSPAWN_PRESENT` overlap the **SIMT coverage gate**, which *does* gate pass/fail.
Keep them distinct: coverage is a hard gate on the built ELF; the same fact surfaced here is early
advisory feedback so the agent fixes it before paying for a simulator run.

## Where the PyTorch front end lives (and does not)

Torch → linalg/arith is **corpus generation, not the agent's job** — exactly as gemmini's model-slice
capsules are minted by `targetgen/model_slice_export.py` and the agent never sees torch. The Vortex
corpus generator runs model2MLIR (`frontends/adapters/m2m.py`: torch export → linalg-on-tensors), slices
it, freezes the linalg as `capsule.interface.mlir`, computes the golden once, and withholds it. xDSL owns
everything from linalg onward; it is not a PyTorch front end and is not used as one here.

## What still needs building

Roughly in dependency order. Items struck through are done; see each for what was actually verified.

1. ~~**Capsule corpus**~~ — **DONE**, `merlin/contract/capsules/vortex/` with
   `generate_vortex_corpus.py` (deterministic; regeneration is byte-identical). **38 capsules**, all
   three families `task/TASK_full.md` scopes plus the hidden set:

   | family | capsules | covers |
   |---|---|---|
   | `isa/` | V0-V10 (11) | elementwise, K-reduction matmul, rank-reducing reduction, transposed contraction, edge shapes (63 elements; all-prime dims), data-dependent select, **unbounded data-dependent loop** |
   | `layers/` | L0-L7 (8) | quantized linear, +relu, +requant->i8, conv2d — each in i8 and f32 — plus a **high-reuse tile** capsule |
   | `model_slices/` | C0-C7 (8) | MLP fc1, relu+fc2, attention Q/K/V projections, QK^T, PV, **softmax (f32)** |
   | `hidden/` | H0-H10 (11) | every public family re-shaped: non-square, prime, and non-power-of-two |

   Interfaces are stock linalg-on-tensors + arith and all parse + verify under xDSL; goldens are
   binary64 (or exact integer) and **independently recomputable from the capsule alone** — the test
   reimplements every op family rather than calling the generator, so it checks the arithmetic and not
   just determinism. Float capsules carry a **derived** `atol`, integer capsules grade bit-exactly.

   **Every golden is verified against the capsule's own IR, executed.**
   `test_vortex_corpus_interpreted.py` (marked `slow`, ~85 s) runs each interface MLIR through the
   **xDSL interpreter** and compares: **37/38 match, zero mismatches** — every integer capsule
   bit-exact, every float capsule inside its derived `atol` with 10-100x margin. This closes a blind
   spot the recompute test cannot: that test reimplements the semantics the author *believed* the IR
   had, so an emitter and a golden wrong the same way would agree. Interpreting substitutes xDSL's
   semantics for ours, which is what actually validates the conv access maps, the transposed `W[j,p]`
   contraction, the softmax chain, the requant clamps, and the `scf.while` trip counts. Verified to
   bite: a single +1 on one of C5's 256 values fails it. The 38th (`L6_conv2d_f32`) uses named
   `linalg.conv_2d_nhwc_hwcf`, which xDSL cannot interpret; a separate test re-runs it as the
   equivalent explicit `linalg.generic` and matches, leaving only "the named op obeys its own
   definition" assumed. A few standard ops xDSL 0.68 lacks (`arith.extsi/negf/maxsi/minsi/trunci/
   select/divf`, `scf.while`, scalar `linalg.fill`) are shimmed in the test — all unambiguous, and none
   of them an op under scrutiny.

   Validated on real hardware, **both tiers** (simx L2 and rtlsim L3), with a stock-clang kernel body:
   V0 PASS at 1086 cycles (simx) / 3069 (rtlsim), V2 bit-exact at 2529 / 5114. The hidden set caught a
   shape-specialised kernel 64/64 and passed a correctly-shaped one bit-exactly. (Those four pilot
   capsules and the original four hidden ones are byte-identical to their pre-widening form, so that
   validation still stands.)

   **Method: hand-authored, matching gemmini.** No torch export runs — the model slices are linalg
   *shaped after* the slices they name, at gemmini's own hyperparameters (SEQ=16, D_MODEL=64,
   D_HEAD=16) so the two corpora are comparable. Divergence from gemmini: its C-family declares
   `source_role: pytorch_model_slice`, which reads as provenance it does not have
   (`model_slice_export.py` f-string-builds the MLIR from M/K/N and its own docstring notes torch is
   not installed); these declare `handauthored_compiler_test` plus a `source_reference` naming the
   slice — same information, no false claim.

   **Divergence IS forced, by V10/H9.** The mechanism matters, so it is worth stating precisely why
   the obvious approaches do not work. `arith.select` (V9) is already branchless. A *bounded* loop is
   no better: it can always be rewritten as max-trip-count straight-line code with selects, needing no
   divergence handling at all. What forces the issue is a loop with **no compile-time bound** — V10/H9
   carry a Collatz step count as an `scf.while` inside a `linalg.generic` body, so a real
   data-dependent exit must survive into the kernel, and on a warp with one PC the lanes have to be
   masked apart. Measured on the generated data: **8/8 warp-sized groups have disagreeing trip
   counts**, 31-35 distinct step counts, ~2300 total iterations (so the cost is trivial).

   Gating on it needed a new coverage form. `simt_classes` is conjunctive, but the ISA spec explicitly
   declines to prefer split/join over predication (§7), so requiring `SPLIT`/`JOIN` would gate on a
   mapping decision. Hence **`simt_classes_any_of`**: a list of groups of which at least one must
   appear in full — here `[[SPLIT, JOIN], [PRED]]`. A test asserts each group is disjoint from the
   always-required set, because `TMC` is required of every capsule and including it would make the
   disjunction vacuous.

   **`shared_mem` cannot be forced, and is advisory by construction.** This is a property of the
   machine, not a gap in the corpus: global memory is always semantically sufficient, so a correct
   kernel may ignore the scratchpad and no input can require otherwise. L7/H10 instead make it *pay* —
   64x reuse of both operands, sized so both 4 KB tiles fit the 16 KB per-core scratchpad
   (`LMEM_LOG_SIZE=14`) at once — and declare `SMEM_LD`/`SMEM_ST`/`BARRIER` under
   **`simt_classes_advisory`**, which never gates. The signal is the cycle count; the classes are there
   so the rtlchecks arm can report their absence as a missed optimisation. A test asserts no capsule
   ever gates on a shared-memory class.

2. ~~**CUSTOM0 coverage checker**~~ — **DONE**, `targetgen/vortex_coverage.py`, wired as the
   `simt_coverage` trace gate (`families.py` `simt` profile) and dispatched by `capsule_runner` **before**
   the oracle tiers, so a scalar-collapsed kernel is rejected without paying for rtlsim. Writes
   `simt_coverage.json` beside the other generated artifacts and raises `CertFailure("coverage_check")`
   on violation. Honors all three coverage forms (conjunctive / `any_of` / non-gating `advisory`).

   **The required class is `CTA_CSR`, NOT `TMC`/`WSPAWN` — the first version of this gate was wrong.**
   Under KMU dispatch the *hardware* launches every `(block, thread)` coordinate and the runner-owned
   startup sets the thread mask, so a correct kernel body emits **no CUSTOM0 op at all**. Requiring
   `TMC`/`WSPAWN` made the gate unsatisfiable by a working kernel — caught by the first real end-to-end
   run, whose object decoded to exactly `{CTA_CSR: 3, GMEM_LD: 8, GMEM_ST: 1}`. What a coordinate-blind
   scalar loop cannot do is read `thread_id`/`block_id`, so that is the anti-scalar-collapse signal.
   `TMC`/`WSPAWN` moved to `simt_classes_advisory`: they appear only if a backend sub-distributes work
   *within* a launched coordinate, which is a legal mapping choice whose cost shows in cycles.
   `tests/targetgen/test_vortex_coverage.py` now carries a `coordinate_only` specimen (CSR reads +
   memory, zero CUSTOM0) that must PASS, alongside the scalar-collapse specimen that must fail — the
   earlier specimens all happened to contain CUSTOM0, which is why they could not catch this.

   **It scans the agent's OBJECT, never the linked ELF — this is load-bearing.** Measured on the real
   staged `vx_start_min.o`: the harness startup alone supplies **TMC** (plus `WSYNC` and a `cta_entry`
   CSR read), because it performs KMU dispatch. TMC *is* the anti-scalar-collapse class, so gating on the
   linked image would let a genuinely single-threaded kernel pass that half of the gate for free. The
   compile therefore goes through a new `vortex_oracle.compile_object`, factored out of `build_image` so
   the gated object and the graded image cannot drift. A test asserts the harness hazard against that
   object, so the trap is demonstrated rather than described.

   **Decoding `funct3` alone would have been wrong.** Building this surfaced a real gap in the spec sheet:
   CUSTOM0 has **two** dimensions. `funct7=0` is warp/thread control (tmc, wspawn, split, join, barrier,
   pred, — , **wsync**) and `funct7=1` is a whole cooperative-thread family
   (**vote_all/any/uni/ballot, shfl_up/down/bfly/idx**). `funct3=0` means `tmc` under one and `vote_all`
   under the other. The spec sheet documented only `funct7=0`, `funct3` 0-5 — so it was withholding
   `wsync` and eight collectives, including `vote_any` (the natural primitive for the unbounded divergent
   loop V10/H9 force) and `shfl_bfly` (the natural warp reduction, which most of the corpus wants). Since
   cycles are the comparison metric, withholding them would have distorted the measurement. §3.1-3.2 of
   the spec sheet now carry the full map. Verified end-to-end: all 15 documented ops assemble with stock
   clang and decode to the right class, and an unassigned encoding (`funct3=6` at `funct7=0`) is reported
   as illegal.

   Guarded by `tests/targetgen/test_vortex_coverage.py` (13 tests) against instructions assembled by the
   real toolchain — not hand-built words, so a decoder mistake cannot be mirrored by the same mistake in
   the test. Covers: scalar collapse fails; split/join **and** predication each satisfy V10's `any_of`;
   a kernel with neither fails; illegal encodings fail; shared-memory classes never gate; and requiring
   an undecidable class is itself reported as a violation rather than failing the kernel.

   **`SMEM_LD`/`SMEM_ST` are undecidable and must stay advisory.** Shared and global accesses use the
   *same* load/store instructions — only the address differs, and the scratchpad base arrives at run time
   in CSR `0xCDF`. So their absence from a report means "not determined", never "not present".
3. ~~**Curated bare-metal harness**~~ — **DONE**, `contracts/harness_curated/vortex-baremetal/`.
   Much cheaper than estimated: work distribution is *hardware* (the KMU launches every
   `(block, thread)` coordinate; identity is CTA CSRs), so the curated lib is Vortex's `libvortex2.a`
   with exactly one object deleted (`vx_spawn`). The one genuine Vortex-toolchain dependency — the
   `annotate("vortex.kernel")` entry stub the KMU launches from — is isolated in a prebuilt
   `vx_entry.o`, letting the agent's compiler stay on stock LLVM. Validated on **both** tiers: harness
   entry + curated lib + a stock-clang body with zero Vortex headers -> `PASSED`.

   **Do not link the archive's `vx_start`.** `libvortex*.a` is built once for all apps, so its startup is
   compiled with every feature enabled; that made all 256 harts run `__init_tls` / `__libc_init_array` at
   CTA entry across 4 non-coherent L1s, leaving output buffers unwritten (`0xBAADF00D`). It is invisible
   on simx *and* on a 1-core rtlsim build, and fails 64/64 on the 4-core one — which is what made it
   expensive to find. The harness now stages a KMU-only `vx_start_min.o`, links it **ahead of** the
   archive, and runs Vortex's `kernel_startup.sh` at link time to fail loudly if the image actually needs
   gp/TLS/init_array. (This was previously logged here as a cycle-count "rough edge"; it was a
   correctness bug, and ~80% of every cycle number measured before the fix was startup — V0 on simx went
   11367 -> 1086.) Only `xlen=64` is staged and validated, which is also the frozen ABI.

3b. ~~**`contracts/hwbringup_vortex_v0/VORTEX_ISA_SPEC.md`**~~ — **DONE**. CUSTOM0 funct3 table with
   operand semantics, full CSR map, the split/join reconvergence contract, memory/ordering model,
   kernel entry convention, frozen geometry, and an explicit "deliberately not specified" section
   (mapping, tiling, layout, barrier placement) to keep it hint-free. Every documented `.insn` form was
   assembled with **stock clang** and disassembled under the Vortex fork's decoder: all eight decode to
   the intended instruction (`vx_tmc`/`wspawn`/`split`/`split_n`/`join`/`bar`/`pred`/`pred_n`), so the
   table is byte-accurate and stock LLVM can express the whole extension. Still wants a second reader
   for neutrality.
4. ~~**Oracle adapters**~~ — **DONE**, `targetgen/vortex_oracle.py` + a `sim_via: vortex` branch in
   `capsule_runner.oracle_adapters` returning `{L2: simx, L3: rtlsim}`. Rather than shelling out to
   `ci/blackbox.sh` (which rebuilds Vortex's own tests), it drives the real pipeline directly:
   llvm-dialect MLIR -> `mlir-translate` -> **stock clang** -> link against the curated harness ->
   `vxbin.py` -> the generic host driver, parsing `OUT`/`METRIC`/`DONE` plus the
   `instrs=/cycles=/IPC=` line. Unavailability raises `VortexUnavailable` (never a vacuous pass).
   Validated end-to-end on **both** tiers: `decode_out` parses real device output bit-exactly against
   the Python golden. The launch plan is now derived inside the adapter — operands and their seeds from
   the capsule (`plan_from_capsule`), grid from the package's own `merlin.grid` module attribute
   (`grid_from_module`, which *raises* rather than defaulting, so mapping stays the compiler's decision).
   A pre-built plan dict may still be passed verbatim, which is what the corpus generator does when it
   computes goldens.
5. ~~**Capability manifest**~~ — **DONE**, `capability_manifests.vortex_manifest()` (`kind: simt`,
   `endpoint_kind: inline_asm_insn`), registered in `MANIFESTS` and schema-validating. Provenance is
   stamped to what was actually verified (CUSTOM0 funct3 from `vx_intrinsics.h` cross-checked against
   the decoder's `comb.icmp` fan-out in the HW-dialect import; CTA CSRs from the generated
   `VX_types.vh`; KMU dispatch from `vx_start.S`). Carries `must_insert_reconvergence` as a compiler
   obligation.

5b. ~~**Target contract (`merlin/targets/vortex/`)**~~ — **DONE**. Item 5 registered the manifest in
   `MANIFESTS`, but nothing had ever *materialized* it, so `merlin/targets/vortex/` did not exist and
   `load_capability_manifest("vortex")` raised `FileNotFoundError` on
   `out/artifacts/targets/vortex/contracts/target_contract.yaml` (no reference tree -> it resolved as a
   *generated* target). Three consequences, the third of which is why this went unnoticed:

   | consumer | without the contract |
   |---|---|
   | `generate_prompt.render_prompt` | raises in `prompt_slots` — **no arm's prompt can be rendered** |
   | `run_baseline_qa_loop.py:113` | `load_capability_manifest(C.TARGET)` — the QA loop cannot start |
   | `capsule_runner._endpoint_of` | catches it and returns `(None, None)`. The oracle path kept working because the descriptor's `sim_via: vortex` self-routes, so **the first end-to-end run passed while the contract was missing** |

   Now `contracts/target_contract.yaml`, **generated** from `vortex_manifest()` (`cm.write("vortex")`) —
   the Python function keeps the per-field rationale YAML emission drops, and a test asserts regeneration
   is byte-identical, so a hand-edit is caught rather than silently reverted. Two fields had to be added
   to the manifest because the family defaults cannot supply them:

   - **`runner.tier_sim: {L2: simx, L3: rtlsim}`** — NOT optional. `runner_config` derives the oracle
     tier LOOP from its keys, so an omitted map means `oracle_tiers == ()` and no tier ever grades; with
     `not_run_is_not_pass` that surfaces as `incomplete`, which reads as a harness fault rather than a
     missing declaration. `rtl_tiers` is left to the family default `("L3",)` — simx is cycle-approximate.
   - **`kernel_abi.symbol: merlin_kernel_body`** — `generate_prompt` guessed `f"{target}_kernel"`, so
     every rendered prompt named `vortex_kernel`, a symbol that does not link. The lookup is additive
     (gemmini declares no `kernel_abi` and still gets `gemmini_kernel`); a test pins the contract's
     `signature` against the header the harness actually compiles, since the contract deliberately
     declares no path to it (that is experiment scaffolding the descriptor already names).

   Everything the `simt` family profile supplies is deliberately NOT restated: `encoding_required=False`
   (the CUSTOM0 map is fixed by the ISA — nothing to derive), the `simt_coverage` trace gate, and the
   linalg-input entrypoint set `parse -> optimize_interface -> lower_interface_to_target ->
   emit_target_artifact`, which drops `emit_command_buffer` — matching what the runner enforces and what
   `task/TASK_full.md` asks for. Also omitted on purpose: **no `dialect_plan.yaml`** (the target dialect
   is the agent's design decision, gemmini's stance — `dialect_plan_from_manifest` would have derived
   `vortex.matmul`/`vortex.elementwise` and pre-baked the vocabulary), and **no `rtl_sim_config`**
   analog of `GemminiRocketConfig` (Vortex's geometry is a build parameter, so "which machine" is an
   experiment decision; the descriptor's `hardware_spec.config.macro` declares itself its single source
   of truth). A test pins the contract's `capabilities.simt` geometry against that frozen block, because
   nothing downstream would fail on drift — cycle counts and `pct_fp_peak` would just quietly stop being
   comparable to the declared peak.

   `tests/targetgen/test_vortex_target_contract.py` (10 tests). Both drift guards verified to bite: a
   one-field mutation fails the byte-identical and geometry tests, and regeneration restores it exactly.

   **Two prompt-generator bugs found here are NOT fixed** (they do not block this experiment, because
   the Vortex arms use the hand-authored `task/TASK_full.md`, which is correct on both counts — but they
   will bite the next target that regenerates):
   - `_TEMPLATE` hardcodes **four** command lines including `emit_command_buffer`, then says "declare
     these four commands ... exactly as the runner expects". For any target whose family entrypoints
     differ (i.e. any non-command-driven target) that instructs the agent to build the wrong package.
     It should render from `manifest.entrypoints`.
   - `prompt_slots` computes `hwbringup_set`, `isa_headers` and `prior_backend_deny` and the template
     **never renders them**. So a generated prompt never tells the agent where the ISA spec sheet is —
     for Vortex it would say "Legal opcodes: unavailable" (the SIMT fact bundle is 0/4 grounded, since
     `_simt_fact_bundle` is muon-specific) with no pointer to `VORTEX_ISA_SPEC.md`. The most dangerous
     shape of prompt bug: it looks complete.
6. ~~**Input bundles**~~ — **DONE**, `input_bundles/{raw_baseline,merlin_assisted,merlin_assisted_rtlchecks}_hwbringup_v0/`,
   **generated** rather than hand-authored:

   ```
   python -m merlin.targetgen.generate_bundles \
       --descriptor merlin/experiments/capsule_bench/targets/vortex/target_experiment.yaml \
       --arms raw_baseline,merlin_assisted,merlin_rtlchecks
   ```

   `generate_bundles` was already target-agnostic ("for ANY target from its `target_experiment.yaml`"),
   so this needed three additions rather than new YAML: the descriptor now declares `hwbringup_set`
   (without which **no arm** would have been granted the ISA spec sheet — every agent would have been
   asked to emit code for an ISA it had no statement of); `TargetExperiment` carries
   `denied_target_software` and `_shared_deny` emits it, so all 11 Vortex software paths are denied in
   every arm; and `--arms` selects a subset of the fixed 4-arm ladder, since this experiment has no use
   for the `cpp_merlininfra` rung and a materialized bundle nobody launches reads as part of the design.
   Also made the `agent_selfcheck.py` grant conditional on the file existing — vortex has no `scripts/`
   yet, and the grant was promising a tool that is not there. `test_generate_bundles` still passes, so
   gemmini's generated allow/deny sets are unchanged.

   Guarded by `tests/targetgen/test_vortex_bundles.py` (13 tests): the deny surface covers
   `vx_spawn.h`/`vx_intrinsics.h`/the fork/PoCL by name, no arm is granted a vendor header, the hidden
   corpus is denied everywhere, all three public families are granted, oracle routes are denied in the
   assisted arms, and no grant is dangling.

   **RESOLVED — arm 3 IS an info-tier change, deliberately.** It inherits the ladder's CIRCT rung, so
   beyond arm 2's tooling it also grants `merlin/targets/vortex/contracts/rtl_facts/` (RTL-derived
   facts) and `targetgen/rtl/` (the generators), letting the agent derive facts itself *in addition to*
   receiving the harness's per-round advisory checks. So arms 1-2 share one info tier and arm 3 adds a
   second; the earlier "one info tier, three arms" phrasing described arms 1-2 only. A test pins arm 3's
   extra grant to exactly the facts pin, so the boundary cannot widen unnoticed.

6c. ~~**Target contract**~~ — **DONE**, `merlin/targets/vortex/contracts/target_contract.yaml`, the file
   that makes `vortex` resolve through the capability spine. It was missing, and the failure mode was
   the interesting part: `load_capability_manifest("vortex")` raised, which blocked the task-prompt
   generator and `run_baseline_qa_loop` outright — but `capsule_runner._endpoint_of` **catches the
   exception and returns `(None, None)`**, so the oracle path kept working by degradation and nothing
   said the contract was absent. That is why the probe run passed without it: the descriptor's
   `sim_via: vortex` self-routed around the hole.

   **GENERATED from `capability_manifests.vortex_manifest()`**, not hand-authored (unlike gemmini's) —
   vortex is already in `MANIFESTS`, so the Python function is the single source and carries the
   per-field rationale that YAML emission drops. `cm.write('vortex')` regenerates; a test asserts the
   tracked file is **byte-identical** to regeneration, so a hand-edit is caught rather than silently
   reverted. Everything the `simt` family profile supplies is deliberately NOT restated
   (`encoding_required=False` — the CUSTOM0 map is fixed by the ISA so there is nothing to derive; the
   `simt_coverage` gate; `rtl_tiers=("L3",)`; the linalg-input entrypoint set).

   The declarations that are NOT derivable, and what each one prevents:

   | declaration | why it cannot be defaulted |
   |---|---|
   | `runner.tier_sim: {L2: simx, L3: rtlsim}` | `runner_config` derives the oracle tier **loop** from these keys. Omitted ⇒ `oracle_tiers == ()` ⇒ no tier ever grades, and `not_run_is_not_pass` reports `incomplete` — reads as a harness fault, not a missing line |
   | `kernel_abi.symbol: merlin_kernel_body` | consumers guessed `f"{target}_kernel"`; `vortex_kernel` fails to link against the curated harness. `generate_prompt` now reads the declaration, defaulting to the convention, so gemmini stays `gemmini_kernel` |
   | `runner.dtype: mixed_i8_f32` | run-identity token; Vortex has no single one (an i8xi8→i32 family AND an f32 family on the same core), so it names both rather than picking one |
   | **no** `force_match_policy` | a float-only SIMT target would set `{compare: float, atol}`; that would keep *passing* the i8 half of the corpus while no longer checking it bit-exactly |

   Two deliberate omissions: **no `dialect_plan.yaml`** (the vocabulary is the agent's design decision —
   `dialect_plan_from_manifest` would pre-bake `vortex.matmul`, handing the assisted arms a head start),
   and **no `rtl_sim_config`** analog of `GemminiRocketConfig` (Vortex's geometry is a build parameter,
   so "which machine" is an experiment decision; the descriptor declares itself its single source).

   Guarded by `tests/targetgen/test_vortex_target_contract.py` (10 tests). Two are drift guards proven
   to bite by mutation: byte-identical regeneration, and the contract's `capabilities.simt` geometry
   pinned against the descriptor's `frozen` block — that pair would otherwise drift with **nothing
   failing**, since cycles and `pct_fp_peak` would just quietly stop being comparable to the declared
   peak. A third pins `kernel_abi.signature` against the harness header that actually declares it.

6b. **Vortex RTL-check compiler** — the `rtlchecks` arm's feedback generator: a Vortex branch of
   `rtl_check_compiler.py` + `rtl_check_runner.py` emitting the check table above, fed by RTL facts from
   the HW-dialect import and the capsule's declared shape. Must hold the non-overfit invariant (no
   golden-derived literals). Needed only for arm 3, so it can land after arms 1-2 are running.
7. ~~**QA gate + grader**~~ — **DONE**. The redacted `qa/verdict.json` loop now runs for Vortex off the
   SHARED gemmini drivers (`MERLIN_TARGET_EXPERIMENT=<descriptor>`); no `scripts/` copy under this
   experiment, because copying a ~1200-line driver per target is what `_common.EXP` exists to avoid.
   Five things had to change, four of them shared-machinery coupling only a second target could expose:

   | # | problem | fix |
   |---|---|---|
   | 1 | `sandbox/toolchain.py` built `SIM_TOOLCHAINS` at IMPORT time with `ext_path("chipyard")`, which RAISES when unset — so `run_baseline_qa_loop.py` was **unimportable for every target**, chipyard-using or not | `_ext_or_none()`; each family keeps its existing `/nonexistent/...` degrade and fails honestly at USE time, via its probes |
   | 2 | no `vortex` entry in `SIM_TOOLCHAINS`, so a sandboxed run would bind nothing — no sysroot, no `libvortex.so`/`librtlsim.so`, no oracle | a `_vortex()` family binding **only** `tools/` + `build/sw/runtime/`. `sw/kernel`, pocl/chipstar and `tests/` stay out: binding the checkout root would hand every arm the SDK this experiment withholds. A test asserts that boundary |
   | 3 | `launch_ab_batch.SCRIPTS = C.EXP/"scripts"` pointed at a directory Vortex does not have, and `--dry-run` printed the commands and said "preflight OK" | fall back to the drivers' own dir when the selected experiment ships no `scripts/`; the plan now checks each driver EXISTS (refuses, rc=2); the banner says "plan resolves ... the cheat gate has NOT run" rather than conflating `--dry-run` with `--preflight` |
   | 4 | the graded language mandate spelled `gemmini-opt` / "gemmini target dialect" / "emit BOTH cb AND LLVM/RoCC" | `_lang_mandate()` derives all of it; the artifact clause follows `manifest.entrypoints`, so Vortex is not told to emit a command buffer it has no entrypoint for. `generate_prompt.tool_stem()` is now the single source of `<target>-opt` |
   | 5 | the SIMT coverage verdict was written to `simt_coverage.json` but **never into `capsule_result`**, so `trace_check` stayed `{"status": "skipped"}` | `capsule_runner.coverage_to_trace_check()`. This one is not cosmetic: `qa_check` builds the agent's verdict from the RESULT, so the QA loop was reporting "coverage: skipped, no violations" on exactly the capsules the gate had just rejected — the plane was there, the violated class names were not |

   `qa_check` emits `coverage_status`/`coverage_violations` as **aliases** of the historical `trace_*`
   pair rather than a rename: gemmini's committed task prompts name `trace_violations` and it has
   measured runs, while `task/TASK_full.md` here names `coverage_violations`. Both are now true.

7b. ~~**`readiness_check.py` graded the wrong target**~~ — **FIXED**, and worth recording as a
   near-miss. It hardcoded `EXP = merlin_dir()/"experiments"/"gemmini_capsule_bench_v0"`, so running it
   with the override set printed a confident verdict computed entirely from gemmini — including
   "[PASS] all 6 bundles present with prompts" for a target with three, and a gemmini capsule under
   "parse a real capsule". Now: `EXP`/`TARGET` honor the override, the corpus comes from the
   descriptor's `capsule_corpus`, the bundle set is derived from the launcher's own arm->bundle table
   (not a hardcoded 6, and not every dir on disk — which would drag in gemmini's retired
   `public_v0`/`realistic_v0` bundles), and the verdict line names the target it graded.

   Three sections are the **chipyard plane** (the CIRCT fact generators, the CIRCT sim-skip gate, and
   the oracle timing run against a known-good reference backend). For Vortex they are not failures,
   they are inapplicable — `sim_via != chipyard`, and this experiment ships **no reference backend by
   design**. Scoring them FAIL produced a NO-GO that could never turn green, which trains everyone to
   launch past the gate. They now report `[N/A ]` with the reason, gated on the descriptor's
   `toolchain.sim_via` (never a target-name branch).

   Result: **vortex 9/9 GO**, gemmini unchanged at 9/15 (its FAILs are the absent chipyard/mlc).

7c. ~~**`grader_private_v0`**~~ — **DONE**. Every arm's generated manifest denied the path and nothing
   was there; a deny naming no referent is not a rule. Hand-authored (the arm bundles are generated;
   this one is not on that ladder), listing the hidden set + grader + the Vortex oracle/coverage
   adapters, and carrying the two operational notes whoever grades will need: scan the agent's OBJECT
   rather than the linked ELF, and grade the hidden set on L2 before spending L3.

7d. ~~**`MERLIN_EXT_VORTEX` / `MERLIN_VORTEX_HARNESS`**~~ — **DONE**, added to `.env.example`
   (`MERLIN_MLIR_TRANSLATE` was already there). The `MERLIN_EXT_VORTEX` note records which subtrees the
   sandbox binds and which are the denied vendor software, so the deny surface is discoverable from the
   env file and not only from the bundle generator.

7e. **OPEN — the oracle's link step is not reachable under the Vortex sandbox bind.** Registering the
   `vortex` sim toolchain forced a choice that had been implicit. `tools/` cannot be bound wholesale:
   it contains `llvm-vortex`, the LLVM fork every arm's bundle denies BY NAME, whose clang implements
   `+xvortex` and inserts split/join reconvergence for you — the exact compiler work V10/H9 exist to
   measure. So the bind is narrowed to `tools/{riscv64-gnu-toolchain,libc64,libcrt64}` + `build/sw/runtime`.

   But `vortex_oracle.build_image` shells out to three binaries under `tools/llvm-vortex/bin/`:
   `clang++` (the link), `llvm-objdump` (the `kernel_startup.sh` check), `llvm-objcopy` (vxbin). Those
   are runner-owned steps, not the graded compile — the graded compile is stock clang, unchanged — but
   they are unreachable under this bind, so an in-sandbox run would fail at the link.

   Two honest fixes, both small: move those three to the stock LLVM already on the sandbox PATH (best —
   they are binary utilities and a linker, nothing fork-specific is used), or give the three binaries an
   explicit narrow bind. What must NOT happen is binding the parent to paper over it: that trades a
   broken link for a compromised measurement, and no deny list would catch it. **Flagged, not hidden** —
   this is the last thing between the rig and a sandboxed launch of arms 1-2.

7e. ~~**A readiness check must not destroy results**~~ — **FIXED**, found by accident and worth
   recording. Section D runs `agg_ab_results.py` purely to prove it exits 0, and it wrote to the
   canonical reports dir. With no matching run dirs present the aggregation is `n_runs: 0`, so running
   the readiness check **overwrote the tracked `out/artifacts/capsule-bench/gemmini/ab_results.json`,
   wiping the measured abc4 cells** — to check that a script runs. Caught because the file showed up in
   `git diff` after my own verification runs; restored from HEAD. `agg_ab_results.py` now takes
   `--out-dir` (the WRITE side only — `collect()` still reads `full_suite_audit.json` from the canonical
   dir, which is an input), and section D points it at a temp dir. A test pins that a smoke aggregation
   leaves the tracked file byte-identical.

7f. ~~**The deny surface had gone dead**~~ — **FIXED**, and this is the most serious thing found in the
   whole bring-up. The Vortex checkout moved out of the repo (`merlin/tmp/vortex`) to a sibling
   (`<workspace>/vortex`), and the descriptor's **14 hardcoded `tmp/vortex/...` paths silently became
   dead** — including all **11** `answer_surfaces.denied_target_software` entries. The one thing that
   makes this experiment measure compiler bring-up rather than SDK usage was naming nothing at all.

   The test that should have caught it made it worse: `test_no_allowed_path_inside_the_repo_is_dangling`
   **exempted the `tmp/` prefix** as a "launcher mount", which — once the checkout lived there — was a
   blanket "do not check anything Vortex". A guard written around a path that later changed meaning.

   Fixed by adopting gemmini's discipline: a committed descriptor holds **no** external checkout path.
   Gemmini declares `sim_via: chipyard` and lets the code resolve `MERLIN_EXT_CHIPYARD`; Vortex now
   declares `rtl.root_env: MERLIN_EXT_VORTEX` and every Vortex path is CHECKOUT-relative
   (`sw/kernel/include/vx_intrinsics.h`, not `tmp/vortex/sw/...`). `TargetExperiment` grew
   `root_env` + `external_root()` / `resolve_external()` / `denied_display()`; `generate_bundles`
   renders denies as `${MERLIN_EXT_VORTEX}/<rel>` so no reader mistakes them for repo-relative — which
   is exactly the confusion that let them rot. Gemmini declares no `root_env`, so its bundles are byte-
   identical (`test_generate_bundles` unchanged).

   Three guards replace the exemption, all verified to bite:
   - `test_every_denied_path_still_names_a_real_file` — resolves each of the 11 against the checkout and
     fails if it is gone (skips only when the checkout is unresolvable). Proven by pointing
     `MERLIN_EXT_VORTEX` at an empty dir: **fails**, does not skip.
   - `test_the_denied_software_is_unreachable_from_the_sandbox` — the bundle deny is documentation; the
     **sandbox bind is the enforcement**, and nothing cross-checked them. Proven by widening the bind
     back to `tools/`: **fails**, because that exposes the denied `tools/llvm-vortex`.
   - the `tmp/` exemption is **gone**; a `tmp/...` grant reappearing is now a real dangling path.

7g. ~~**A GO that meant nothing**~~ — **FIXED**. `readiness_check` reported **9/9 GO for Vortex while
   both oracle tiers were unavailable**, because section G (the only part that runs an oracle) is
   chipyard-plane and correctly `[N/A]` here — so nothing checked the tiers that actually grade. It
   measured tooling wiring and said nothing about whether one capsule could be graded.

   Now the non-chipyard branch runs `_oracle_availability()` first: tiers from the manifest's `tier_sim`
   (the same map `runner_config` builds the grading loop from), availability from the adapter module the
   descriptor names in `toolchain.runner`. No target name appears in the code — a test parses the source
   with `ast` and fails on `simx`/`rtlsim`/`vortex` appearing in any non-docstring literal.

   Behaviour change, verified both ways: checkout absent -> **10/12 NO-GO** naming the two dead tiers;
   checkout resolvable -> **12/12 GO**. Gemmini unchanged at 9/15 (it takes the chipyard branch).
   `MERLIN_EXT_VORTEX` is now set in the gitignored `.env`, so `available()` is True for L2 and L3.
8. ~~**Bundle materialization**~~ — **DONE**. All three bundles now carry `input_bundle_manifest.yaml` +
   `allowed_files.txt` + `denied_files.txt` + `README.md` + `STARTER_PROMPT.md` + `bundle_lock.yaml`.

   Note `STARTER_PROMPT.md` is **not** `render_prompt` output — gemmini's is a 33-line target-agnostic
   *method* hint ("derive the ISA encoding once, up front"). The main task prompt is `task/TASK_full.md`,
   which for vortex is authored and correct. So `render_prompt` is not on this experiment's critical path.

   **The derived companions are now GENERATED, not hand-kept** (`materialize_bundles`), because the
   hand-kept version had already failed: gemmini's `allowed_files.txt` lists **8** paths where its own
   manifest grants **10** — missing `hwbringup_gemmini_v0` (the whole shared hardware-spec set) and
   `agent_selfcheck.py`. Its `README.md` still opens with the retired id `raw_baseline_public_v0`. Nothing
   reads either file programmatically, so a reader asking "what does this arm get?" got the wrong answer
   with nothing failing. Gemmini's files are left untouched deliberately — its older `public_v0`/
   `realistic_v0` bundles predate the current descriptor and may not regenerate faithfully, so that is its
   own item, not a side effect of this one.

   **`bundle_lock.yaml` is written by a new target-generic `lock_bundles`**, not by gemmini's
   `preflight.py`. Porting preflight was the wrong move: beyond `EXP` it needs bwrap canaries, bareMetalC
   anchors, a captured claude stream, and a hardcoded 2-arm list. Its adversarial checks (canary isolation,
   negative fixtures, freeze tamper) remain a separate item; the *lock* is just a content hash per grant.
   A granted FILE is hashed by its own bytes, never its parent dir — pinning the parent would pin siblings
   the arm was not granted.

   **What the lock buys, concretely:** "every arm gets the same hardware inputs" was an assertion about a
   mutable tree; it is now a checkable fact. All three arms hash **identically** on the six shared inputs
   (`merlin/contract/`, the three public capsule families, the spec-sheet set, `task/`). Only two grants
   are legitimately unlocked: `third_party/llvm-install/` (a launcher mount) and arm 3's
   `merlin/targets/vortex/contracts/rtl_facts/` (mlc-generated).

   Guarded by `tests/targetgen/test_vortex_bundle_materialization.py` (30 tests). Four proven to bite by
   injection: a grid-mapping hint added to a brief, a dropped line in a derived listing, and a
   spec-sheet hash made to differ across arms were each caught.

   **The briefs deliberately have NO lowering-plan section, and a test enforces that.** Gemmini's baseline
   brief safely describes its dataflow (preload -> compute -> readout, im2col) because Gemmini is
   fixed-function: the dataflow leaves tiling and scratchpad management — where its compiler work lives —
   untouched. For a programmable SIMT core the analogous "method" hint IS the deliverable, so the briefs
   state the encode-once discipline, the reconvergence and L2-visibility obligations, the two-tier iterate
   strategy and the no-shape-special-casing rule, and stop there. `test_brief_states_no_work_distribution_
   mapping` rejects `block_id`/`thread_id`/`block_dim`/the CSR numbers/"one thread per"/"grid-stride"; a
   second test rejects every vendor name (`vx_spawn`, `vx_intrinsics`, `xvortex`, …) since the *name* is a
   pointer even when the file is withheld.

8b. **The assisted arm's kit is thinner for Vortex than for gemmini — measured, not assumed.**
   `oot_starterkit.parse_interface` is written for the fixed `merlin_iface` grammar. On this target's stock
   linalg/arith capsules it **does not raise** — it returns `{tensors: 0, commands: 0, target: ""}` for
   every capsule tried (V0, L6, V10). So gemmini's assisted brief, whose headline advice is "parse the
   input into VERIFIED IR — never regex", would point a Vortex agent at a tool that answers confidently and
   emptily. `CommandBufferBuilder` likewise has no stage on this target. **Both are named as inapplicable
   in the assisted briefs**, and a test asserts they stay named — telling the agent what to skip is worth
   as much as telling it what to use.

   What genuinely does work, verified: `verify.validate(module)` / `verify_module` run xDSL's native
   verifier on a Vortex capsule module (`{"ok": True, "findings": []}` on a valid one, and it **bites** —
   a return-type/signature mismatch yields `{"ok": False, "findings": ["xDSL verify: ..."]}`);
   `transforms.im2col` is pure shape math and applies to the conv family; and the Step-0 factory work pays
   off — `factory.build_dialect("<yours>", plan={...})` builds a **self-designed SIMT vocabulary** from an
   in-memory plan (no committed file), the tensor-resident accessors correctly raise `AttributeError`, and
   `generate.xdsl.generate(plan)` emits real IRDL naming the plan's own ops rather than the empty template.

   `merlin.kernels` CCA is granted and readable but was built around tensor-resident targets; the briefs
   call it reference, not a path.

### Arm 3 must NOT launch yet — its distinguishing grant is empty

The lock makes this concrete: arm 3's only extra grants over arm 2 are `targetgen/rtl/` and
`merlin/targets/vortex/contracts/rtl_facts/`, and the facts pin hashes to **null** (mlc absent, and item 6b
unbuilt). So arm 3 would be materially identical to arm 2 while its brief promises RTL-derived facts and
per-round advisory checks that never arrive — the "promises a tool that is not there" failure the bundle
generator already guards for file grants. Gate arm 3 on 6b landing **and** a non-null `rtl_facts` hash.
9. **`grader_private_v0` bundle** — every arm's manifest *denies*
   `input_bundles/grader_private_v0/`, but it does not exist. `generate_bundles` emits only the deny;
   gemmini's is hand-authored (manifest + README).

### No `scripts/` port is needed — with one exception that must be fixed

`_common.py` already supports `MERLIN_TARGET_EXPERIMENT=<descriptor>`, and the drivers that matter honor
it (`run_agent_experiment`, `launch_ab_batch`, `qa_check`, `qa_check_rtlchecks`, `run_fullsuite`,
`run_baseline_qa_loop`, `preflight` via `C.BUNDLES`). So the gemmini harness can drive Vortex directly.

**But `readiness_check.py` hardcodes `EXP = merlin_dir()/"experiments"/"gemmini_capsule_bench_v0"` and
ignores the override.** Run with `MERLIN_TARGET_EXPERIMENT` set to the Vortex descriptor it prints a
confident verdict computed **entirely from gemmini** — including `[PASS] all 6 bundles present with
prompts, missing=[]` while vortex has 3 bundles and zero prompts, and a `parse_interface` check against
`capsules/layers/B4_conv2d_relu_i8`. It reads as a Vortex readiness verdict; it is not one. Same
false-confidence class as the vacuous guards logged elsewhere here. Fix: take `EXP`/`BUNDLES` from
`_common`, and derive the expected bundle set from the descriptor's arms instead of a hardcoded 6.

### `generate_prompt.py` carries two more gemmini-shaped assumptions (generator bugs, not run blockers)

Landing the contract made these renderable, hence visible. Vortex's authored `task/TASK_full.md` already
gets both right, so they bite whoever *generates* a prompt for the next target:

- **It templates four fixed entrypoints**, including `emit_command_buffer`, then says "Declare these four
  commands in `manifest.yaml` exactly as the runner expects" — contradicting `RunnerConfig.entrypoints`,
  which for a SIMT target drops `emit_command_buffer` and adds `optimize_interface`. An agent following
  it would build a package the runner then fails for a stage mismatch (probe bug #2's plane).
- **`hwbringup_set`, `isa_headers` and `prior_backend_deny` are computed in `prompt_slots` but never
  rendered.** So the prompt never tells the agent where the ISA spec sheet is — and for Vortex, whose
  derived fact bundle is empty (0/4 grounded, `simt_config` needs mlc), the ISA section renders as
  "Legal opcodes: unavailable" with no pointer to `VORTEX_ISA_SPEC.md`, which is the entire ISA statement
  for this experiment. Worst kind of failure: it looks like a plausible prompt.

Also, `render_fact_bundle_for` routes SIMT through the **systolic** renderer (there is a standing TODO),
so a Vortex brief asks "Mesh DIM" and "On-chip capacity" — honest ("unavailable, not guessed") but noise.

## FIRST END-TO-END RUN — the chain works (2026-07-30)

A throwaway probe package (four entrypoints, hardcoded to V0's shape, kept outside the repo so it can
never become an answer surface) was graded through the real `capsule_runner`:

```
STATUS  : pass
NUMERIC : status=pass  max_abs_diff=0  mismatch_count=0  golden_source=vortex_corpus_binary64
  L0: skipped   (no command buffer for this target — nothing for reference/simulate to interpret)
  L1: skipped   (same)
  L2: pass      cycles=1207   simx
  L3: pass      cycles=3352   rtlsim   derived_from_rtl=True  cycle_accurate=True
```

So the whole path is live: 4 entrypoints -> LLVM-dialect module -> `mlir-translate` -> stock clang ->
coverage gate -> link against the curated harness -> simx **and** rtlsim -> compared against the
withheld golden, exactly (0 diff, not merely inside tolerance).

Getting there surfaced **six** integration bugs, none of them in the Vortex work itself — every one was
target-coupling in shared machinery that only a non-gemmini target could expose:

| # | bug | fix |
|---|---|---|
| 1 | `manifest.schema.json` **required** `commands.emit_command_buffer`, so a correct Vortex package could not validate | required set trimmed to the stages every target has; the per-target set is enforced by the runner, which knows the target |
| 2 | a package missing a stage *its* target needs would `KeyError` in argv resolution (a RUNNER_CRASH, i.e. our bug not theirs) | `run_entrypoints` validates `entrypoints` against the manifest up front -> clean `CertFailure("contract")` |
| 3 | `toolchain_shas()` resolved **gemmini's chipyard** unconditionally, so any Vortex capsule crashed at the very END — after entrypoints, gate and oracle had all passed | degrades honestly: records `merlin` always, chipyard/gemmini only when present |
| 4 | `capsule_golden.golden()` gated "read the shipped golden" on a **float** policy, so an integer capsule with a perfectly good independent golden went to the recompute path and died with `unsupported operation 'elementwise'` | gate on declared provenance (`golden_source`), not dtype |
| 5 | the Vortex goldens declared no `golden_source`, so they defaulted to `merlin_tensor_int` (= recompute me) | generator emits `vortex_corpus_binary64` / `vortex_corpus_exact_int`. **Golden VALUES unchanged** (verified against a pre-change snapshot: 0 of 8 differ), so the V0/V2 hardware validation stands |
| 6 | the coverage gate required `TMC`/`WSPAWN`, which a KMU kernel never emits | require `CTA_CSR`; see item 2 above |

`status: incomplete` on the first attempt was **correct behaviour**, not a bug: V0 declares
`required_oracle_tiers: [L2, L3]` and only L2 was wired, so `not_run_is_not_pass` refused to call it a
pass. Worth remembering as the integrity backbone working.

Still unexercised by this run: the divergence capsules (V10/H9 `scf.while`), softmax, conv2d, the
`any_of` gate, the hidden set, and the QA-loop/grader wrapper. The probe handles one capsule shape by
construction.

To reproduce: `VORTEX_HOME=<vortex checkout>` and `MERLIN_MLIR_TRANSLATE=<circt>/llvm/build/bin/mlir-translate`
(neither has a `MERLIN_EXT_*` entry yet — worth adding).

## Environment prerequisites for an actual run

Two sibling checkouts, neither on PyPI nor declared in `pyproject.toml` — so **no `uv sync` flag
installs either** (`--all-extras` is already maximal and there are no dependency groups). See
`docs/guides/getting_started.md` §1.

- **`aet` — INSTALLED** (`../agentic-eval-tool`, [ucb-bar/agentic-eval-tool](https://github.com/ucb-bar/agentic-eval-tool),
  editable into `.venv`). Verified: all five modules merlin imports resolve, all seven
  `FailureCategory` members it references exist, and the QA gate runs end-to-end against the Vortex
  descriptor with no stub. Re-install after a fresh `uv sync` with
  `uv pip install -e ../agentic-eval-tool`. Note the PyPI package named `aet` is an **unrelated**
  third-party CLI — installing it would satisfy the import and then fail with `AttributeError`s.
- **`mlc` — still absent.** Only arm 3 needs it (see below).

Why the deferred imports still matter now that aet is present: they are what lets a target's oracle
adapters resolve, and the structural planes run, in an environment where aet is not installed — which is
every fresh clone until someone runs the extra install step above.

### Why the aet imports are deferred (keep them that way)

Runs are aet-managed: `capsule_common.make_run_paths` builds a `RunSpec`/`RunPaths` and
`oot_runner._record` writes the ledger through `ArtifactStore` / `FailureRecord` / `EvalRunLogger`.
There is no substitute — grading a capsule needs aet. But its absence used to block far *more* than
grading: `capsule_runner`, `capsule_common` and `oot_runner` imported aet at module scope, so a
target's oracle adapters could not even be **resolved** and the QA gate could not be imported at all.
Those imports are now deferred to the functions that genuinely need them (matching the convention
`capsule_common._cat` already used), with `FailureCategory` a lazy proxy so its 19 call sites are
untouched. `capsule_runner` also imported `rocc_decode`, which builds its ISA table from gemmini's RTL
facts at import and so demanded `MERLIN_EXT_CHIPYARD` even for a target with no RoCC; that moved to the
one `trace_gate == "rocc_insn"` branch a SIMT target never takes.

Net effect: `oracle_adapters("vortex", "vortex") -> {L2: simx, L3: rtlsim}` resolves with neither aet
nor chipyard installed, and `qa_loop_adapters` picks L2 (simx) as the fast tier. Keep it that way —
`tests/targetgen/test_lazy_aet_imports.py` enforces it, including the failure mode that bit once: a name
left under `TYPE_CHECKING` while a runtime use remains raises `NameError` **only where aet IS
installed**, so it is invisible in a fresh clone and fires in a real run.

- **`mlc` is required for arm 3's RTL facts.** Absent, so `merlin/targets/vortex/contracts/rtl_facts/`
  does not exist (nor does gemmini's — it is generated at launch). Arms 1-2 do not need it.
- **`MERLIN_EXT_CHIPYARD` is gemmini-only.** With aet installed it is the next unmet dependency in the
  test suite, but every one of those failures is a gemmini test (chipyard is its spike/verilator).
  Vortex grades on simx/rtlsim from the Vortex tree and never needs it.

**Trap: always pass `sim_via`.** `oracle_adapters("vortex", "vortex")` gives `{L2: simx, L3: rtlsim}`,
but `oracle_adapters("vortex", None)` silently falls through to the **mlc arc adapter** — the wrong
oracle, and it fails as "arc model unavailable" rather than as a misroute. The reason is that
`_endpoint_of` resolves `endpoint_kind` from a capability manifest **materialized under
`out/artifacts/targets/<target>/`**, which only gemmini has in this workspace, so it returns
`(None, None)` for vortex and the routing falls back to the default. The real callers are fine —
`qa_check.py` reads `sim_via` from the descriptor — but any direct call must pass it. (The same missing
fixture is why `test_arc_oracle_adapter.py::test_external_backend_target_uses_the_program_oracle` fails:
atlas is routed to the arc adapter instead of the program oracle. Pre-existing, unrelated to Vortex,
and newly *visible* only because that file used to fail at collection.)

## Open decisions (flagged, not silently resolved)

- **Contract extension** — the RUNNER now supports the Vortex stage set; the contract DOC has not caught
  up. `RunnerConfig.entrypoints` (keyed on compute-unit kind, overridable via a contract's
  `runner.entrypoints`) decides which CLI entrypoints run, and `simt` resolves to
  `parse -> optimize_interface -> lower_interface_to_target -> emit_target_artifact`. Two fixes this
  needed, both of which were breaking every Vortex capsule:
  - `capsule_common.run_entrypoints` invoked `emit_command_buffer` **unconditionally**, so a
    programmable core — which has no command stream — failed at the `target_to_command_buffer` plane.
    It is now gated, and a target that emits none gets `cb=None`; L0/L1, which exist only to interpret a
    command buffer, are then skipped `not_applicable` with that reason, alongside the existing
    float-datapath skip.
  - `optimize_interface` was never invoked at all, so the global linalg-optimization stage `TASK_full.md`
    declares went ungraded. It now runs, writes `optimized.interface.mlir` as evidence, and fails on its
    own plane.

  Still to sign off: `contract/mlir_oot_backend_contract.yaml` v0.1 documents neither
  `optimize_interface` nor an optional `emit_command_buffer`, and its `kernel_abi.symbol` says
  `gemmini_kernel` (Vortex's is `merlin_kernel_body`; `generate_prompt.py` templates a third spelling,
  `{target}_kernel`). Doc-level only — nothing enforces it — but it should be a `vortex` profile or a
  v0.2 bump. Guarded by `tests/targetgen/test_runner_entrypoints.py`.
- **Does optimization quality gate?** Currently no — cycles are diagnostic-only, matching gemmini. That
  keeps pass/fail crisp, but this bench exists to compare how well compilers optimize for the hardware,
  so a reported-but-ungated perf number may be too weak a signal. An explicit perf tier (e.g. "within
  Nx of a reference schedule") is the alternative.
- ~~**Float tolerances.**~~ **RESOLVED — derived, not guessed.** `atol` comes from the standard
  floating-point error bound for the capsule's own operation and operands
  (`gamma_K = K*u/(1-K*u)`, `u = 2^-24`, times the sum of term magnitudes, times a safety factor), which
  admits *any* legal reassociation of the reduction by construction while still separating legal
  reordering from real bugs by ~2000x on the pilot shapes. See `vortex_oracle.dot_error_bound` /
  `derive_matmul_atol` / `derive_elementwise_atol`. Integer capsules stay bit-exact.
- **L3 subset.** Which capsules require rtlsim. At ~5-30 kHz simulated-cycle throughput, this must stay
  small and small-shaped.
- **Whether the authored spec sheet can stay neutral.** We derive it from the RTL and the headers we are
  denying the agent. Writing it without leaking mapping strategy (e.g. describing `wspawn` without
  implying how to distribute a loop nest across warps) is a real authoring hazard, and it silently sets
  the difficulty of every arm. Worth a second reader.
- **Arm 3's fact sourcing** — measured against `Vortex-sv2v.mlir`; see the section below. The opcode set
  is recoverable from the flat blob; the CUSTOM0 sub-decode is not (it needs the `(funct7, funct3)` pair,
  recoverable only from the per-module import), and the extraction tool is absent (mlc). Decide whether
  arm 3 sources those literals from the authored spec sheet — legitimate, since they are a declared ISA
  structural rule rather than a golden, and `vortex_coverage.CUSTOM0_CLASSES` already encodes them — or
  waits on mlc so the claim "RTL-derived" is literally true for arm 3.

## Measured: what the Vortex HW-dialect import actually yields

Run against `Vortex-sv2v.mlir` (1.5 MB, 19,623 lines, 341 `hw.module`s). Three findings that shape arm 3:

**1. The core pipeline is inlined into one module.** sv2v's interface flattening dissolved every
interface-carrying module, so there is **no `VX_decode`, `VX_issue`, `VX_schedule`, or CSR module**. The
top `@Vortex` is **7,211 lines — 37% of the whole design**; every other module is a leaf utility (caches,
FIFOs, arbiters, RAMs, `VX_ipdom_stack`, `VX_split_join`). Any decoder analysis must work *inside* one
flat, optimized, unnamed-SSA module body rather than on a named decoder module — the opposite of the
Gemmini shape the existing tooling was built for.

**2. The opcode set IS recoverable — the gemmini technique works.** 24 distinct `comb.icmp ceq` compares
against 7-bit constants in `@Vortex`; 21 are exactly the RISC-V opcode map, **including CUSTOM0 (0x0B)
and CUSTOM1 (0x2B)** — so Vortex uses two custom opcodes, worth confirming when authoring the spec sheet.
The 3 non-opcode hits (0x00, 0x01, 0x04) are trivially filtered by intersecting with the known opcode map.
Note the op is `comb.icmp ceq`, not `eq`.

**3. In the flat blob, the CUSTOM0 sub-decode is NOT recoverable by funct3 fan-out alone.** All eight
values {0..7} appear as i3 compares — the complete 3-bit space — so the fan-out cannot narrow which
funct3s are legal. **Note what this actually meant**, understood only later while building the coverage
gate: it is not noise. `funct3` is not the whole selector — CUSTOM0 is two families keyed by `funct7`,
and across them **every** funct3 value genuinely is legal (0-5 and 7 under `funct7=0`; all of 0-7 under
`funct7=1`). The flat blob was reporting the truth; the reading of it was wrong, because the spec sheet
at the time listed only `funct7=0`, `funct3` 0-5. Any RTL-derived decode must therefore recover the
**(funct7, funct3) pair**, not funct3. **Fixed by the per-module import below**, which makes the walk
tractable.

**4. The extraction tool is absent.** The `comb.icmp` fan-out lives in **external mlc**
(`MERLIN_MLC_DIR`), not in the in-repo `rtl/circt_introspect.py` — which only reads `hw.module` port-list
text and `hw.instance` lines, i.e. none of the above. `MERLIN_MLC_DIR` is unset and no mlc checkout
exists in this workspace, so today *nothing here can run finding 2 automatically*, though it is ~20 lines
of regex over the module body.

## The fix for the inlined pipeline: per-module import (VERIFIED)

Do **not** try to de-interface the RTL (112 modules carry interface ports) and do **not** wait on CIRCT
interface-*array* support (43 array sites block the whole-design direct import). Neither is needed,
because arm 3 does not need a whole-design import — it needs *specific modules*. Import them one at a
time via the direct (non-sv2v) path, which preserves module boundaries.

The only obstacle is that a top-level module cannot have unconnected interface ports
(`error: top-level module 'VX_decode' has unconnected interface port 'fetch_if'`). A ~15-line synthetic
wrapper that instantiates the interfaces and binds them fixes it. All three of `VX_decode`'s interfaces
are parameterless, so the wrapper is trivial:

```systemverilog
module VX_decode_probe import VX_gpu_pkg::*; (input wire clk, input wire reset);
    VX_fetch_if fetch_if(); VX_decode_if decode_if(); VX_decode_sched_if decode_sched_if();
    VX_decode dut (.clk(clk), .reset(reset), .fetch_if(fetch_if),
                   .decode_if(decode_if), .decode_sched_if(decode_sched_if));
endmodule
```

```
make mlir TOP_LEVEL_ENTITY=VX_decode_probe PREFIX=probe EXTRA_INCLUDE=-I<wrapper_dir>
```

(The Makefile's `EXTRA_INCLUDE` hook already exists for exactly this "sub-block top + its deps" case.)

**Result: 945 lines, three named modules — `@VX_decode` preserved at 919 lines**, versus 7,211 lines of
anonymous flat logic. And the decode logic arrives as **structured control flow** (`cf.cond_br` over the
case statement), not a flattened `comb.and` soup, so extraction is a CFG walk rather than dataflow.

### Verified extraction recipe

1. `%50 = comb.extract %instr from 0 : (i32) -> i7` — the **opcode field**, with provenance.
2. `comb.icmp ceq %50, %cN_i7` — the legal opcode set. Requiring the operand to be *that extract*
   eliminates the 3 false positives seen in the flat blob (they compared unrelated 7-bit values).
3. `%468 = comb.icmp ceq %50, %c11_i7` → `cf.cond_br %468, ^bb61, ^bb83` — the CUSTOM0 branch.
4. Walk the CFG from `^bb61`, collect branch predicates, resolve each definition. Predicates are
   **hoisted above the block**, so a naive in-block scan finds nothing — resolve by def-use.

That yields, against `%52` (the funct3 field): `%109: funct3==0`, `%114: funct3==1`, `%69: funct3==2` —
i.e. **tmc / wspawn / split**, matching the spec sheet exactly. The fact arm 3 was missing is recovered.

**Consequence for arm 3:** both the opcode set and the CUSTOM0 funct3 classes are genuinely RTL-derived,
so the checks can claim full RTL grounding. Remaining caveats: the extraction is ~100 lines of new code
(the in-repo `circt_introspect.py` reads only port-lists/instances, and external mlc is absent), and
operand *semantics* — that funct3=1 means "spawn N warps at PC" — is still inference, so op **naming**
comes from the spec sheet while op **legality** comes from the RTL.

Two build items follow: promote the probe wrapper into
`contracts/hwbringup_vortex_v0/probes/` (it is currently only in a scratch dir), and decide which
modules get probes — `VX_decode` for the ISA, plus likely `VX_csr_unit` (CSR map), `VX_split_join` /
`VX_ipdom_stack` (reconvergence), and a cache/LSU module for capacity.

## mlc / ModeLIR — Vortex RTL facts are now DERIVED (2026-08-02)

`mlc` is **ModeLIR** (`git@github.com:copparihollmann/ModeLIR.git`); the importable package is `mlc/`
at the repo root, so `MERLIN_MLC_DIR` is the repo root — not the `modeling/` subdir the stale default
(`/scratch2/agustin/mvp-lhwir/modeling`) implies. Cloned as a sibling, editable-installed, `.env` set.

**Muon was the wrong template, and that mattered.** `muon_introspect` is a MERLIN module, not an mlc
one, and it reads elaborated **FIRRTL** from chipyard's VCS generated-src for `RadianceMuonConfig`.
Vortex is not a chipyard design and has no FIRRTL. Its input is a CIRCT **HW-dialect** import
(`Vortex-sv2v.mlir`), which puts it on gemmini's `circt_introspect` road — i.e. mlc's target-agnostic
`discover` machinery — not muon's.

**The blocker was one predicate.** mlc's `discover_decode_signals` matched only
`comb.ICmpPredicate eq` (0). Vortex's graph carries **267 `ceq`** — SystemVerilog four-state `===`,
which CIRCT emits for anything reaching the HW dialect through **sv2v** rather than FIRRTL. So mlc was
structurally blind to a whole CLASS of targets: before the fix `discover_opcode_set(width=7)` returned a
4-bit cache-state signal. Fixed generically in `mlc/discover/decode.py`
(`_EQ_PREDICATES = {eq, ceq}`, 18 lines) — **local only, not upstreamed** (Agustin's repo).

Result, in `@Vortex` at fanout 25 — the RISC-V opcode map, **CUSTOM0 `0x0b` and CUSTOM1 `0x2b`
included** — plus 73 distinct 12-bit CSR addresses covering the whole contiguous `0xcd0`-`0xce1` block,
so all five CTA identity CSRs are decoder-verified. This makes a claim the capability manifest had been
asserting BY HAND ("24 distinct 7-bit equality compares in @Vortex … including CUSTOM0 0x0B and CUSTOM1
0x2B") machine-reproducible: 21 real opcodes + 3 folded states = 24. The CTA CSRs previously came from
the GENERATED `VX_types.vh`; a header can list a CSR the RTL never matches, and now the decoder is read.

`targetgen/rtl/vortex_introspect.py` (new) derives `isa`; `_simt_fact_bundle` dispatches through a
`_SIMT_INTROSPECTS` map instead of hardcoding muon, and carries each introspect's `undetermined`
reasons through as the field's evidence — so a reader never mistakes silence for "not present".
**1 of 5 fields derived, and the other four say why:**

| field | why not derived |
|---|---|
| `simt` | geometry is a `VX_CFG_*` **build parameter** the descriptor freezes — claiming RTL provenance for a config choice is the false-provenance pattern this tree strips out |
| `registers` | no register-file signal recovered from the HW dialect |
| `shared_memory` | not separated from the cache hierarchy by the generic pass |
| `fp_datapath` | `recognize_fpu_datapath` / `discover_fpnew_formats` both return None (Vortex's FPU is not FPnew) |

**A trap worth remembering:** `geometry.discover_mesh_dim` reports `dim=8` on this graph — from a
64-cell carry-save adder tree (`VX_csa_32`/`FullAdder`), not a systolic mesh. It is a systolic-shaped
question asked of a SIMT design. A test asserts no mesh dimension ever reaches a SIMT target's facts.

Guarded by `tests/targetgen/test_vortex_introspect.py` (6). The opcode test also asserts the RISC-V base
opcodes (OP/BRANCH/JAL/SYSTEM) are present, so a 7-bit signal that merely happens to contain `0x0b`
cannot pass for the opcode decoder.

Environment: `MERLIN_EXT_CIRCT` added (mlc's own `third_party/circt` default is unpopulated; the
sibling CIRCT build is what exists). mlc's suite is unchanged by the fix — `4 failed, 416 passed,
340 skipped, 6 errors`, byte-identical to a stashed baseline.

**Not done / known mlc gaps:** `mlc/discover/{fine_roles,opu_roles}.py` are absent from `main` (merlin
imports both, for gemmini's fine probe and the OPU roles — neither on the Vortex path; there is an
unchecked `feature/discover-datapaths` branch). `mlc.backends.{cosim_core,protocols}` raise
`FileNotFoundError` at IMPORT time wanting a gemmini discovery cache — the same eager-side-effect bug
class as the chipyard one fixed here; merlin survives it via function-local imports. Gemmini's decode
golden was NOT re-verified against the predicate widening (needs chipyard, absent) — the code comment
says so rather than claiming it.

## Sandbox: one mount plan, two isolation backends (2026-08-02)

`bwrap` cannot run on this host and installing it does not help. The sysctls look permissive
(`unprivileged_userns_clone=1`, `max_user_namespaces=1031003`) but
**`kernel.apparmor_restrict_unprivileged_userns=1`** (Ubuntu 24.04+ default) denies the `uid_map` write
to any binary without an AppArmor profile. Proven, not assumed: `conda-forge::bubblewrap 0.11.2`
installs and every variant dies at `setting up uid map: Permission denied`. Conda is doubly blocked —
it cannot install setuid binaries, and a conda-prefix binary is not on AppArmor's allowlist. Only the
distro package works, because it ships `/etc/apparmor.d/bwrap-userns-restrict`.

Rather than fork the isolation code for docker, the **verification** moved to a backend-neutral plan.
That matters because `coverage_gap()` is the proof that no answer surface is reachable: a second argv
parser would have meant two implementations to keep in step and a guard silently covering one of them.

- `sandbox/plan.py` (new) — `Mount` / `MountPlan`, deny-by-default assembly, and `is_exposed` /
  `coverage_gap` / `apply_answer_masks` operating on the PLAN.
- `sandbox/bwrap.py` — now a renderer + the historical entry points as shims.
- `sandbox/docker.py` (new) — the second renderer. **Host-mount model**: binds `/usr`, `/bin`, `/lib`,
  `/etc` exactly as bwrap does, so the mount table (and therefore the proof) is identical and there is
  no image to maintain.
- `Sandbox.backend` + `build_sandbox(..., backend=)`; `--sandbox bwrap|docker|none`.
  **bwrap stays the default** — gemmini's measured runs used it, and changing the default would put a
  different isolation mechanism under results meant to be comparable.

**Phase 0 found a real gap.** `run_agent_experiment._bwrap_wrap` wrapped the REAL agent launch
(not, as its docstring implied, a canary) with `base_argv` alone: **no answer masks** — and `base_argv`
binds `~/.claude`, so the experimenter-memory surface, the exact historical cheat gap `answer_surfaces`
exists to close, was readable — and no claude/toolchain binds, so `claude` was not even on PATH. Drift,
not intent: the QA-loop driver's own wrapper applied the masks; this was its stale twin. Both callers
now go through `build_sandbox(...).wrap()`, verified byte-identical before the hand-rolled copy was
deleted.

**Acceptance bar.** `tests/data/sandbox/bwrap_argv_baseline.json` is a real pre-refactor capture
(6 roster targets x {empty, worst-case}). The **mount table is compared ordered and exactly**; two FLAG
blocks (`--chdir`, `--unsetenv`) moved to the tail, which bwrap treats identically, so the flag multiset
is asserted separately — the move cannot hide a dropped or added flag.

**Three host realities the live docker probe surfaced**, none of which a design review would have caught:

| symptom | cause |
|---|---|
| `error while creating mount source path …: permission denied` | home is **NFS with root_squash**, so the daemon (root -> `nobody`) cannot traverse `0700` paths like `~/.local/share/claude`. `/scratch` is local ext4 where the daemon is real root, hence repo mounts worked. Fixed by staging unreachable sources onto the workspace — destinations unchanged, nothing made world-readable |
| `exec: "/sbin/docker-init": no such file or directory` | docker injects its reaping shim at `/sbin/docker-init`; `/sbin -> /usr/sbin`, and the host-mount model binds the host's `/usr` OVER it. `docker run --init ubuntu:24.04 true` succeeds standalone, so a probe cannot see this — it is the interaction. `--init` dropped; the shell is pid 1 and reaps its own children |
| duplicate mount destinations | bwrap resolves ties by LAST op; docker rejects duplicates outright. `_dedupe_last_wins` collapses them in the renderer, preserving exactly what `is_exposed` predicts |

`bwrap.available()` / `docker.available()` are **real probes, not `which`** — an installed-but-unusable
bwrap is precisely this host's situation, and a launcher that reported "sandboxed" while nothing was
isolated is the failure being guarded. The launcher refuses with that wording.

`tests/infra/test_sandbox_isolation.py`: the coverage guard is parametrized over both backends, the
vacuity guard runs on the plan, and a live docker probe verifies inside real containers that the
container runs as the invoking uid and a golden reads empty — for all six roster targets. A final test
FAILS (not skips) if no backend can isolate, so "the live probes all skipped" can never read as green.
**41 passed** in that file; affected buckets at their pre-existing baseline (4 failed / 3 errors, all
absent-chipyard/atlas plus a `design_pressure` CLI test that was already failing).
