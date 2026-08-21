# Vortex target (prototype)

Vortex (`vortexgpgpu/vortex`) — a RISC-V **SIMT GPGPU** — as a Merlin target. Status: **prototype**,
certification **uncertified** (functionally verified on both simulator tiers; perf is not RTL-certified).

Vortex is the first target here that is **not** tensor-resident. There is no systolic mesh, no
weight-stationary residency and no accumulator file: compute is ordinary `arith`/FP instructions on a
programmable core. So the tensor-resident vocabulary (`pack`/`matmul`/`commit`/`evict` over
`resident_tensor`/`accumulator`) does not apply, and neither does the command-buffer plane — the
`simt` family profile drops `emit_command_buffer` and adds `optimize_interface`, because a programmable
core executes a compiled kernel rather than consuming a command stream.

The accelerator-specific surface is exactly three things:

1. **CUSTOM0** (opcode `0x0B`) — two op families keyed by `funct7`: `funct7=0` is warp/thread control
   (`tmc`/`wspawn`/`split`/`join`/`barrier`/`pred`/`wsync`), `funct7=1` is the cooperative-thread family
   (`vote_all|any|uni|ballot`, `shfl_up|down|bfly|idx`). `funct3` alone is ambiguous across the two.
2. **CTA identity/geometry CSRs** — `thread_id.x 0xCD3`, `block_id.x 0xCD6`, `block_dim.x 0xCD9`, …
3. **A hardware Kernel Management Unit (KMU)** that dispatches every `(block, thread)` coordinate.
   Consequence worth internalizing: **a correct kernel body may emit no CUSTOM0 op at all** — the
   hardware launches the coordinates and the harness startup sets the thread mask. Reading the identity
   CSRs is what a coordinate-blind scalar loop cannot fake, which is why `CTA_CSR` (not `TMC`/`WSPAWN`)
   is the anti-scalar-collapse signal in the coverage gate.

## `contracts/target_contract.yaml` is GENERATED — do not hand-edit

Unlike `gemmini`/`saturn`/`toy_npu`, whose contracts are hand-authored, this one is emitted from
`merlin.targetgen.capability_manifests.vortex_manifest()`:

```
python -c "from merlin.targetgen import capability_manifests as cm; cm.write('vortex')"
```

The Python function is the source of truth and carries the per-field rationale that YAML emission drops;
`tests/targetgen/test_vortex_target_contract.py` asserts regeneration is byte-identical, so an edit here
is reverted by the next `write()` and caught by the suite. Everything the `simt` family profile already
supplies (`encoding_required=False`, the `simt_coverage` trace gate, `rtl_tiers=("L3",)`, the
linalg-input entrypoint set) is **not** restated in the contract — see `targetgen/families.py`.

Two deliberate omissions:

- **No `dialect_plan.yaml`.** The target dialect is a design decision the *agent* owns — the same stance
  `gemmini` takes. Nothing here pre-defines that vocabulary.
- **No `rtl_sim_config`.** Vortex's geometry is a build parameter, so "which machine" is an experiment
  decision, not a target fact; the frozen macro lives in the experiment descriptor
  (`experiments/vortex_capsule_bench_v0/target_experiment.yaml`, `hardware_spec.config.macro`), which
  declares itself the single source of truth for it.

## Oracle ladder

Vortex ships its own two-tier ladder; no chipyard is involved (`toolchain.sim_via: vortex`).

- **L2 `simx`** — functionally complete, cycle-*approximate*; `derived_from_rtl: false`. The default
  numeric oracle, fast enough to grade every capsule.
- **L3 `rtlsim`** — Verilator on the real RTL, cycle-exact; `derived_from_rtl: true`. The only
  RTL-derived tier, so it is the whole of `rtl_tiers`. Reserve for capsules that declare it.

Both are driven by `targetgen/vortex_oracle.py` (llvm-dialect MLIR → `mlir-translate` → **stock** clang
→ link against the curated harness → `vxbin.py` → host driver), not by Vortex's `ci/blackbox.sh`.

## Used by

- `targetgen/vortex_coverage.py` — the `simt_coverage` trace gate (scans the agent's **object**, never
  the linked ELF: the harness startup itself supplies `TMC`).
- `targetgen/vortex_oracle.py` — the L2/L3 adapters.
- `experiments/vortex_capsule_bench_v0/` — the three-arm compiler-bring-up experiment.
