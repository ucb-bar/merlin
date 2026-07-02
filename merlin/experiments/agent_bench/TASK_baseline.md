# Task — generate an out-of-tree Gemmini target-backend package (baseline)

You are producing a **target-backend package** that satisfies the experiment ABI **v0.1**. Your
working directory is this sandbox; everything you need is here. **Do not** read or access anything
outside this sandbox except the toolchains explicitly listed below.

## What to build

A self-contained package in `submission/` that the grader can build and invoke **only through CLI
entrypoints** (it is never imported). Read the contract first:

- `bench_contract/README.md`, `interface_grammar.md` — the `merlin_iface` input grammar.
- `bench_contract/mlir_oot_backend_contract.yaml` — the manifest + the four entrypoints + the
  `gemmini_kernel` ABI.
- `bench_contract/command_buffer_abi.yaml` + `schemas/command_buffer.schema.json` — the cb you emit.
- `bench_contract/oracle_runner_contract.yaml` — how your lowered kernel is run + the `OUT/METRIC/
  DONE` output the runner-owned harness prints (you do NOT write the harness).
- `bench_contract/integrity_policy.md` — the rules below, enforced by an integrity scan.
- `bench_contract/examples/{g0_matmul,g1_relu,g2_acc_scale}.interface.mlir` — public inputs;
  `expected_command_buffer_g0.json` — the golden cb for g0.

Your `submission/manifest.yaml` declares `artifact_type: mlir_oot_target_backend`, `language`
(python or cpp), `integrity_exempt: false`, the entrypoints, and (if C++) a `build` block using
the `{package}`/`{mlir_dir}`/`{llvm_dir}` tokens. Implement the four entrypoints:

1. `parse` — parse/verify the interface.mlir (nonzero exit on error).
2. `lower_interface_to_target` — emit non-empty target MLIR (stdout).
3. `emit_command_buffer` — write a schema-valid `command_buffer.json` (must match the golden for g0).
4. `lower_target_to_llvm` — emit LLVM/RoCC MLIR (stdout) defining `llvm.func @gemmini_kernel(
   weight*, lhs*…, out*…)` per the kernel ABI; it must compile and be **bit-exact** on the oracle.

## Hard rules (violations fail closed)

- The package must emit **both** a command buffer AND a lowered LLVM/RoCC kernel. A C compute
  kernel alone, or target MLIR with no command buffer, is the wrong artifact class.
- **No harness/reference imports**: no `import merlin` / `from merlin`, no `merlin.runtime.
  reference`/`simulator`, no `reference_outputs`. Do not read any file outside this sandbox.
- Integer workloads ⇒ exact equality. Your kernel must reproduce the reference, not approximate it.

## Success criteria

- **Required:** g0 and g1 (public) pass through the grader (K1–K6; spike + verilator three-way
  bit-exact when available).
- **Hidden grading (operator):** g0/g1/g2 variants with different data — your kernel must
  generalize (be data-independent), not hardcode outputs.
- **Stretch:** g2 (`acc_scale` float requant → i8).

## How to iterate

Self-check against the PUBLIC examples only:

```
bash grade.sh submission g0 spike       # fast functional bootstrap
bash grade.sh submission g0 verilator   # RTL certification
```

A run prints `status: pass/fail`, the per-entrypoint result, and (on failure) the plane +
category. Iterate until g0/g1 pass.

## Allowed toolchains (outside the sandbox, but explicitly permitted)

- LLVM/MLIR 23 install: env `MERLIN_MLIR_INSTALL` (the grader injects `{mlir_dir}`/`{llvm_dir}`).
- RISCV gcc, spike, Verilator — invoked for you by `grade.sh`.
- Public Gemmini ISA headers + the MLIR `examples/standalone` OOT template (copied into `docs/`).
