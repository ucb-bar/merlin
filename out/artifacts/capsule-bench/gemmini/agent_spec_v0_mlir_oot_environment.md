# agent_spec_v0_mlir_oot — environment snapshot (2026-06-16)

Frozen toolchain + repo state for the Merlin-side evidence package. Machine-readable copy:
`generated_targets/gemmini/agent_spec_v0_mlir_oot/certification/environment_snapshot.yaml`.

## Toolchain

| component | version / SHA |
|---|---|
| LLVM/MLIR | 23.0.0git @ `a47bddccec30` (`third_party/llvm-install`) |
| clang | 23.0.0git |
| riscv64 gcc | 13.2.0 (`gc891d8dc23e`) |
| spike | 1.1.1-dev (`--extension=gemmini`) |
| Verilator | 5.022 (2024-02-24) |
| python / xdsl | 3.13.10 / 0.65.0 |
| Chipyard | `6a9b4cd95081100dc932e422cc52777bc9a98b2d` |
| Gemmini | `8c3f9923a44a2fe2c7930587be297d6d4f8c09ca` |

## Repo state

- Merlin HEAD at snapshot: `f684ab53ed6d227676ec18dfb71338a185dedc5c` (branch `feature/kernel-policy-mining`).
- **SHA drift (honest note):** the package was authored at HEAD `2a8500441f…`; HEAD has since advanced
  because a **concurrent session** committed unrelated work. The `agent_spec_v0_mlir_oot` sources were
  not changed by that move.
- **Nothing is committed.** The package and the entire Experiment-ABI substrate (`bench_contract/`,
  `oot_runner.py`, `targetgen/contract/`, `generated_targets/`, `results/`, `runs/`) are untracked
  working-tree files.
- 9 tracked files are **modified**, all by the concurrent session (llvmlower `lower.py`/`pipeline.py`;
  runtime `reference.py`/`simulator.py`/`tensor.py`; xdsl `lowering/*`; `.gitignore`). **None** were
  touched by this evidence work.

## Correctness-dependency note

The three-way certification gate (`oracle == reference == simulate`) rides on
`runtime/{reference,simulator,tensor}.py`, which are currently modified-in-tree by the concurrent
session. Mitigant: a pass requires the **Verilator RTL** oracle to agree as well, so RTL is the
ground-truth anchor — a buggy reference/simulator would make the three-way *disagree* (fail), not
silently pass.
