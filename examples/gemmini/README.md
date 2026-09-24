# Gemmini compiler-generation example

Start here for the public inputs and phase handoffs. The single functional
definition is [`experiment.yaml`](experiment.yaml), registered as
`gemmini-functional` in the [experiment catalog](../../experiments/catalog.yaml).
This example is not a prequalified compiler or a self-contained hardware setup.

| Step | Authored inputs and instructions | Generated result |
| --- | --- | --- |
| Target setup | [`target/`](target/README.md): descriptor, reference contracts, explicit OOT support selection | Extracted facts and tool qualification, kept outside examples |
| Phase 0: hardware-guided test generation | [`phase0/`](phase0/README.md): public coverage recipe | Run-owned capsules, then an explicitly reviewed corpus release |
| Phase 1: functional compiler generation | [`phase1/`](phase1/README.md): prompts and public runtime harness | Frozen compiler submission and separately attributed certification |
| Phase 2: performance optimization | [`phase2/`](phase2/README.md): selecting frozen inputs and the shared templates | Optimization runs and evidence tied to the exact functional compiler |
| Whole-model inspection | [`whole-model/`](whole-model/README.md): lowering an existing capture and inspecting IR | Lowering stages and tensor inspection payloads; not accelerator certification |
| Artifact navigation | [`artifacts/`](artifacts/README.md): map each generated file to its input and next phase | Nothing generated is stored or committed in the example |
| Published compiler smoke | This page, below: run the pinned OOT compiler away from the Merlin checkout | Parsed interface, Gemmini lowering, command buffer and LLVM-dialect artifact; not simulator or whole-model qualification |

Install Merlin and `merlin-experiments`, then provision the explicit OOT provider,
toolchains and descriptor-selected resources described in target setup. Some
resources still require a provisioned source workspace. Phase 1 and the measured
Phase 2 coordinator have installed commands. The measured-claims catalog route
uses the installed managed envelope; this example's RTL-checks treatment also
uses the installed controller. Model-portfolio mode uses the installed portfolio
launcher with an explicit deployment record. See the phase guides.
Private holdouts,
goldens and credentials must never be copied into this public example.

These commands inspect configuration without starting an agent or simulator:

```sh
merlin experiment inspect gemmini-functional --phase 0
merlin experiment inspect gemmini-functional --phase 1
merlin experiment preflight gemmini-functional --phase 1
```

Preflight does not prove engine or hardware readiness. Run Phase 0 separately,
review and seal its release, then explicitly select that release for Phase 1.
Do not use a combined invocation to bypass review. Phase 2 requires the exact
frozen Phase 1 evidence, not merely a successful process exit.

## Find stored work

```sh
merlin experiment runs --target gemmini
merlin experiment status /absolute/path/to/orchestration-run
```

Discovery reads orchestration records beneath the configured `out/runs/`; it does
not search every legacy artifact. For an explicit noncanonical run directory, use
`status` directly. The status records link to native engine attempts; AET remains
the owner of native run accounting. Neither command resumes execution.

Keep authored definitions in source control, new execution state under the
configured output root, and historical receipts unchanged. Use a newly frozen
run after changing source or inputs; changing paths does not renew old evidence.

## Use the published compiler without Merlin

The published `gemmini_xdsl_rtl_v0` branch of
[gemmini-mlir](https://github.com/ucb-bar/gemmini-mlir) is a separate compiler
payload. To reproduce a small offline lowering smoke from the Merlin checkout,
copy the example input first and run from a temporary directory with a clean
environment. The commit check deliberately refuses a silently updated branch.

```sh
(
set -eu
qual_root=$(mktemp -d /tmp/merlin-gemmini-oot-XXXXXX)
git clone --branch stable/gemmini_xdsl_rtl_v0 \
  https://github.com/ucb-bar/gemmini-mlir.git "$qual_root/compiler"
test "$(git -C "$qual_root/compiler" rev-parse HEAD)" = \
  390623a67db81bcf595ec3805f0921f1fb69e378
cp merlin/contract/examples/g0_matmul.interface.mlir "$qual_root/input.mlir"
(
  cd "$qual_root"
  env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
    "$qual_root/compiler/gemmini-opt" --verify-diagnostics input.mlir
  env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
    "$qual_root/compiler/gemmini-opt" --convert-iface-to-gemmini input.mlir > lowered.mlir
  env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
    "$qual_root/compiler/gemmini-opt" --convert-iface-to-gemmini \
    --emit-command-buffer=commands.json input.mlir
  env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONDONTWRITEBYTECODE=1 \
    "$qual_root/compiler/gemmini-opt" --convert-iface-to-gemmini \
    --emit-target-artifact input.mlir > artifact.mlir
)
printf 'Inspection files: %s\n' "$qual_root"
)
```

This exact published commit passed those four commands on 2026-09-23 from an
independent local clone, with no Merlin import path or checkout environment.
`lowered.mlir` contained `gemmini.matmul`, `commands.json` contained
`MATMUL_RESIDENT`, and `artifact.mlir` contained an LLVM-dialect function with
Gemmini inline assembly. Its bundled xDSL Python is enough for this smoke;
LLVM/CIRCT, framework capture, simulator, numerical correctness and historical
certificate renewal were **not** exercised. Current publication policy treats
the old embedded certification claim as historical; execution alone does not
make a new verified compiler certificate.
