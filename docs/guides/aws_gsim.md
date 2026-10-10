---
title: Provisioning Merlin's Gemmini gSIM on a Linux worker
kind: guide
status: current
owner: runtime
last_verified: 2026-10-10
related: [getting_started, target_resolution, simulator_selection, phase0_specification]
code_refs: [src/merlin/targetgen/gsim_emulator.py, build_tools/scripts/package_worker_inputs.py, packages/merlin-experiments/src/merlin_experiments/phase0/rtl_intake.py, packages/merlin-experiments/src/merlin_experiments/runner.py, packages/merlin-experiments/src/merlin_experiments/cli.py, packages/merlin-experiments/src/merlin_experiments/phase1/timing.py, packages/merlin-experiments/src/merlin_experiments/phase1/preflight.py, packages/merlin-experiments/src/merlin_experiments/phase1/canary.py, packages/merlin-experiments/src/merlin_experiments/phase1/providers/codex_runtime.py]
---

# Provisioning Gemmini gSIM on a Linux worker

This is a source-and-input setup procedure, **not** an AWS qualification result. Provision the
worker, storage and access controls yourself. Use an x86-64 Linux worker for the documented native
model; an x86-64 emulator does not run natively on an ARM/Graviton worker. Keep private inputs,
credentials and generated simulator artifacts outside Git and candidate-visible workspaces.

The native fork procedure below is retained inspection material. Ordinary catalog
experiments require verified hardware inputs, independent target support and the
selected session's actual startup checks. These build commands alone do not admit
an engine or its helpers. The additional component qualification route is described
in [fresh compiler origin](../design/fresh_compiler_origin.md). The handwritten
support provider and copied kernel headers have been removed from Merlin.

## Clone and install the source stack

Install Python 3.12+, `uv`, Clang 19 or newer (the native Gemmini path has used Clang 23), GNU Make,
Flex with `FlexLexer.h`, Bison and GMP development files. Follow [getting started](getting_started.md)
to install Merlin and `packages/merlin-experiments` into a Python environment:

```sh
git clone --branch feat/compiler-readiness https://github.com/ucb-bar/merlin.git merlin
cd merlin
git rev-parse HEAD  # record and review the full source commit before freezing a run
uv venv
uv pip install -e '.[dev,xdsl,targetgen]'
uv pip install -e packages/merlin-experiments
```

Installing Python packages does not install LLVM/MLIR, CIRCT, the guest RISC-V compiler or model
checkpoints. Use the [LLVM toolchain guide](llvm_toolchain.md) and the selected provider's toolchain
configuration. If regenerating frontend captures, also follow the [model2MLIR guide](model2mlir.md)
and record its source and framework versions independently.

Select independently derived and qualified target support explicitly. A Merlin
clone supplies shared orchestration and runtime mechanisms; target implementation
and the private handwritten reference remain out of tree.

Clone the [public gSIM fork](https://github.com/copparihollmann/gsim) at an exact commit of its
`merlin` branch (not `master`). The published Merlin integration commit below is an example pin;
record the full commit actually selected for a new build:

```sh
git clone --branch merlin https://github.com/copparihollmann/gsim.git gsim
cd gsim
git checkout --detach f38de1704dac25b540d1552fbe1d23fc48463212
git rev-parse HEAD
export GSIM_CLANGXX=/absolute/selected/toolchain/bin/clang++
make -j2 CXX=./cxxwrap_portable.sh build-gsim
make CXX=./cxxwrap_portable.sh ext-clock-output-check dynamic-clock-check
"$GSIM_CLANGXX" -std=c++17 chipyard_harness/terminal_dump_selftest.cpp \
  -o build/terminal-dump-selftest
build/terminal-dump-selftest
python3 chipyard_harness/test_build_inputs.py
```

The [fork's Merlin build contract](https://github.com/copparihollmann/gsim/blob/merlin/MERLIN.md)
documents `cxxwrap_portable.sh`, the build prerequisites and the native Chipyard harness. Its old
`cxxwrap.sh` is host-specific receipt input; do not reuse it on a new worker or modify an old receipt
to make a new build appear identical.

## Build and select a new native model

Supply the **complete selected** `TestHarness` FIRRTL, a matching Chipyard checkout with initialized
TestChipIP and FESVR, the selected compiler and an unused output directory. Install Merlin core plus
`merlin-experiments` in the Python used to run the builder. From the pinned gSIM checkout:

```sh
/absolute/merlin-venv/bin/python chipyard_harness/build.py \
  --firrtl /absolute/selected/TestHarness.fir \
  --emitter "$PWD/build/gsim/gsim" \
  --compiler "$GSIM_CLANGXX" \
  --chipyard /absolute/selected/chipyard \
  --out /absolute/fresh/out/build/rtl_engines/native-gsim \
  --comb-extmod EICG_wrapper --optimization 2 --jobs 2 --merlin-receipt
```

For an adopted-FIRRTL rebuild, a verified input bundle may supply just the native builder's selected
Chipyard dependency closure instead of the whole checkout. Its root must contain
`generators/testchipip/src/main/resources/testchipip/csrc/`,
the complete `.conda-env/riscv-tools/include/` tree and `.conda-env/riscv-tools/lib/libfesvr.a`.
Pass that root as `--chipyard`; it cannot elaborate RTL or replace the recorded hardware sources.
FESVR can include sibling headers, so copying only `include/fesvr/*.h` is insufficient. The builder
binds the complete selected regular-file include tree and checks it again after compilation.

The optimization level here follows the fork's documented example; changing it changes the built
binary and requires its own receipt and qualification. The builder's `native/build_receipt.json` and
`native/emulator` must travel together. Building from supplied FIRRTL binds those bytes to the model;
it does **not** prove which RTL revision originally elaborated the FIRRTL. Retain that source recipe,
Chipyard/toolchain revisions and the generated ABI header separately.

From the Merlin checkout, select admitted independent support, exact model and facts explicitly:

```sh
export MERLIN_TARGET_PATH=${MERLIN_INDEPENDENT_TARGET_SUPPORT:?independent runtime support required}
export MERLIN_EXT_GSIM=/absolute/pinned/gsim
export MERLIN_CHIPYARD=/absolute/selected/chipyard
export MERLIN_EXT_CHIPYARD="$MERLIN_CHIPYARD"
export MERLIN_GSIM_EMU_GEMMINI=/absolute/fresh/out/build/rtl_engines/native-gsim/native/emulator
export MERLIN_GEMMINI_GSIM_EMU="$MERLIN_GSIM_EMU_GEMMINI"
export MERLIN_REQUIRED_RTL_ENGINE=gsim
export MERLIN_GSIM_REQUIRE_RECEIPT=1
export MERLIN_RTL_FACTS=/absolute/selected/facts.json
```

Keep both binary override spellings identical. Set the selected guest RISC-V toolchain and LLVM/MLIR
paths for the intended workflow; do not inherit unreviewed machine defaults. Merlin checks the
receipt's binary identity and, with `MERLIN_RTL_FACTS`, its FIRRTL digest against selected facts.
Run a bounded nonzero numerical smoke with complete outputs and accelerator-activity evidence before
using a new model for grading. The per-user native gSIM admission limit is five concurrent processes;
do not bypass the guard to make a worker appear ready. A passing build or smoke is not whole-model,
numerical-all-domain or hardware-equivalence certification.

## Transfer explicit private inputs, if needed

The worker-input delivery tool packs **only** paths named by the operator. Choose a complete,
reviewed input roster; it does not discover private captures, approve a policy, or seal a release.
The archive is private and belongs in the generated `out/artifacts/delivery/` product, not Git or an
agent-visible directory. For example, from the Merlin checkout:

```sh
.venv/bin/python build_tools/scripts/package_worker_inputs.py pack --target gemmini \
  --input rtl-facts=/absolute/selected/facts.json \
  --input frozen-inputs=/absolute/selected/private-inputs
```

Save the resulting product's `worker-inputs.tar` and `archive.json`. Transfer the archive through
an authorized private channel and its `archive_sha256` through an independently trusted channel.
On the receiving worker, verify every member before optional extraction to a **new** private path:

```sh
.venv/bin/python build_tools/scripts/package_worker_inputs.py verify \
  /absolute/private/worker-inputs.tar --sha256 FULL_TRUSTED_ARCHIVE_SHA256 \
  --extract /absolute/private/new-selected-inputs
```

Verification checks byte identity, membership and safe extraction; it does not authenticate the
sender by itself or upgrade old frozen evidence. Any new worker or simulator selection needs fresh
host admission, source/facts/receipt checks and a newly reviewed freeze. Never restamp historical
receipts or expose private model inputs to a compiler author.

A private-input YAML file is not its input closure: transfer its referenced model captures, weights,
selected SDK/toolchain and attestation inputs separately, or regenerate them through the normal
capture workflow. A small simulator bringup bundle alone does not make an EL4 full-model gate ready.

## Prepare the catalog Phase 0 and Phase 1 handoff

Use a reviewed experiment definition with explicit recipe, conformance and synthesis
inputs, verified evidence mode, exact facts and capability contract, and an
operator-owned hidden cohort. The retained diagnostic example cannot be promoted
by changing its status. Select independently reviewed support with
`MERLIN_TARGET_PATH`; Merlin's metadata-only example does not supply its executable
backend. Keep handwritten references and final-validation inputs outside public
derivation and author grants.

The ordinary catalog route uses `capsule_derivation` and `capsule_bench`.
Explicit component-only preparation and `FreshPhase1Inputs` are a separate route;
their diagnostic ledgers are not the catalog launch prerequisites. Both routes
retain their own correctness and independence gates.

Set the variables below to the reviewed definition and fresh destinations. Run
Phase 0 separately from Phase 1, using the same definition and selections through
inspection, preflight and execution:

```sh
merlin experiment inspect "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment preflight "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment run "$SPEC" --phase 0 --run-dir "$P0"
merlin experiment corpus coverage "$P0" --spec "$CONFORMANCE"
merlin experiment corpus prepare "$P0" --generated-only --output "$RELEASE"
merlin experiment corpus inspect "$RELEASE"
```

The release belongs beneath the configured `out/artifacts` root. Supply
`--private-baseline` to `prepare` when selecting a separate private cohort. After
actual operator review, seal the exact inspected digest:

```sh
merlin experiment corpus seal "$RELEASE" \
  --expected-digest "$REVIEW_DIGEST" \
  --reviewed-by "$OPERATOR" --review-note "$REVIEW_NOTE"
merlin experiment preflight "$SPEC" --phase 1 --run-dir "$P1" \
  --corpus-seal "$RELEASE/private/seal.json" \
  --bundle-manifest "$RELEASE/payload/experiment/input_bundles/$BUNDLE/input_bundle_manifest.yaml"
```

Choose `BUNDLE` from this release's regenerated inputs. The seal and manifest must
belong to the same release; the runner derives the released descriptor and bundle
identity. Provider/model overrides, when selected, must remain identical through
inspection, preflight and execution. See the
[reviewed handoff](../../experiments/README.md#reviewed-phase-0-handoff) for the
available flags.

Catalog preflight reports `engine_readiness: not_executed`. It checks configuration
and recorded timing bindings without launching a simulator or an author. The
execution worker still needs a genuine target/config/engine/binary-bound Chipyard
oracle timing record, usable compiler tools, exact bundle snapshots and successful
native sandbox checks. Readiness records the actual passed L3 grade's host wall time
on the selected engine. Version 2 timing also binds gSIM's strict receipt and selected
FIRRTL facts; legacy Verilator records apply only to a Verilator selection. These
observations size timeouts, not accelerator cycle predictions. Producing one requires
an explicitly selected independent operator-only probe backend; a simulator binary
alone cannot provide it.

To exercise ordinary startup on the worker and stop before authoring, invoke the
installed continuation with the released descriptor, seal and regenerated bundle:

```sh
python -m merlin_experiments.phase1 \
  --descriptor "$RELEASE/payload/experiment/target_experiment.yaml" \
  --repo "$OPERATOR_ROOT" --bundle "$BUNDLE" \
  --bundle-manifest "$RELEASE/payload/experiment/input_bundles/$BUNDLE/input_bundle_manifest.yaml" \
  --corpus-seal "$RELEASE/private/seal.json" --oracle-timing "$ORACLE_TIMING" \
  --level EL4 --sandbox bwrap --run-id "$FRESH_PREFLIGHT_ID" --preflight-only
```

Retain the same provider, treatment and optional tool selections as the intended
experiment. The explicit seal must agree with any `MERLIN_CORPUS_SEAL` selection.
The private `preflight_result.json` records startup completion with
`formal_complete: false` and `provider_started: false`. It cannot certify a compiler
or fresh-client isolation. Use another fresh run ID for the author experiment;
preflight-only sessions cannot resume into authoring.

For a Codex client, select `--driver codex`, an explicit `--model`,
`--codex-binary`, `--codex-auth-source` and `--codex-home-root` for all three
installed invocations: preflight, canary and ordinary authoring. Use
`--codex-canary --round-timeout 300` with another fresh run ID for the bounded
client check. The [package instructions](../../packages/merlin-experiments/README.md#sealed-fresh-codex-client-check-aws-execution)
show the commands. They use the same runtime selection and sandbox composition;
the parent home root can be reused, while each round home must be fresh. The
credential file is mounted for the client, denied to its tools, and never copied
or hashed. The canary requires the actual completed fixed public/tool probe,
rejects additional tool activity, and preserves the sealed task. Its private
result establishes only the observed selected-client check. The outer catalog's
legacy default client is not covered by a differently selected canary.

## Launch readiness boundary

Handoff is ready when one pinned source revision installs, the selected verified
Phase 0 inputs produce a reviewed release covering its mandatory obligations, an
independent nonzero probe checks complete outputs and rejects an incorrect output
on the selected engine, and ordinary native preflight passes on the execution
worker without bypasses. A short fresh-client canary must separately verify usable
tools and inaccessible protected inputs. Retain exact selections, budgets, digest
and logs. This establishes launch readiness, not compiler correctness or final
performance.

The additional component qualification ledger, Phase 2 calibration and final
whole-model performance comparison remain their own acceptance work. They are
not additional prerequisites of the ordinary catalog launcher. After this handoff,
expand preparation only when the selected run reports a concrete missing requirement.

## Worker isolation and access

Keep inbound access restricted to your trusted SSH source addresses or use your managed access
service. Do not publish dashboards or container ports on all interfaces. Keep API credentials out
of bundles, source history and agent grants. EL4 additionally requires its author-driver
authentication and a successful Bubblewrap user/network-namespace isolation check; do not remove
network isolation to bypass a failing check. Chia execution requires Merlin's managed-worker cleanup
contract, not an arbitrary unmanaged host. Re-run these checks on the actual AWS worker before
launching experiments; local relocation checks do not qualify a different operating system or host.
