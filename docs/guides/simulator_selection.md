---
title: Selecting and checking a simulator
kind: guide
status: current
owner: runtime
last_verified: 2026-10-06
related: [phase0_specification, target_resolution, reproducing_whole_model_on_rtl]
code_refs: [src/merlin/targetgen/gsim_emulator.py, src/merlin/targetgen/program_engine_policy.py, src/merlin/targetgen/program_oracle.py]
---

# Selecting and checking a simulator

Simulator availability, numerical agreement, and hardware/source qualification
are separate checks. A process exiting successfully is not proof that it wrote
every result or executed the requested accelerator operation.

The model compiler evaluator requires a network-isolated `bwrap` worker. Its
preflight tests network-namespace construction as well as the base sandbox;
some hosts permit the latter but deny loopback setup with a `NETLINK_ROUTE`
error. That outcome is `sandbox inoperable (netns_denied)`, not a compiler
refusal or an accelerator fallback. Move qualification to a worker that can
construct the required sandbox; do not remove `--unshare-net` to obtain a pass.

## Select the support package and engine explicitly

Choose a target-owned support provider with `MERLIN_TARGET_PATH`; compiler
candidates alone do not necessarily supply an execution backend. Use the
provider's documented toolchain and harness. In particular, a generated
parameter header from another configuration can compile successfully and
produce incorrect results on the selected hardware.

Program-driven engines consume assembled instruction words and memory regions.
Their directory contains `<engine>_run.py` exposing `run_program`. ELF-driven
engines consume a linked executable; GSIM's conventional binary is `emulator`.
These are different invocation interfaces, not interchangeable files.

Without an explicit override, the generic resolver checks a run-local home,
then the repository-anchored installed home:

```text
out/build/rtl_engines/<target>/<engine>/
```

Setting `MERLIN_OUT_ROOT` for a run does not require reinstalling machine tools
under that run's directory. Program-wrapper discovery follows the same rule as
binary discovery. An explicit missing path is an error, not permission to select
a different engine.

For program engines, `MERLIN_EXT_<TARGET>_GSIM` or
`MERLIN_EXT_<TARGET>_VSIM` names the exact wrapper directory. For ELF-driven GSIM,
`MERLIN_GSIM_EMU_<TARGET>` names the binary; a provider may also retain a documented
historical spelling. Keep machine-specific paths in process configuration or
the local, untracked `.env`, not in shared library code.

## Inspect the choice before execution

```python
from merlin.targetgen.program_engine_policy import select_rtl_engine

selection = select_rtl_engine(target)
print(selection)
```

For a provider using the declared Chipyard interface, inspect
`merlin.targetgen.oracle_policy.select_chipyard_engine(target)` instead.
`MERLIN_REQUIRED_RTL_ENGINE` constrains that Chipyard selection; unavailable
required engines cannot silently fall back to another engine.

GSIM receipts bind the resolved bytes. Older receipts primarily establish
binary/FIRRTL identity; a v3 build receipt additionally checks its recorded
artifact, tool, support-input and command commitments. An adoption record is
weaker than a built-and-bound receipt. `MERLIN_GSIM_REQUIRE_RECEIPT=1` refuses
unreceipted/adopted GSIM wrappers, including explicitly registered wrapper homes.
It does not make older receipts stronger or establish that the simulator's
FIRRTL equals a separate Phase 0 source selection.

That source comparison is automatic only when the run names its selected facts.
With `MERLIN_RTL_FACTS` pointing at a facts file (frozen runs set it to their
verified input snapshot), the Chipyard engine selection and the capsule GSIM
adapter require a *bound* build receipt whose `firrtl_sha256` equals the facts'
FIRRTL digest; an unbound receipt or a different digest makes GSIM unavailable
with that reason. Without `MERLIN_RTL_FACTS`, GSIM keeps its availability
semantics, but its selection reason records that source identity is unverified
and cannot be cited as evidence of it.

Compare the actual engine's FIRRTL digest and configuration with the selected
Phase 0 facts. Do not infer equivalence from a shared target name or array size.
If an engine uses another retained elaboration, derive a new facts bundle from
those exact bytes using the [source-production workflow](phase0_specification.md).
Keep the original facts and captures intact. If build inputs or the original
elaboration are unavailable, record that limitation rather than inventing a
build receipt.

The generic producer can derive an exact instance hierarchy directly from the
selected FIRRTL circuit, without a separately authored hierarchy file:

```sh
python -m merlin.targetgen.rtl.source_selection \
  --target "$TARGET" --generator "$GENERATOR" --config "$CONFIG" \
  --firrtl /absolute/selected/model.fir --core-root "$CORE_MODULE" \
  --firtool /absolute/toolchain/bin/firtool \
  --output out/artifacts/rtl-audits/engine-source-1
```

It records the exact circuit hierarchy root, requested compute-module closure,
source/tool digests and FIRRTL-to-HW command. `--hierarchy PATH` remains available
when selecting an explicitly authored subtree root; missing/mismatched modules
are not repaired through guessed aliases. The generated `source-selection.json`
is the input to `circt_introspect --source-bundle`. Keep the source configuration
explicit and retain any permitted metadata-only annotation preparation in the
production receipt. This establishes extraction-source consistency, not the
missing simulator build provenance.

## Require a numerical smoke, not just an availability probe

For a self-hosted-ISA provider, `program_oracle.emit_bundle` runs the provider's
declared program emitter in its model environment. The generated bundle contains
the instruction words, encoded input regions, output layout and independent
program golden. Output directories may be relative to the caller; they remain
under that run root even though the model assembler uses its own checkout as cwd.
The emitter's `runner.program_emitter.path` is resolved inside the explicitly
selected support provider; a missing declaration, escaping path or missing file
refuses instead of falling back to an in-tree target-specific script. Optional
declared string arguments are provider policy, not an inferred ISA encoding.

Use a program that exercises nonzero operands, the actual compute instruction,
memory transfers and a declared termination. Run it with a bounded cycle budget
and wall-clock deadline through the existing execution adapter. A program-driven
GSIM run holds one of a fixed number of per-user GSIM slots for the whole
simulation; when every slot is busy it waits up to its timeout and then fails,
rather than oversubscribing the host. Check:

- termination and absence of design assertions or traps;
- complete decode by the functional model: the program oracle, for grading and
  for the debugger alike, refuses a run whose model substituted any submitted
  instruction it does not support, or that reports no decode coverage at all;
- every output region's exact shape, byte length and precision;
- the complete result against the independent golden;
- the simulator binary/wrapper, input executable or words, and source digests;
- unchanged selected inputs before and after execution.

An engine-agreement comparison supplements the independent golden; two engines
can agree on the same incorrect harness. A numerical smoke qualifies its tested
operation, shapes and value domain only. It is not whole-model validation,
overflow coverage, full instruction conformance, or universal timing equivalence.
Report the source-qualified scope separately from historical diagnostic results.

For a selected core that can be lowered through CIRCT's Arcilator, the
[Gemmini](../../examples/gemmini/target/README.md) and
[Atlas](../../examples/atlas/target/README.md) examples show a narrower native
numerical observation. Their target-owned runners use the existing OOT protocol
drivers; the shared observation producer only pins and verifies exact RTL,
HW/Arc/LLVM/native stages, tools, driver source, case inputs, independent golden
and observed output. A saved bundle's runner with --replay-bundle checks all
pinned bytes and re-executes from those frozen files. Atlas additionally checks
that its compiled core is exactly the selected tile's module closure. This does
not create a GSIM receipt, qualify an untested readout format, or extend a
core-only result to its surrounding SoC.

Persist producer-generated inputs, intermediate MLIR/assembly, executable,
console, output tensors and receipts under a new run/artifact directory. Do not
hand-edit the simulator's source, a frozen compiler payload, or existing results
to make a qualification pass.
