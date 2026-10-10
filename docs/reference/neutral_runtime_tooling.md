---
title: Neutral contract-driven runtime tooling
kind: reference
status: current
owner: runtime
last_verified: 2026-10-10
related: [target_resolution, simulator_selection, aws_gsim]
code_refs: [src/merlin/runtime/backends/base.py, src/merlin/runtime/backends/chipyard_rocc.py, src/merlin/runtime/harness_render.py, src/merlin/targetgen/rocc/semantics.py, src/merlin/targetgen/rtl_checks_generic.py, src/merlin/targetgen/build_cache.py]
---

# Neutral contract-driven runtime tooling

An independent grader needs execution tools and a calling convention. It must
not depend on the compiler implementation being evaluated. Merlin's shared
Chipyard RoCC tooling binds explicit target data without loading a target support
module. Candidate lowering, tensor packing, tiling, scheduling, device kernels
and completion implementations remain candidate work.

## Selection

`runner.backend: chipyard_rocc` selects the shared family. The selected capability
contract is read through `TargetInfo.load_contract`, including an observed sealed
snapshot or explicit `MERLIN_TARGET_CONTRACT`. Existing facts are read without
regeneration, including an observed facts snapshot or `MERLIN_RTL_FACTS`.
Selection precedes executable provider discovery and is never cached by target
name. An invalid explicit selection refuses rather than loading legacy support.
Targets without this selector retain their existing provider behavior.

`runner.chipyard_rocc` version 1 pins a toolchain, configuration and nonempty
engine roster. Toolchain data names the compiler and linker script by path/hash,
fixed core runtime and header IDs by hash, load address, flags and entry stack
bound. IDs select core startup, console and small libc resources; arbitrary
source/include libraries cannot become compiler helpers through these fields.
Engine entries pin a binary, argv with exactly one ELF placeholder, environment,
working directory, timeout, console budget and diagnostic failure markers.
Supported engines are explicit Spike and gSIM selections. gSIM uses the existing
strict-v3 receipt checks against its actual binary and the selected FIRRTL.
Spike uses an explicitly pinned extension and provides functional evidence.
Metadata binding does not execute tools, build RTL or establish native readiness.

## Logical tensors

The version 2 `logical_pointer` harness ABI explicitly declares entry and
completion symbols, alignment, byte order, main convention and complete readback
transport. The shared backend currently admits B64 only; coherent-memory
rendering remains available separately and needs its existing trusted reader.
Inputs are exact dense row-major logical tensors; outputs are poisoned
before invocation and read completely. Device padding, swizzling, im2col and
operation-derived operands are refused. Existing typed invocation/counter plans
can be used where their original requirements are satisfied; no timing authority
is inferred from a successful call or a reported number. Its selected complete
readback policy reaches ordinary build receipts and full output-roster checks,
including callers that omit an explicit policy argument.

No vendor kernel header, C snippet, numerical reference, lowering routine or
preferred instruction sequence is supplied by this ABI. A completion symbol is
an ABI obligation, not a supplied accelerator fence implementation.

## Instruction interpretation

`rocc_operand_roles` version 1 names each instruction code, its semantic label,
and any operand register bundles. Codes must exist in the selected hardware
decode table. Named bundles provide exact field offsets and widths from the
selected register-layout facts. Unknown widths, overlapping fields, absent
bundles and duplicate interfaces refuse. Unknown operands retain unknown fields;
no pointer address or tensor layout is guessed. Representable signed LLVM
constants retain their exact two's-complement register bits.

The generic `rtl_checks` list currently admits only `decode_clean` and
`legal_funct`. A partial hardware decode table cannot prove universal legality.
Memory bounds and temporal protocol checks are not implemented here. They must
not be substituted with preferred schedules, tile coverage, reuse or command
balance assertions. Complete numerical output and existing linked-ELF policy
checks remain separate mandatory evidence.

## What these tools establish

Source controls check selection, data custody and refusal behavior. A worker
still needs its actual toolchain, simulator receipt, nonzero numerical smoke and
fresh-client isolation check before experiments. These shared tools do not
qualify the native worker, prove complete Phase 0 coverage, or certify end-to-end
compiler accuracy or performance.
