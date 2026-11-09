---
title: "Independent linked instruction policy"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-11-09
related: [fresh_compiler_origin, component_compiler_convergence]
code_refs:
  - src/merlin/targetgen/contract/elf_admission.py
  - src/merlin_experiments/phase2/component_instruction_policy.py
  - src/merlin_experiments/phase2/component_instruction_audit.py
  - src/merlin_experiments/phase2/component_runtime_instruction_control.py
  - src/merlin_experiments/phase2/component_runtime_fixture.py
---

# Independent linked instruction policy

The coordinator selects exact public command symbols in a protected minimal
software policy before authoring. The policy contains no command numbers or
performance results. Live independent declaration and source predicate intakes
resolve those symbols in the selected function declaration span and corroborate
the values against actual public routing expressions. Configuration payload
aliases outside that span do not replace command identity.

The linked ELF audit inspects every declared executable section. It derives byte
order from the ELF header and checks the explicitly selected ELF machine ABI.
Actual independently compiled public native accessors supply instruction lengths,
field values and opcode macros. The walk covers compressed and variable lengths;
it does not import a handwritten instruction table or assume fetch alignment.
Each native decode invocation and the whole-artifact decision retain pinned
source, input, output and report records. A prohibited instruction is an actual
reported refusal, including its exact linked address, word and source symbol.

The shared explicit build route runs the audit before simulator dispatch. A
refusal returns `execution=not_attempted` with the actual ELF and audit evidence;
the enclosing observation completes before the diagnostic caller raises its
gate refusal. The refused path emits no numerical result or timing claim.
Accepted ELF/report bytes are reopened immediately before execution and again
afterward. This gate can accompany the prepared independent runtime context.

This proves a protected policy over declared executable sections under the
selected public accessor ABI. It does not prove physical CPU or simulator
equivalence, dynamic code absence, executed instruction presence, numerical
effects, ownership, synchronization or timing. Those remain separate mandatory
runtime controls. Diagnostic identity/add executions cannot issue their missing
authority or authorize a Phase 1/2 launch.

The prepared context also exercises its fixed instruction negative through the
ordinary source, translation, object and link route. It retains the original
three output expressions and numerical gate. A private assembly source adds one
prohibited word whose field value and length the live native accessor actually
observed. A scoped build recipe pins that source and retains its symbol in the
executable section; the selected assembler determines byte order. The complete
linked audit must find that exact source-bound word before the runner is called.
The control reopens the actual ELF, audit and native proof, checks that no
execution invocation exists, and then attributes a typed policy refusal. Compiler
or linker failure does not satisfy the instruction negative. This mutation stays
private to the evaluator and never supplies a compiler scaffold or author input.

The optional real integration regression selects public source checkouts,
compiler binaries, decoder member names and protected symbols through
`MERLIN_TEST_INSTRUCTION_SELECTION`. It reissues every live intake and links a
positive ELF plus a native-verified prohibited word in a negative ELF. These two
static binaries are not executed. An optional explicitly selected diagnostic
service owner additionally runs the prepared positive through the stock normal
pipeline with complete output decoding, then proves that the linked negative
never reaches execution. The positive's physical stage witness remains refused.

`component_instruction_policy.py` binds the protected symbol selection;
`component_instruction_audit.py` inspects the actual linked sections using the
selected native accessors. The shared `contract/elf_admission.py` interface
installs that independent check before execution rather than allowing candidate
code to select its own instruction acceptance.
