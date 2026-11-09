---
title: Execution memory admission
kind: reference
status: current
owner: core
last_verified: 2026-11-09
related: [architecture, runtime]
code_refs:
  - src/merlin/runtime/execution_memory.py
  - src/merlin/runtime/elf_audit.py
  - src/merlin/runtime/backends/spike_model.py
---

# Execution memory admission

An ELF load-segment check does not cover absolute runtime heap and stack
addresses. A simulator's backing-store capacity also does not establish which
addresses its execution model decodes. Merlin's optional admission checks both
the loaded image and declared runtime storage against a selected provider-owned
decoded memory map.

## Provider contract

`MemoryMapBinding(path, sha256, execution_identity)` closes the exact map artifact
and compares its identity to the caller's independently selected execution
identity. The provider owns deriving these facts and binding them to the actual
engine, board or elaboration. Merlin verifies the artifact and declared evidence;
it does not infer hardware facts or prove their derivation from source files.

The JSON artifact has these fields:

| Field | Obligation |
| --- | --- |
| `schema` | `merlin.execution-memory-map.v1` |
| `status` | `complete`; unknown/incomplete maps refuse |
| `execution_identity` | Nonempty identity equal to the selected binding |
| `identity_mapped` | Exactly `true`; virtual translation needs a separate contract |
| `address_bits` | 32 or 64, agreeing with the ELF class |
| `regions` | Nonempty list of distinct `name`, unsigned `begin`, positive `bytes`, and explicit `permissions` using `R`, `W`, `X` |
| `file_pins` | Nonempty source/elaboration evidence list: declared `path`, SHA256 and optional exact byte length |

Regions must not overlap. A requested interval can cross adjacent regions if
every byte has all required permissions. Gaps, unknown permissions, overflow,
evidence drift and changed execution identities refuse.

## Runtime and loaded storage

`admit_execution_memory(elf, binding, reservations)` strictly reads every ELF
LOAD row, including its physical address. The entry must lie in executable
loaded memory. Physical and virtual load addresses must agree with the selected
identity-mapping contract. GNU readelf's execution flag `E` is normalized to `X`
in this strict path. Malformed LOAD rows cannot silently disappear.

Each `MemoryReservation(name, begin, bytes, permissions="RW", alignment=1)`
declares one disjoint absolute allocation, live for the complete execution.
The immutable tuple must cover all storage outside loaded segments. Reservations
must satisfy alignment and cannot overlap other reservations or loaded segments.
Storage already inside a loaded segment is covered by that segment and should
not be declared twice. Aliasing and reuse across phases require additional
ownership/lifetime proofs and are not authorized by this API.

The successful JSON receipt binds actual ELF bytes, map bytes, execution identity,
loaded intervals, runtime reservations and evidence. It proves interval admission;
it does not prove stack/heap peak demand, numerical correctness or performance.

## Normal bare-metal build

`spike_model.build` accepts optional `execution_memory_map` and
`execution_memory_reservations`. It checks the selected map before lowering and
again after linking. Its allocator reservation comes from the same layout and
size used in the runtime compile definitions. Its stack comes from the linked
`_stack_top` and `MERLIN_STACK_BYTES` symbols and must agree with the selected
build size. Loaded code, static I/O and weight bytes are checked from the ELF.
The caller declares additional absolute external buffers explicitly.

Admission must finish before the compilation recipe publishes completion.
Unknown/refused invocations remove a previous admission receipt. The optional
gate creates `execution_memory_admission.json` and returns the same structured
receipt. Unselected builds retain their emission policy and do not claim memory
admission; the gate itself changes no compiled image bytes.

Phase 0 should expose provider memory evidence and runtime demands; phase 1
should carry them through compilation and require admission before launch;
phase 2 should retain the exact admission with its measurement. Execution
providers must still close the selected map at launch. Supplying this API does
not make existing execution paths or historical measurements automatically
qualified.

`MemoryMapBinding.validate` checks the selected provider data;
`admit_execution_memory` checks actual absolute ELF load segments and the complete
explicit runtime reservations. Neither routine derives a SoC memory map from
simulator capacity or chooses a runtime platform.
