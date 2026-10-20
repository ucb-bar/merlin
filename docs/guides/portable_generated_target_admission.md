---
title: Portable generated-target instruction admission
kind: guide
status: current
owner: merlin-experiments
last_verified: 2026-10-19
related: [shared_pass_phase2, ../design/independent_instruction_policy]
code_refs:
  - src/merlin/targetgen/contract/elf_admission.py
  - src/merlin/targetgen/rtl/source_selection.py
  - src/merlin/targetgen/rtl/unit_correspondence.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/rtl_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/accessor_intake.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_instruction_declaration.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/private_instruction_coordinator.py
---

# Portable generated-target instruction admission

A second server must issue its own live instruction-admission service from the
reviewed source inputs. Copying a successful report, approval or package does not
reconstruct `LinkedElfAdmissionService` authority. This guide describes source and
operator-input transport; it grants no compiler, runtime or timing qualification.
All commands below run in the receiving server's managed workspace with its own
resource admission, fresh output directories and temporary directory.

## Select public source before provisioning inputs

For the V26 Gemmini example, select these exact published Git commits in managed
checkouts; do not substitute branch tips:

| Repository | Commit |
| --- | --- |
| `https://github.com/ucb-bar/merlin` | `f8a43c657f11ee417f4c96d8920fa2f7e66f3f5b` |
| `https://github.com/ucb-bar/gemmini-mlir` archive | `8b8fd3629f4e8e1c3349176241620621048b96c7` |

The archive's `compiler/` contains 13 Python files and its manifest. Its original
16-file submission additionally contained two author reports and had tree digest
`29213759089b71de26f8a5ab26ca6ef1b8071dc0bb690041eb7de5d0efb53f24`.
Bind the actual received compiler directory independently. Do not attribute the
original tree's evidence to the 14-file projection by substituting its digest.
`python "$ARCHIVE/reproduce/verify_archive.py"` checks published archive bytes,
not native execution or certification. V26 does not declare `target_ir_consumer`
or `emit_target_bundle`; its analysis-bundle command does not promise to consume
an independently transformed target IR.

## Obtain the complete operator input set

The archive's `reproduce/input-manifest.json` identifies separately owned inputs.
The following public RTL/source and reviewed operator data are needed for the
linked instruction service; private model checkpoints, golden arrays and hidden
capsules are not needed and must not be exported for this purpose.

| Input | Required content and origin |
| --- | --- |
| Hardware descriptor | Minimal reviewed target descriptor; V26 original SHA `9d331b293d3e6ebf0b8ae230607b3e6c57f8271c8acab1373f9ebcaac644877f`. |
| RTL source bundle | `source-selection.json`, selected FIRRTL, hierarchy, SoC HW and exact core HW closure; also selected FIR-to-HW tool and any explicitly selected elaboration receipt. Public elaborated source artifacts are operator-provided, not embedded in V26. |
| Command/predicate source | Public Gemmini checkout at `63f0b68a68f185a016a2993f2fb32e1661ea9e6c`: `GemminiISA.scala`, `LoopMatmul.scala`, `LoopConv.scala`. Keep tracked provenance, not extracted command tables. |
| Native accessor source | Public riscv-isa-sim checkout at `9c190a07c6838f6392bafa4ad83acea462c7f759`. The corresponding selected include tree must contain byte-identical tracked public headers. |
| Accessor specification | Reviewed type/member names only. Original SHA `09d2323b4c213f9f33e4f602df85e984efac7557edeec279ef2abd89dadc4e47`. |
| Software policy | Original SHA `3da7fd4266b1f48297b4cc4fbf211792137975b208bc6408f1fccc1b2786880b`: 15 prohibited `LOOP_WS*`/`LOOP_CONV_WS*` source symbols. It supplies no numeric command values. |
| Tools | Selected `firtool`, `circt-opt`, C++20 host compiler and their usable host dependencies. Original executable hashes are listed below; matching version strings are insufficient. |
| Reviewed declaration | `merlin.private_instruction_selection.v1`, current absolute canonical paths, exact file hashes, source commits, source span/predicates, decoder selection and explicit forbidden roots. |

The original accessor specification selects header `riscv/rocc.h`, union
`rocc_insn_union_t`, word member `i`, fields member `r`, conversion `insn_t`,
length member `length`, fields `opcode,rd,xs2,xs1,xd,rs1,rs2,funct`, and constants
`ROCC_OPCODE3,ROCC_OPCODE_MASK`. The decoder selects those two constants, member
`funct` and the explicitly reviewed RISC-V ELF machine value. The original command
span is `// funct values` through `CONFIG_EX`. The four source predicates are
`is_loop_run_cmd` and `is_loop_config_cmd` in each loop source, on
`cmd.bits.cmd.inst.funct`. Obtain the original files and review these selections;
this description is not a replacement approval.

The accessor issuer compiles its own fixed observer with `-std=c++20 -O2 -MMD`.
It checks **every discovered non-system header dependency** against the selected
tracked checkout and include tree. Provision the full matching include closure;
copying only `rocc.h` is insufficient. Host compiler/system-library dependency
closure and physical RTL equivalence remain explicitly unqualified by that test.

For re-elaboration rather than transfer of exact public artifacts, the original
Chipyard selection was `3e9bfb4bf58a7dfe6e1082496a9578a5dcd83831`, with Gemmini
as above, Saturn `e16a301d261fa4b5a374eb7069b78107a8c9aefb`, and FireSim
`b084672c2f8cf32e55d78f73a001074c23f8a2b8`. The selected
`GemminiGsimFireSimAlignedConfig.scala` content hash was
`5392120d1ed547b3a70c3786dd5a3d3d697f7817f9ce55e1ffc25b6c2ff6b2ec`.
A Git revision alone does not provision that configuration file or prove the old
elaboration; obtain its exact public source or select and review a new one.

## Relocate prospectively, then review the new declaration

Keep received original manifests/receipts byte-exact as historical provenance.
Place approved source inputs under a receiving-server input root. Do not symlink,
rewrite an old receipt's paths in place, or recreate the first server's directory
layout to masquerade as its owner. Record SHA-256 for each transferred file.

A portable route is to regenerate a **new** source selection with the existing
producer on the received public FIRRTL and hierarchy:

```python
from pathlib import Path
from merlin.targetgen.rtl.source_selection import produce_selection

source_selection = produce_selection(
    target="gemmini", firrtl=Path(FIR), hierarchy=Path(HIERARCHY),
    generator="gemmini", config="GemminiGsimFireSimAlignedConfig",
    core_root="Gemmini", firtool=Path(FIRTOOL), output=Path(FRESH_SOURCE_OUTPUT),
)
```

This runs the selected source tool, not a target simulation. It derives the
complete module closure and current producer attribution. Preserve any differing
output as a new selection; it does not inherit the old artifact's identity.
Select an elaboration receipt only if its original verifier accepts the actual
source/configuration on this host. Otherwise historical source-to-FIRRTL origin
stays UNKNOWN. The old `production.producer_source` is historical metadata:
current admission does not reopen that old path/hash. It pins current reader
modules, reruns the selected tool, and requires reproduced SoC/core HW bytes to
match the newly selected source bundle. Actual source/tool/output drift refuses.

Create a new closed declaration following
`private_instruction_declaration.read`'s schema. Rebind hardware descriptor and
source bundle, command checkout/source/tool, four predicate files, accessor
checkout/include/compiler/spec, policy and forbidden roots. Hash the complete new
declaration **after** path selection. A legitimate operator review must approve
those current choices and exclusions; retain the new approval separately. The
old selection hash `93c2102f37914cffb7e6a2c884b7d62d8a3149f0cc4ef72703a3ea026d265ec6`
and approval `20524cfe3f4447bea7d83a0668f626a3b093b84972d6d2e82d1fd1496fec7340`
identify first-server history and cannot approve a changed declaration.

## Issue the original live service

With the receiving environment selected, invoke the fixed original coordinator:

```python
from pathlib import Path
from merlin_experiments.phase1.feedback import private_instruction_declaration as D
from merlin_experiments.phase1.feedback import private_instruction_coordinator as C

prepared = C.prepare(
    D.read(Path(REVIEWED_DECLARATION), target="gemmini"),
    target="gemmini", output=Path(FRESH_ADMISSION_OUTPUT),
)
prepared.verify()
service = prepared.service
service.verify("gemmini")
```

`FRESH_ADMISSION_OUTPUT` must be canonical, absent, disjoint from selected inputs
and exclusions, and owned by the receiving task. The constructor reruns hardware,
command, predicate, native accessor and policy issuers. Keep this live object;
neither the report nor a saved `status: accepted` can restore it.

Pass `service` as the original build API's `linked_elf_admission` selection. It
checks the actual final linked ELF before dispatch; refusal means no execution.
The build reopens the unchanged admission report and ELF before and after native
execution. A direct diagnostic can call `service.evaluate(elf=..., target=...,
evidence_root=...)` and `service.revalidate(...)`, but must retain all original
numerical, source/effect, runtime and timing gates. The structural
`model_check.admission.service` only records digest/declared ISA and is not a
replacement for this independent instruction-policy service.

## Preserve the two hardware identities and the timing boundary

The V26 facts hash is
`b8ff28edc096392cec356f9fececdf17ba70cd3e4b02e6c5931aedc43bb117af`.
Its embedded `facts.simulation_correspondence` records **110 Gemmini unit modules**:
110 exact FIRRTL bodies, 109 exact HW modules and 110 structurally identical HW
modules; the remaining text difference is an `sv.namehint`. Its target FIRRTL is
`b7d9826b1c9f16fd00bc757b4334a7f330a842c6a494e0c28608dc351b305bf0`;
the gSIM FIRRTL is
`6748b9756799c086a6fb06e6d80b4b052f2771f9eb1858c172e14dcf441ba149`.
They are **different artifacts**. `unit_correspondence.execution_binding` checks
the original facts' target-side identities and returns the selected simulation
side; it does not recompute the equivalence record. Reproduce or independently
admit this join on the receiving host. Do not fabricate identical FIR hashes to
satisfy `prepared.require_facts(actual_public_private_binding)`.

This unit comparison excludes SoC memory/cache, clocks and bridges. It is not
simulator fidelity or timing qualification. Original tool/runtime reference hashes:

| Artifact | SHA-256 |
| --- | --- |
| firtool | `cd4a5be2461207f2d1261f8cfba7ddb8c74648ddc3f2bf64a4a1a23abe11726a` |
| circt-opt | `41512929e7c37de1c25190dbc0e3888b2f230cf6a9ee8eeaa6ca68c5258883b3` |
| Host C++ compiler | `2360901d864cf10bfd6296e261cb2c14053552a80377761ab07146ec9ec9a2c0` |
| Functional Spike | `5df9c82ca3e921823da8606284a258b491541defbafee4586fb938639e03c6b9` |
| Functional extension | `2dbd8445376cdda0df63abcf217c7a19f974d34a16ada21b549ed82fffd718eb` |
| Original gSIM label producer | `76ddbfdc5b3beadef6b6b37768a08522122327057b92d71d3539a9529e4ea651` |

Different native tool bytes require explicit current selection/review and fresh
original issuer evidence. A compatible instruction-admission service proves only
the selected static prohibition over complete executable sections. It does not
complete V26's resource/capsule certificate, qualify shared transformations or
full models, or transfer the frozen Spike model's calibrated domain. The published
capsule-certified Phase 2 route still requires its original certificate and
publication; passing capsule counts cannot replace those inputs.
