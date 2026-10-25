# Modular W8A8 complete-program policy correspondence

Worker candidate, October 22, 2026. Scope: Merlin's `lower_contraction_int8` and actual
before/after modules. Production transforms, shared/frozen interfaces and other workers'
files remain unchanged. No staging, commit, integration or push is authorized.

Read the user-approved [eight-day goal](../../progress/planner/eight-day-paper/README.md)
after compaction. Target: October 30. This packet exposes the local policy result separately
from the [rotary/source-prefix candidate](../rotary-domain/README.md). It does not complete
the full pass, approve the draft reference or establish successful whole-model execution.

## Precise theorem and obligations

`PolicyW8A8Correspondence.check_sound` proves that acceptance produces a reference
`W8A8Reference.expand(config, before)` and that the final checked target has **exactly the
same outcomes** as that reference for every context, function and input in
`W8A8SourceSemantics.opSem`. The checker covers the entire module: all function bodies,
internal calls, signatures, declarations, visibility, outer metadata and every supplied
suffix boundary. The suffix joins also bind actual MLIR bytes before checking.

This theorem requires **zero additional numerical laws** and no proof that the original
model executes successfully. It reuses the existing proved parametric alpha laws for the
named source interpretation. It does not prove that this interpretation is native MLIR,
that the draft reference expresses the intended quantization policy, or that the original
floating model equals the numerically changed result.

Two separate corollaries keep stronger obligations visible:

- `successful_reference`: if this particular reference execution returns a result list,
  the final checked target returns that exact list. All returned values are included.
- `defined_reference`: if reference execution is successful and nonpoison throughout an
  explicitly supplied context/input domain, so is final-target execution on that same domain.
  The reference-definedness proof is an explicit premise.

An unmodelled instruction can yield equal `stuck` outcomes. Such an equality does not
establish initialized execution; a dedicated regression checks that distinction. A complete
execution claim still needs support/definedness and legitimate input/capture facts. The older
seven source laws, three dynamic laws, arithmetic agreement and joint-model obligations remain
unchanged in the stronger numerical routes. They are not silently admitted by this packet.

All four broad flags remain false: `whole_pass_verified`, `policy_reviewed`,
`primitive_model_laws_verified` and `concrete_context_inhabited`. Every receipt remains
`abstracted`, exit 1. This opt-in standalone route does not enforce every production compile.

## Implementation and source delivery

| Owned file | Purpose |
| --- | --- |
| `merlin/lean/MerlinVerify/Check/PolicyW8A8Correspondence.lean` | Complete-module/suffix checker and outcome, success and nonpoison transfer theorems |
| `merlin/lean/MerlinVerify/Driver/W8A8Correspondence.lean` | Explicit claim boundaries and independent execution/review flags |
| `merlin/lean/W8A8CorrespondenceMain.lean` | Separate interpreted entry; shared Main/CLI remain unchanged |
| `src/merlin/verify/lean/correspondence_{evidence,policy}.py` | Authoritative exports, strict evidence types, fresh theorem audits, runtime/source identities and regenerated replay |
| `merlin/tests/ir/test_lean_w8a8_correspondence.py` | Actual pass outputs, multiple outputs, helpers, suffixes, compiler-valid mutations, refusal limits, tamper/replay and complete native comparisons |

`implementation-v2.patch` contains these six additive source/test files.
`combined-w8a8-owned-v2.patch` contains **all 68 additive W8A8-owned source/test files** from
the earlier bounded, numerical, rotary and current work, against main checkpoint
`229f42c627854c49476a101cb393d0b15044fab8`. It excludes shared Main/CLI, frozen definitions,
specs/locks, central ledgers and foreign candidates. `combined-owned-files-v2.json` records
the bytes; `source-closure-v2/` retains them.

`source-integration-inventory-v1.json` compares the private Lean source set to that base:
370 matching files, 41 absent library sources and one private Main configuration; no
unexpected differences to existing Lean sources. This is a conservative source inventory,
not a claim that all 412 files are imports of the new entry. The private Main is build-only
configuration and is not included in the delivery patch. Standalone entries/fixtures account
for the other Lean source files in the owned closure. Central integration remains pending.

## Frozen build and qualification

New private root: `out/build/lean-w8a8-correspondence/repo`. Earlier builds and source snapshots
remain preserved. Runtime SHA-256:
`cd803001728c8a9cb9f624cc774f44ad93d1e85717767e0e247fb9cc2efb7fb7`.
`freeze-v2.json` seals 1,309 records, including the 826-file runtime and owned source closure.
The actual proof roots are freshly audited in receipts; the private full guard audited
**9,984 theorems with zero violations** and passed the existing frozen lock.

Final qualification: **33 passed**, zero failures/errors/skips, in **139.851 seconds**.
The retained TinyLlama complete boundary **passes correspondence and independently replays**
against the original authoritative exports. The source/target census finds zero/155 generics
with actual i8/i8/i32 body arguments; that census is an empirical structural observation,
not an additional numerical execution theorem. Its stronger numerical checker still refuses
quantizer 1533. The final tests also retain 21 positive correspondence receipts and nine
refusals; one positive deliberately demonstrates equal unmodelled outcomes without initialized
execution. Two complete target-output observations agree across Lean, native MLIR and the
ordered recipe, covering floating and mixed-byte inputs. Those observations are not universal
native fidelity proofs. The positive replay test regenerates its actual checks separately.

All **1,309 frozen records**, the earlier rotary/numerical capsules' 1,263/1,230 records,
the historical 752-file runtime and six earlier qualified sources retain their recorded
bytes. Both six-file and complete 68-file additive patches pass `git apply --check` in an
empty private directory, without staging or changing the main checkout.
`qualification-v1.json` indexes the results, exact identities, receipts and limitations.
There are no pending build/test/replay processes for this milestone.

Development evidence remains separate:
the first run hit an overlapping private executable rebuild; the second passed 28 cases
but two native probes used a missing snapshot filename. The corrected snapshot/native and
new reduction-bound/omitted-transform checks then passed all five targeted cases.

Ruff, package docs and artifact layout pass. The structure gate retains the unrelated
`zephyr_model.py` ceiling failure, 3557 > 3537 lines. No foreign file was changed to address it.

## Reproduction and next work

Use the environment from [the earlier handoff](../handoff/README.md), changing
`MERLIN_REPO_ROOT` to the new private root. Keep the original virtualenv path for
`MERLIN_M2M_PYTHON`; invoking its resolved base Python loses the MLIR environment.

```sh
PYTHONPATH=src .venv/bin/python -m merlin.verify.lean.correspondence_policy \
  PATH_TO_ACTUAL_STAGE [PATH_TO_JOINED_SUFFIX ...] \
  --destination NEW_OUTPUT_DIRECTORY \
  --m2m-python "$MERLIN_M2M_PYTHON" --scope captured
```

`correspondence_policy.replay(receipt_path, new_destination, m2m_python=...)` regenerates
exports, audits and checks before comparing typed JSON evidence and all subjects. Do not
rebuild an existing frozen capsule to replay it; relocated/combined builds need new identities.

Next: qualify and independently review the intended policy/source-interface/domain; complete
selected capture/version/scales binding; resolve common-model compatibility and the stronger
numerical premises; centrally assemble and qualify the authorized combined candidate. Whole
W8A8 source-definedness still refuses quantizer 1533. Ordered attention bounds and justified
transcendental/softmax denominator reasoning remain the stretch track, with original
counterexamples and production variants preserved. See [worker status](../worker-status/README.md).
