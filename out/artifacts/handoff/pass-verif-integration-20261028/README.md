# Pass verification integration checkpoint — 2026-10-28

**Paused for the requested colleague handoff. Latest-cut qualification is unfinished.**
Continue from `feat/pass-verif-integration`. Compiler work is committed at
`1ac44938b`; merge `ba4fb6b34` includes the colleague's `018c3795d3` and primary
`89ac040ae`. Original histories are retained. This checkpoint is a development
branch, not a completed qualification or human semantic approval.

The latest user requested immediate publication for colleague review, without
a full test suite. The follow-up starts from published `82430791d` and adds the
small MACC theorem and review findings below. Both follow-up workers have
finished; no compiler adapter or numerical approval was added.

## Start here

- [Compiler stack: location, interface, example and claim](../../../../docs/guides/lean_compiler_stack.md)
- [Interactive offline stack](merlin-lean-stack.html): five compiler checks and two explicit configuration APIs
- [Machine-readable checkpoint and remaining test nodes](qualification.json)
- [Quantization progress and review tasks](../quantization-lean-20261028/README.md)
- [Assumption ledger](../../../../merlin/lean/LEDGER.md)
- [Verification design](../../../../docs/design/merlin_lean_verification.md)
- [Shared compiler tooling](../../../../docs/guides/verified_compiler_tooling.md)
- [Root handoff and historical checkpoints](../../../../VERIFICATION_HANDOFF.md)

Fetch and check out `origin/feat/pass-verif-integration` before continuing.
The old `feat/pass-verification-dce-cse` branch does not contain this compiler
integration. No numerical or hardware approval is implied by either branch.

## Implemented

The compiler selector registers `prune_dead_pure_tensor_ops`, `cse`,
`ordered_loop_unrolling`, `ordered_loop_segmentation` and `buffer_reuse`.
Transform selection is separate and off by default: `--loop-unroll-factor`,
`--loop-tile-size`, `--buffer-reuse-allocator`, or typed
`compiler_transforms.Options`. Selected changing boundaries require fresh
proofs before returned compiler products can be delivered.

The loop adapter captures actual source SCF boundaries without a historical
producer checkout. Individual unrolling uses auxiliary segmentation only as
proof scaffolding; the auxiliary output is never delivered. Standalone tiling
uses an identity unroll stage. Explicit private memory takes a direct
SCF/memref-to-LLVM route with its existing allocation/free lifetimes.

Current-compiler replay support now reconstructs individual/composed profiles
and optionally delivers the exact target through a fresh completion. Its three
host binding/root controls pass. The new separate-process native replay tests
are written but have not yet run on the latest merged cut.

The colleague's configuration-write elimination and scoped partial loop-hoist
proofs are merged. They bind complete instruction-MLIR or Kernel endpoints,
selected ISA/effect facts, canonical proof sources, root/type/import audits,
process artifacts and live delivery/replay. Cleanup-helper binding fixes are
included for elision, buffer and standalone loop receipts. Hoist receipts also
observe imported helper origins. The hoist unit fixture now regenerates exactly
matching local facts/software and accepts `MERLIN_HOIST_REFERENCE`, avoiding
retained paths into another machine's home directory.

Configuration APIs remain explicit conditional interfaces, outside the five
MLIR compiler selectors. Elision removes only already-redundant total pure
constant writes. Hoisting moves one constant pure setter before a positive
counted loop with an ordered preserving body. Both preserve complete modeled
error/terminal states, including failing retained calls. Zero-trip/nested/
completion-token/asynchronous and overlapping-body-write hoists are excluded.

## Evidence and limits

### Follow-up review for this handoff

`MerlinVerify.Check.Bf16F32AccBoundary.check_sound` composes the existing exact
output/reference equality and strict local-site qualification, requiring at
least one collected source contraction. It includes unsuccessful executions
in the equality claim, under explicit SSA-renaming/call-locality laws. This is
conformance to the source-derived **draft numerical reference**, not equality
to the original half-accumulation program. The new module compiled with Lean
4.35.0; its theorem uses only `propext`, `Classical.choice` and `Quot.sound`.
No compiler selector, adapter, admission record or frozen semantics changed.

The read-only review found these remaining tasks; it ran no adversarial tests:

- `bf16_policy_gate.check_policy` obtains trusted admission/review roots through
  `repo_root()`, which honors `MERLIN_REPO_ROOT`. Bind governance records to the
  installed source owner and test override refusal before enabling a policy gate.
- `loop_transform_gate` records compiled-artifact hashes after its audit and
  completion checks the recorded roster without checking tree membership.
  Bring this in line with the buffer gate's pre-audit inventory and mutation
  checks. No bypass was demonstrated.
- The loop compiler profile requires one changed top-level loop, exactly two
  entry arguments and 1–4 word carries. Tails, empty and fully inlined loops
  are excluded. Selected mixed unchanged/changed loop stages are refused.
- The buffer compiler route requires normalization, reuse and tiling all to
  change. Already-normalized and reuse-only chains need proof/adapter work.

The new theorem's targeted compilation is the only new proof execution in this
follow-up. Earlier results below retain their original source cuts. The full
suite was not rerun, as requested.

The original compiler cut passed **35 distinct compiler tests**: a retained
33-test run plus a nine-case focused recheck overlapping seven cases. Thirteen
positive compiler cases accepted fresh proofs and emitted byte-identical gated
and ungated LLVM (ten loop and three memory cases). Native tests compared three
scalar variants over six vectors and one memory variant over three vectors,
including wrapping values and caller-memory observations. Two actual loop
corruptions reached Lean and were rejected; two memory corruptions were
structural `unsupported` refusals, not Lean proof counterexamples.

Those results precede the latest replay/helper-binding changes. Their exact
25-file source cut is retained locally in
`out/build/pass-verif-integration-20261028/qualified-before-configuration/`.
Do not relabel them as final qualification of the merged source.

On this host, isolated `018c3795d3` source passed a fresh baseline check, exact
live delivery and a separate-process replay for each configuration family:

| Family | Required roots | Transitive theorem audit | Axioms |
| --- | --- | --- | --- |
| Configuration elision | 17 | 247 | Standard three only |
| Scoped partial hoist | 15 | 315 | Standard three only |

Counts overlap and are not summed. These are new baseline repetitions, not a
new full campaign. The colleague's historical 13-pair elision and seven-pair
hoist campaigns retain their original cuts and approval limits.

The earlier full `MerlinVerify` guard passed: 566 tensor-checker modules,
15,300 shared theorems, zero violations, unchanged locks/pinned statements.
Loop and memory closures were audited separately by their gate jobs.
Latest focused check results and source hashes are in `qualification.json`.

The legacy 780-test regression was gracefully interrupted for this handoff:
**661 passed, three failed, 116 not completed**. All three failures reproduce
with the exact pre-integration Python source at `1f1b04a64`, identical captured
before/after IR and identical checker refusals. Two body checks do not verify;
one source-support check refuses generated `scf.for`. This establishes that the
failures predate this integration; it does not establish whether each underlying
rewrite is semantically incorrect. Preserve them and do not relax expectations.
The remaining 116 node IDs, including the interrupted case, are in the checkpoint.

Full-tree hygiene checks also have inherited findings: structure 16, artifact
layout 48, target names 7 and regex 44. Every flagged source matched the merged
base bytes when audited. Changed-file/staged checks passed before this latest
checkpoint; final checkpoint checks are recorded separately.

## Next owner

1. Review the changes on `origin/feat/pass-verif-integration` and the findings
   above. No full-suite run is needed to inspect this development branch.
   For qualification, run focused current compiler controls in
   `merlin/tests/ir/test_compiler_formal_transforms.py` (38 tests), including
   its new separate-process replay checks. Repeat
   the accepted loop/memory profiles after the updated source inventories; do
   not recover authority from saved reports.
2. Complete the 116 pending legacy nodes listed in `qualification.json` and
   retain the three independently reproduced baseline failures. The earlier
   suite is not an all-green full regression.
3. Check/replay both configuration APIs on the final installed source path and
   chosen SDK. Isolated producer capture ignores `PYTHONPATH`; use Python with
   the producer's normal dependencies installed. `MERLIN_HOIST_REFERENCE` must
   select a read-only detached source checkout at
   `e0484f4886ae0780d0325cdfa998842b61bb647c`; elision uses
   `4499eebb959c0f92cefd92cc6d80646ba66474ed`. Preserve the hoist source manifest.
4. Refresh the final source/evidence manifest and stack status only after those
   checks. Raw test/proof/SDK trees remain local under
   `out/build/pass-verif-integration-20261028/`; saved JSON is descriptive data.
5. Resume MACC through the [quantization handoff](../quantization-lean-20261028/README.md).
   Decide the source contract through independent review, harden its admission
   root, then integrate exact captures and fresh replay. Candidate-reference
   conformance must retain its distinct status until numerical admission.

Loop proofs concern equal returns/errors/terminal world under their common
scalar/SCF interpretation. Buffer proofs concern possible successful public
observations under total/private/unobservable allocation, not full error/world
or finite-capacity allocator equivalence. Python projection/transport, native
lowering and hardware fidelity remain separate. Frozen tensor semantics,
SPEC_LOCK, A9 and numerical admission are unchanged.
