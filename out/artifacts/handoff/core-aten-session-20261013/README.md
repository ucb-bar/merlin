# Core ATen integrated checkpoint

The primary development branches are `core-aten` in
[ucb-bar/merlin](https://github.com/ucb-bar/merlin/tree/core-aten) and
[ucb-bar/model2MLIR](https://github.com/ucb-bar/model2MLIR/tree/core-aten).
Integrate current upstream `main` before publishing a checkpoint. Keep exploratory
branches and worktrees on the contributor forks, then review changes before
bringing them into these integrated branches. No AI coauthor trailers are added.

## What the two branches deliver

Merlin provides full-call capture and execution grading for the canonical 193
operators and sealed 10,160-case suite. It preserves multiple outputs, complex
scalar precision, mutation, observed strides/offsets and borrowed/fresh result
storage. It supports proved readonly expanded inputs, safe resize growth and
authenticated legacy ABI results. Missing raw LayerNorm-statistics backing bytes
produce typed per-case refusals rather than unsafe replay or invented input data.

Its device path retains source/target pins, declared quantization and transfers,
exact integer references, raw accumulator and ordered f32 readout checks, compiled
artifact bindings and attributed device instructions. Host fallback is reported
separately. Phase 0/1 tooling supports explicit case identities and full-call
capsules; the older 38-case reference remains auxiliary evidence.

model2MLIR supplies exact static/traced result structure, tuple projection and RNG
lineage, source-only allocation/layout metadata, complete real-FFT spectra and
reviewed CPU numerical lowering. Repairs cover storage rounding, normalization
statistics and layout dispatch, pooling, half softmax/rsqrt, grid coordinates,
Boolean padding and BF16 embedding eligibility. BatchNorm mean reduction now
preserves the distinct dense and strided CPU paths. The integrated revision also
includes upstream's Transformers static-cache compatibility and post-export eager
execution fix. Merlin's software pin selects the exact integrated revision and
byte-checks every declared required source path.

## Completed baseline evidence

The complete audited baseline is Merlin
`5a2be400530dc49ba7a8c0e29d610e2b4a4b632d` and model2MLIR
`587133ffd7befa88b2793140b9c2d06557ffd6ed`:

| Corpus | Scalar | Gemmini |
| --- | --- | --- |
| Canonical 193 | 193 observed-semantic passes, all host | 193 passes: 3 device, 190 host |
| Sealed 10,160 | 10,141 observed-semantic passes, all host | 10,141 passes: 48 device, 10,093 host |

Each full track retains six known contraction mismatches and 13 typed
input-storage refusals. All five BatchNorm regressions from the preceding
`3325/479` checkpoint are repaired, with zero regressions against that checkpoint.
The earlier `007900/e9e` continuation has 83 repaired failures and nine historical
PASS-to-storage-refusal transitions; those transitions remain explicit, and its
original reports and denominator are unchanged.

Source/runtime/device bindings pass. Strict overall qualification remains BLOCK:
nine historical passes lack authenticated physical backing bytes, and two
inherited prepared-to-MLIR trace identities remain unresolved per track. Passing
semantics covers checked values, mutation and observed metadata; it does not
establish unrestricted full semantics. The 193 set has 190 numerical passes and
three metadata-only cases. Secondary output-only counts overlap and are not added
to observed-semantic totals. Spike results establish functional execution only.

Evidence remains local under
`out/artifacts/verification/core-aten/remaining-20261012/integration/batchnorm-correction/`,
including `core193/final-campaign-audit.json` and
`full-recheck-independent/final/{independent-final-report.md,completed-audit.json}`.
These full results belong to the exact baseline pair above. Qualification of the
upstream-integrated delivery is recorded separately; the old run is not relabeled.

## Placement, quantization and remaining work

The unchanged reviewed software spec permits the existing TorchAO/PT2E W8A8 flow:
int8 operands and i32 accumulation. Exact device gates are separate from error
against the original float result. Of the 48 device passes, 25 use capture
quantization and 23 retain “device path conforms; host int8 narrowing not declared.”
The frontend's 22 TorchAO scheme descriptors do not certify INT4, FP8 or MX device
execution. The rejected compensated-addmm experiment remains excluded.

Three device calls in the 193 set measure current routing, not hardware capacity.
BMM readout is blocked by a rank-2 host-cast declaration; pooling, operand sums,
fused readout and movement need explicit source-derived admission proofs. Derive
Gemmini coverage and the minimum gap set jointly with Loren under the existing
spec. Correct host fallback is fair game; count verified device coverage separately.
An optional-device Phase 1 protocol is a fork exploration pending review, not part
of this checkpoint or a launched experiment.

Next: review legal routing representations, prepare a fresh source-bound release,
qualify its reference/seal/private-answer isolation and launch Phase 1 on 193 then
10,160 cases. FireSim awaits AWS approval. The concise live journal is
`out/artifacts/handoff/phase1-journal.md`; no new Phase 1 or FireSim session follows
from publishing source.

## Preserved lineage

The original handoff is `out/artifacts/handoff/codex-handoff-20261012.md`.
Fork archive refs preserve the unfinished starting checkpoint
`b5a7cd49f19ebadd473ce7f72be4a8f3bc8b60b6`, reviewed baselines and blocked experiments.
That unfinished checkpoint is not a production ancestor and is not merged into
the integrated line. Detailed commit inventories and branch reviews live under
`out/artifacts/verification/core-aten/remaining-20261012/upstream-preparation/`.
Normal Merlin gates retain only the exact independently checked 17 inherited
wiring diagnostics; all other required gates must pass.
