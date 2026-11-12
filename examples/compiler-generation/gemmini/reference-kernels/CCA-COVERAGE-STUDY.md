# Reference-kernel coverage study protocol

This document specifies the unlaunched source-ordered study design of 2026-10-29.
The corpus preparation checks and offline observations below are separate from
performance results. They do not establish an executed authoring comparison.

Author inputs exclude this protocol, experiment-wide subset selection, audit outcomes, earlier
gap diagnoses and the complete corpus. Each arm receives only its declared
selection. The corpus addition is recorded at commit `971bbc7ea`.

## Current design: source-ordered banks (v3, 2026-10-29)

This design supersedes the model-only count curve below with cumulative banks in the
ablation order **upstream Exo, then ResNet50, then TinyLlama, then SmolVLA**.
Every arm receives the same raw grant: the whole `originals/` folder (23 files,
Exo included). Each bank arm also receives exactly one pinned read-only
`cca-evidence/` bank and the native bank-only reference-analysis service.

| Arm | Profile | Bank units (cumulative) |
| --- | --- | ---: |
| Raw | `cca-coverage-v1/raw` | 0 |
| cca-exo | `cca-coverage-v1/bank-cca-exo` | 27 |
| cca-resnet50 | `cca-coverage-v1/bank-cca-resnet50` | 81 |
| cca-tinyllama | `cca-coverage-v1/bank-cca-tinyllama` | 87 |
| cca-smolvla | `cca-coverage-v1/bank-cca-smolvla` | 93 |

Pins, stages and the per-source acceptance audit are in
[subsets.v3.json](subsets.v3.json). The v3 corpus lifts every machine original
and the ahead-of-time compiled Exo C sources with the machine-code lifter only.
Source correspondence comes from symbols and DWARF, never C pattern matching.
Every listed accelerator site falls in a lifted region, which is checked
independently. The historical F10 `REFERENCE_MAP.json` is refused in every arm
(see the 20261028 `REFERENCE_MAP_DISCLOSURE.md`). The C4–C66 tables below are
retained as the frozen earlier design and are not used by this study.

## Earlier fixed design (superseded)

The earlier design has six distinct author conditions, with no repeats.
Every condition starts independently from the same frozen, qualified
`F10-FULL-NOREF-NOCCA` compiler and receives the same original model files:
`resnet50.elf`, `smolvla.elf`, `tinyllama.elf`. Exo material is excluded from this
study in both treatments, although the larger database retains it separately.

| Condition | Raw model files | CCA units | Distinct bodies: ResNet / SmolVLA / TinyLlama | Original occurrences |
| --- | --- | ---: | --- | ---: |
| Raw | All three | 0 | 0 / 0 / 0 | 0 derived |
| C4 | Same three | 4 | 2 / 1 / 1 | 4 |
| C8 | Same three | 8 | 6 / 1 / 1 | 10 |
| C16 | Same three | 16 | 13 / 1 / 2 | 18 |
| C32 | Same three | 32 | 25 / 4 / 3 | 36 |
| C66 | Same three | 66 | 54 / 6 / 6 | 73 |

These are strictly nested sets: C4 is a subset of C8, and so on. The exact body
IDs, original occurrence IDs, corpus pins, condition pins and coverage counts
are in [subsets.v2.json](subsets.v2.json); its membership/order and raw grants
match the frozen [v1 selection](subsets.v1.json). Identity means exact machine bytes;
occurrence-local analyses stay distinct. Stripped ELF spans are not verified
logical kernels, source functions or model layers. The database's headline 358
bodies includes host/source material and is not this study's denominator.

All authors can inspect all raw kernels. Consequently this measures **the effect
of supplied CCA coverage given constant raw-reference access**. It does not vary
total expert-kernel availability. Raw versus C66 is the full treatment comparison;
the other conditions describe the CCA coverage curve. One author run per
condition gives an exploratory curve, not an estimate of author-search variance.

## How the subsets are chosen

One deterministic diversity-first ordering is constructed before author outcomes:

1. Prefer an uncovered source model so the first three units cover all three.
2. Prefer units with selected ISA roles beyond configuration/synchronization.
3. Prefer previously unseen ISA roles, then instruction classes, then recovered
   field/knownness/value-bin features, then control-structure features.
4. Break remaining ties by smaller code size and then exact body SHA.

The shared selector has no Gemmini opcodes or model-gap labels. Instruction
classes and roles come from the selected ISA observer. Values use coarse type
and integer-magnitude bins; addresses and numerical scaling constants are not
matched to known winning settings. It never reads candidate compilers, hidden
cases, reference timings or performance outcomes. Missing values remain missing.

C4 supplies a compact cross-model core. C8 and C16 add observed variations in
fields, value populations, configurations and scalar control. C32 covers nearly
all coarse feature bins in this particular bank; C66 includes the remaining
variations and many structures similar under those coarse observations. This
does not prove their semantics or optimization opportunities are duplicates.

The full offline bank is used to choose the ordering, including the ordering for
C4. This is a declared curated-bank coverage experiment. Charge/disclose that
full preprocessing and selection cost for every condition. Do not claim that
only four kernels were analyzed to create C4, that these are random samples,
or that this is a preprocessing-cost scaling experiment. The curriculum is
designed to reveal diverse observations early; any measured plateau is
conditional on that ordering, common seed and author budget.

## What the actual raw-versus-CCA audit found

The original model ELFs contain 451,185 listed instructions and 65,838 selected
accelerator sites. Raw users can disassemble and analyze them using the same
common tools and selected target definitions. They have to identify code spans,
recover scalar operands, interpret instruction fields and track uncertainty.

CCA automates part of that work. However, the present canonical CCA is too thin
to be the sole treatment input:

- Across 73 accelerator occurrences there are only seven canonical signatures
  after removing provenance; 59 occurrences share one signature.
- None has a complete reconstructed schedule. Layer/source correspondence,
  dynamic counts in unbound loops, complete storage effects and timing remain
  unknown where their proofs are missing.
- The richer ISA-derived sidecars retain 276,162 field-site observations, of
  which 86,813 have unresolved values. They also retain bounded actual value
  profiles, complete field fingerprints where available, instruction roles,
  repeated recovered tuples and lexical/control witnesses.

The prepared treatment therefore supplies the full selected sidecars as well
as canonical CCA JSON/MLIR, occurrence cards and bounded lookup. It is the effect
of this CCA assistance pipeline, including organization/retrieval; this design
does not separately isolate the contribution of each component.

| CCA units | Occurrences | Coarse observed feature bins | Canonical signatures |
| ---: | ---: | ---: | ---: |
| 4 | 4 | 345 | 2 |
| 8 | 10 | 432 | 3 |
| 16 | 18 | 497 | 6 |
| 32 | 36 | 526 | 7 |
| 66 | 73 | 529 | 7 |

Feature coverage is not performance. Its flattening is partly a consequence of
the declared diversity-first ordering and coarse features. It must not be
reported as measured compiler-performance saturation.

## How CCA could help, and the test of that hypothesis

The hypothesis is that CCA reduces the effort required to find and substantiate
a useful compiler intervention. More analysis files alone do not establish that.

**Make observations easy to retrieve.** The new `reference_subsets query` selects
only the admitted bank, accepts a requested ISA role/instruction/field, ranks
actual observed values ahead of empty profiles, reports knownness and emits
bounded results with evidence pointers. Initial lookup avoids repeated copies of
the same byte body. Full occurrence-local reports remain reopenable. Local
example queries produced roughly 5.2 KB replies in 3–68 ms, including file-pin
verification; this is an operator lookup check, not an agent cost/speedup result.

**Connect expert evidence to the selected candidate.** The author must be able
to inspect the actual public source, selected consumer and emitted native
function. A useful comparison should expose an observable difference such as
transfer field values, issue structure or configuration changes, along with
the missing semantic/capacity conditions. It must not treat instruction-count
similarity as the objective. Reference-only analysis and typed compiler-decision
receipts do not prove a source/binary or source/consumer relation.

**Keep hypothesis and optimization choice with the author.** It is reasonable
for the agent to ask whether a reference observation applies to a public shape,
derive resource conditions from the selected target facts, and try a generic
compiler change. The bank provides no answers such as an expert's winning panel
size, a human-authored gap recipe, or which transformation to run. Recovered
instruction field values are original reference evidence, not chosen settings.

**Verify transfer rather than fallback correctness.** Require per-shape reporting
of whether the intervention actually applied, why it was legal or refused, and
whether inherited fallback was emitted. Cover M/K/N and resource boundaries,
including the predeclared dense M32 case. Preserve the original numerical recipe,
and require the ordinary correctness/performance checks for every accepted change.

## Preparation, lookup and access boundary

The source APIs live in the optional mining package:

```sh
python -m merlin.analysis.mining.reference_subsets prepare \
  --corpus "$CORPUS_ROOT" --out "$NEW_PRODUCT_ROOT" \
  --artifact jack-resnet50 --artifact jack-smolvla --artifact jack-tinyllama \
  --sizes 4 8 16 32
python -m merlin.analysis.mining.reference_subsets query \
  --bank "$SUBSET_ROOT/conditions/cca-4" \
  --instruction MVIN --field rows --limit 3
```

Allocate `NEW_PRODUCT_ROOT` with `merlin.common.artifacts.new_product`, as in the
main [rebuild guide](README.md). The full selection is automatically included
alongside the requested proper subsets. Instruction/field names in the lookup
example are selected-target data; the generic query implementation contains no
target-name or opcode cases.

The prepared local alias is
`out/artifacts/kernel-mining/gemmini/expert-kernel-subsets-v2/` (v1 remains frozen):

```text
selection.json                 Operator-only complete ordering and coverage
raw-mount-selection.json       Operator-only original-file mount selection
PUBLICATION.json               Corpus, bank, raw/order compatibility pins
CCA-AUDIT.json                 Actual corpus JSON/MLIR and observation audit
NATIVE-COMPONENT-CANARY.json    Actual expert contrasts through copied client/broker
DO_NOT_LAUNCH.json              Remaining native preflight requirements
conditions/
  raw/{bank.json,raw-grant.json}
  cca-4/{bank.json,raw-grant.json,units/,mappings/}
  cca-8/...
  cca-16/...
  cca-32/...
  cca-66/...
```

Raw originals are referenced by pinned per-file grants, not copied six times.
Every condition's raw grant has identical bytes. Author sandboxes must mount
those three originals read-only at the declared relative paths and only their
own condition's derived payload. The global corpus, other conditions, operator
planning material, full SQLite index and global audit must stay outside the
author view. The new query validates all selected file pins and refuses raw-only
or excluded-unit lookup. That is offline API enforcement, **not proof of native
sandbox/broker enforcement**.

Disable automatic reference-CCA routes for unselected IDs in the actual native
tool path. The author can still analyze and reason from its granted raw files
within its ordinary budget; forbidding that would change the raw baseline.
Any supported route that silently supplies the excluded precomputed bank would
invalidate the coverage condition. Keep the common target/ISA inputs, compiler
tools, seed, prompts/goals, author resources, evaluation policy and budgets fixed.

## Preconditions for launching, and paper analysis

The bank is prepared but this study is **not yet launch-qualified**. The native
grant/broker path has not exercised these new bank selections, a useful
candidate-bound comparison has not been demonstrated for this exact study, and
the common seed qualification plus exact budgets are not yet pinned. The local
`DO_NOT_LAUNCH.json` and tracked selection explicitly record those gaps.

The v2 [instruction observation fix](CCA-OBSERVATIONS.md) closes data loss between
the selected ISA observer, canonical CCA and the bounded native comparison reply.
Copied-client/broker checks now expose actual expert load-stride differences and
scale differences hidden by equal instruction-class counts. They exercise
unmodified ISA decoding, but compare expert spans to each other. They do not
qualify the study's actual seeded candidate, enforce the CCA subset at native
launch, or establish schedule equivalence. The whole-corpus audit and canary are
operator material and must not be supplied as author guidance.

Resolve those with short deterministic component/canary checks, without author
campaigns: reopen selected data through the real native client; test raw and
excluded-bank refusal; inspect an actually emitted public candidate and expert
comparison; confirm a useful observed difference survives the bounded response;
qualify the frozen common compiler. Attention's remaining ABI/source/build
limitations must stay explicit. A successful reference-only field query is
insufficient evidence that the entire feedback loop works.

Freeze the matrix, ordering and harness before examining author outcomes. All six
conditions start from the original seed in fresh contexts. If the full raw/CCA
pair runs first, later subsets still restart from that original seed and their
membership cannot be tuned using the pair's results. There are no repeat author
campaigns and no adaptive additions beyond the six conditions.

Evaluate fixed-budget checkpoints and final compilers on the same frozen roster.
Keep holdout checkpoint evaluation sealed until author freeze. Report correctness,
per-shape intervention/fallback coverage, time to the first verified useful
change, seed-relative speedup and author/preprocessing cost separately. Do not
drop failed shapes from an aggregate without disclosure. Repeated hardware timing
samples are measurements, not repeat author campaigns. Multiple shapes and
checkpoints are not independent author-search replicates.

Plot performance against the supplied CCA count, and separately against author
effort. Set a practical gain/noise threshold before reading treatment outcomes.
Ask whether the curve flattens; do not promise that it will. One run per condition
can suggest a knee for this curated bank and budget; it cannot identify a general
saturation threshold or establish a statistically robust CCA advantage.

Matching common conditions follows the nuisance-factor control principle in the
[NIST design guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm).
The caution about few independent runs is consistent with
[Agarwal et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html),
which studies a different application area. The matrix and pipeline above are
our compiler-study proposal, not findings from those sources.
