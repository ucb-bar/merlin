---
title: "Design: how a capsule corpus is generated"
kind: design
status: current
owner: core
last_verified: 2026-09-07
related: [derived_corpus_sources, derived_capsule_axes, capsule_phase_split, compiler_verification]
code_refs:
  - merlin/contract/capsules/generate_corpus.py
  - merlin/python/merlin/targetgen/corpus_spec.py
  - merlin/python/merlin/targetgen/corpus_synth.py
  - merlin/python/merlin/targetgen/capsule_source.py
  - merlin/python/merlin/targetgen/capsule_golden.py
  - merlin/python/merlin/targetgen/conformance.py
  - merlin/contract/schemas/capsule.schema.json
---

# How a capsule corpus is generated

This is the mechanism note: what a capsule is, and how one gets made. The *requirement* side — where
the coverage cells come from — is [deriving the corpus from the spec, the RTL facts and the
workloads](derived_corpus_sources.md), and the axes a cell cannot express are in [deriving the
axes](derived_capsule_axes.md). This note picks up where those leave off and follows a single entry
from a declarative profile row to a written, scrubbed, provenance-stamped directory on disk.

The one-sentence version: **a capsule corpus is a pure function of the target's own facts.** Which
capsules exist is derived from the conformance requirement; every per-capsule field is derived from
the target's descriptor, capability manifest and RTL facts; the golden is computed by an engine chosen
by the numeric regime; and regeneration is byte-stable. Nothing in the pipeline's control flow names a
target.

---

## 1. The unit

A capsule is a directory, not a file. The authoritative bundle list is `_CAPSULE_FILES`
(`targetgen/contract/materialize.py:41`):

| file | what it is | tracked? |
|---|---|---|
| `capsule.yaml` | the declaration: op, shapes, dtypes, epilogue, expected instruction classes, required oracle tiers, semantic block | yes |
| `capsule.interface.mlir` | the program, in the `merlin_iface` dialect — what the backend must compile | yes |
| `capsule.pytorch.py` | the frontend-faithful source, for capsules authored in PyTorch | yes |
| `capsule.linalg.mlir` | the captured linalg-on-tensors module, for model-slice capsules | yes |
| `capsule.weights.safetensors` | captured weights, for model slices | no — large regenerable binary |
| `golden.yaml` | **the answer key** — expected outputs plus oracle provenance | **no — answer key** |
| `expected_instruction_coverage.yaml` | **an answer surface** — the instruction classes the trace must contain | **no — answer key** |
| `README.md` | human prose describing the case | yes |

What stays tracked is the **contract** — `capsule.interface.mlir` and `capsule.yaml`, the op, shape
and dtype the agent compiles against, plus `MANIFEST.yaml`. Everything that constitutes an answer is
untracked, and `merlin/contract/capsules/.gitignore` is worth reading in full because it is the
clearest statement of the benchmark's threat model. It excludes, each with its reason:

- `golden.yaml` and `expected_instruction_coverage.yaml` — the graded answers;
- the entire `hidden/` subtree at any depth — *even the interfaces reveal held-out shapes*;
- `profiles/*.hidden.yaml` — the holdout **specification** is itself an answer, and the tracked
  profile sits inside the `merlin/contract/` grant every arm gets read-only;
- `capsule.weights.safetensors` and `*.safetensors` — large regenerable binaries;
- `counterexample_inputs.json` — solver counterexample inputs are an answer key in the same sense:
  the exact inputs on which a refuted program disagrees with its specification.

The sandbox independently masks the answer surfaces from the agent under test
(`targetgen/sandbox/answer_surfaces.py`), so the protection is two-layer: untracked *and* masked.

The schema `capsule.schema.json` describes only `capsule.yaml`; it requires eight top-level fields —
`name`, `kind`, `source_role`, `label`, `operation`, `numeric_policy`, `expected`,
`required_oracle_tiers`.

`kind` ∈ `isa | layer | model_slice | model` and `label` ∈ `public | dev | hidden`.

---

## 2. The pipeline

Eleven stages. Entry point is `generate_corpus.generate_target(target)` (`:2923`), driven by
`main()` (`:3270`) with only two flags — `--target` and `--comparison-manifest`. The output root is
**derived, never configurable**: `out_root = Path(te.capsule_corpus).parent` (`:2926`).

| # | stage | code | produces |
|---|---|---|---|
| 0 | read the target's facts | `load_target_experiment`, `load_capability_manifest`, `rtl.facts` | descriptor, manifest, RTL facts |
| 1 | derive the requirement | `conformance.spec()` / `required_cells()` | `conformance/<target>.yaml` — cells + 8 axes |
| 2 | synthesize entries | `corpus_synth` via `synth_capsule_corpus.py --write` | `profiles/<target>.synth.yaml` |
| 3 | merge the profile chain | `load_profile()` (`:358`) | one profile dict |
| 4 | derive the binding | `corpus_spec.derive_binding()` (`:370`) | `CorpusBinding` |
| 5 | expand sweeps | `expand_sweeps()` (`:2577`) | flat entry list |
| 6 | resolve extents | `_resolve_flat_extents()` (`:2079`) | entries with concrete shapes |
| 7 | build each entry | `corpus_spec.build()` (`:1671`) → `BUILDERS` | capsule dict + interface MLIR |
| 8 | compute the golden | regime-dispatched (`_write_capsule_inner`, `:1935`) | `golden.yaml` |
| 9 | scrub | `_scrub_capsule_dir()` (`:486`) | no local paths, no absolute weight refs |
| 10 | prune + stamp provenance | `_prune_superseded_synth()` (`:3079`), `update_provenance_manifest()` (`:3144`) | `MANIFEST.yaml` |

### Stage 3 — the profile merge chain

`load_profile` merges **five files in a hardcoded order**. It is *not* a glob, and that has bitten
before: `counterexamples.py` once documented it as a glob, wrote `<target>.smt.yaml`, and every entry
went to a filename nothing read.

| order | file | contributes |
|---|---|---|
| 1 | `profiles/<target>.yaml` | `datapath:` block + `capsules:` + `sweeps:` |
| 2 | `profiles/_perf.yaml` | the single shared performance template; appends `sweeps`, records path + sha256 |
| 3 | `profiles/<target>.synth.yaml` | `capsules` only — the derived requirement (stage 2) |
| 4 | `profiles/<target>.smt.yaml` | `capsules` only — SMT counterexamples promoted to capsules |
| 5 | `profiles/<target>.hidden.yaml` | gitignored holdout sidecar; appends `capsules` **and** `sweeps` |

Entries append, preserving declaration order. A duplicate `name` raises. `profile_targets()` (`:418`)
excludes `_`-prefixed and **dotted** stems, so a dotted stem is always a sidecar, never a target.

### Stage 4 — the binding: what is derived per target

`CorpusBinding` (`corpus_spec.py:70`) is the whole "no target literals" story in one dataclass. Every
field comes from the descriptor, the capability manifest or the profile's `datapath:` block:

`tile_dim` (from RTL facts / manifest; software default only for a target with no hardware mesh),
`operand_dtype`, `accum_dtype`, `integer`, `tiers`, `compare`, `atol`/`rtol`, `scaling`,
`scale_block` (elements per E8M0 block), `requant_output_dtype`, `requant_shift`,
`subnormal_operand_flush`, `inapplicable_tiers`, `semantic_defaults`, and `classes_for` — a *callable*
that maps (op, output dtype, epilogue, movement) to the instruction classes the trace must contain.

Two of these carry hard-won rules worth quoting in a paper:

- **`requant_shift` defaults to nothing.** If a target has not declared it, a capsule asking for the
  `requant` stage *fails to build*. Three engines (golden, reference, simulator) each carry their own
  fallback of 4, so an undeclared shift makes them agree with each other by coincidence while the
  backend under test sees no shift at all and cannot know what to emit.
- **`inapplicable_tiers`** records oracle tiers that cannot corroborate a result *for this target*,
  each with its reason. A tier whose model disagrees with the RTL about the machine itself grades a
  correct kernel wrong every time. It is carried into every capsule and honoured as an explicit
  `skipped` — never as a pass.

### Stage 5 — sweeps

A sweep is a cross-product over named axes plus a shared `base`, producing exactly the flat entry
dicts the per-capsule pipeline already consumes, so nothing downstream changes. Extents are written in
tile-relative tokens (`tile`, `tile-1`, `tile+1`, `tile/2`) and resolved against `binding.tile_dim` by
`resolve_extent` (`:2175`). Two rules are enforced, not documented:

- **Every fitted axis needs at least two distinct points.** K is always fitted when present, because
  one reduction depth cannot separate a tiled unit's rate from fixed overhead. A one-point fit prices a
  parameter confidently and wrongly.
- **Names must be unique** across generated and hand-authored entries, or one capsule silently
  overwrites another's directory and the corpus shrinks without saying so.

`_MAX_SWEEP_CAPSULES = 128` (`:2072`) bounds the expansion, and exceeding the budget **raises** rather
than truncating — a silently dropped point reads downstream as a covered one.

### Stage 7 — the builders

`build()` dispatches on the entry's `op` through `BUILDERS`: **14 op tokens → 12 distinct builders**
(`matmul`, `linear` and `fused_matmul_bias` all resolve to `build_matmul`). Each returns
`(capsule dict, interface MLIR)`. Shapes arrive in tile units and are scaled by `binding.tile_dim`.

The schema's `operation.op` enum holds **35** tokens — a superset, pinned by
`test_the_op_enum_covers_every_builder`. The extra 21 (`gelu`, `layernorm`, `rope`, `patch_embed`,
`attention_full`, …) belong to the **second authoring path**: capsules captured from PyTorch rather
than built from a template. That path lives in `capsule_source.py` and is selected per entry:

- `source: pytorch` / `pytorch_ref:` → `write_pytorch_capsule` — frontend-faithful, lowered to linalg
  via model2MLIR, graded against a host torch-eager golden;
- `source: spec` / `spec_ref:` → `write_spec_capsule` — program *and* bit-exact golden come from the
  `specir` verification spec;
- `kind: model` → `write_model_capsule` — a whole network lowered end to end.

All three are **additive**: without the m2m venv or `specir` they skip loudly and the direct-MLIR
corpus still regenerates. A capture that *starts* and then fails still raises.

One guard here is worth a paper sentence. `build()` refuses an epilogue the builder did not carry:
only `matmul` and `conv2d` read the entry's `epilogue:`, so a stage declared on, say, a `movement`
entry used to vanish from the capsule while `_semantic_block` still credited that stage's *family* in
`composed_families`. The capsule would then count as evidence for arithmetic no engine ever performed.

---

## 3. The four golden regimes

`_entry_regime` picks the engine; there is no default and no fallback.

| regime | engine | datapath | compare |
|---|---|---|---|
| `int` | `capsule_golden.golden` — dependency-free `Tensor` primitives | exact int matmul, round-half-even `acc_scale`, saturating i8 cast | `exact_int` |
| `specir` | external `specir` fp8/bf16 refmodel | `acc <- round_bf16(acc + round_bf16(a*w))`, k sequential, per-step, RNE | `tolerance_float` |
| `mx` | `mlc.validate.mx_ref` (transcribed from the RTL's own golden) | 16-deep systolic per-column accumulate; one E8M0 scale per 32-element K group; bf16 accumulate | `tolerance_float` |
| `simt` | numpy IEEE float | ordinary IEEE math, fp32 accumulate, format-rounded operands | `tolerance_float` |

Every `golden.yaml` carries an `oracle_provenance` block naming the engine, the datapath, operand and
accumulator dtypes, the grade policy, and the input digests. Two properties matter:

- **The golden is computed from the capsule's declared operation, never from the emitted command
  buffer** — so a wrong command buffer is caught by `golden != reference(cb)` rather than agreeing with
  itself.
- **The reference is independent of the target RTL.** `specir` is an external refmodel; the MX engine
  is transcribed from the hardware's own golden and is bit-exact against spike; the SIMT engine is
  plain IEEE. None of them is a self-oracle.

The float path additionally decodes operands **the way the datapath does** — if the unit flushes
subnormals, the golden flushes them too, so the two references implement one datapath rather than two.

---

## 4. Determinism

Leaves are materialized by `Tensor.deterministic` (`runtime/tensor.py:63`), which fills from the
tensor's *name* with no RNG, indexed by `(row, col)` rather than flat position — so rows and columns
differ and a row-stride, offset or transpose bug changes the output. The command-buffer materializer
and the device harness call the same function, so L0 cannot silently diverge from L2/L3 on leaf data.

The default range is `lo=0, hi=3`. **That is a real limitation and should be stated in any paper
section**: four of 256 i8 values, all non-negative, so sign, saturation and overflow behaviour are
untested by construction.

Regeneration is byte-stable: scrubbing rewrites a file only when its content actually changes, and a
golden cache keyed on a source digest (`_golden_cache_key`, `:603`) avoids recomputation without
affecting output.

---

## 5. Fail-closed rules

These are the load-bearing design decisions — the part worth a paper subsection, because each one
replaced a measured failure.

1. **A requirement that produces no capsule is an error, not an omission.** `corpus_synth` raises,
   naming the cell, when no available op expresses a required family. A silently-empty requirement is
   indistinguishable from a satisfied one.
2. **Scrub per capsule, not per corpus.** Scrubbing at the end means one unrelated failure aborts the
   run with every capsule so far still carrying its absolute `prov.weights_file`. Measured: a run that
   died on the last entry left `/scratch/.../weights.safetensors` in tracked MLIR across six capsules,
   in a repo that is published. Hygiene that only holds on the happy path is not hygiene.
3. **One failing capsule must not destroy the corpus.** Failures are collected, reported by name, and
   re-raised at the end. A capture `torch.export` refuses used to take every later entry down with it,
   leaving a coverage gap for an unrelated reason.
4. **A roster model that cannot be captured is recorded, never covered.** Whether a network can be
   captured depends on things outside the repo — a missing dataset, a package the venv lacks, a scheme
   torch.export refuses. Those are recorded as `not_built` with the reason, and **no capsule exists to
   be graded**, so nothing can pass in its place.
5. **A host-lane capsule whose forbid is not provable is deleted, not left on disk.** A directory the
   corpus does not list is exactly the half-written state a seal cannot see, and the next glob would
   pick it up as though it had been built.
6. **Budget overruns raise; they never truncate.**
7. **An epilogue the builder dropped is refused** (§2, stage 7).

---

## 6. Labels, holdouts and answer surfaces

`label` partitions the corpus into `public`, `dev` and `hidden`. The holdout sidecar
`profiles/<target>.hidden.yaml` is itself gitignored. `MANIFEST.yaml` is tracked and sits inside the
tree every experiment arm is granted read-only, so holdouts are **counted, never named** — listing a
`hidden/<capsule>` path would tell the agent under test the op family of a held-out case.

`_perf` capsules are a separate phase: category `_perf`, label `dev`, and
`TargetExperiment.corpus_siblings` skips any directory whose name starts with `_`, so they never enter
the functional grade.

---

## 7. Provenance

`source_role` is a closed enum in the schema. Measured across the tree today:

| `source_role` | count | who writes it |
|---|---:|---|
| `derived_sweep` | 312 | `corpus_synth` (prefix `SY`) |
| `pytorch_model_slice` | 187 | `model_slice_export` / `capsule_source` |
| `handauthored_compiler_test` | 79 | typed by hand in a profile |
| `model_derived` | 40 | `corpus_synth` model axis |
| `uplifted_from_bareMetalC` | 9 | ported from the vendor test suite |
| `spec_derived` | 3 | hand-written, spec-referenced |
| `smt_counterexample` | 0 | `verify/counterexamples.py` |
| `behavioral_specimen` | 0 | *no producer — enum only* |

`update_provenance_manifest` **merges, never replaces**: a path this run emitted becomes `generated`,
everything else keeps its classification and defaults to `hand_authored`. That ordering matters —
rebuilding from scratch would reclassify the frozen hand-authored source-of-record (A1, B3/B4, the
held-out set) as generated the first time a run happened to emit something at the same path.

One consequence to design around: `_prune_superseded_synth` deletes on-disk capsules whose
`source_role == derived_sweep` and which the regenerated profile no longer asks for. **An externally
imported corpus stamped `derived_sweep` would be auto-deleted by the next generation run.**

---

## 8. Measured state (2026-09-07)

630 `capsule.yaml` files across the whole contract tree. Graded corpora per target:

| target | graded capsules | dominant source |
|---|---:|---|
| radiance | 166 | `pytorch_model_slice` 75, `derived_sweep` 56 |
| gemmini | 113 | `derived_sweep` 44, `pytorch_model_slice` 40 |
| atlas | 62 | `pytorch_model_slice` 30, `derived_sweep` 22 |
| mx_gemmini | 38 | `pytorch_model_slice` 17, `derived_sweep` 15 |

By label: 467 public, 114 dev, 49 hidden. By kind: 275 `isa`, 233 `model_slice`, 96 `layer`, 26
`model`. 18 profile files across six targets.

**More than half the corpus (352 of 630) is machine-derived** — `derived_sweep` plus `model_derived`.
That is the number the "which capsules exist is derived too" claim rests on.

---

## 9. Diagram recipes

Three figures the above supports directly.

**Figure A — the pipeline (main diagram).** A left-to-right flow in three bands:

- *Facts* (left, cylinders): RTL facts, capability manifest, target descriptor, captured models,
  workload spec.
- *Derivation* (middle, boxes): `conformance.spec` → `conformance/<target>.yaml` → `corpus_synth` →
  `<target>.synth.yaml`; alongside it the four other profile sources feeding the merge box.
- *Materialization* (right): merge → `derive_binding` → `expand_sweeps` → `build()` → golden engine →
  bundle → scrub → MANIFEST.

Draw the binding as a *side input* to `expand_sweeps`, `build()` and the golden engine — it feeds all
three, which is what makes the pipeline target-agnostic. Mark `golden.yaml` and
`expected_instruction_coverage.yaml` with a distinct fill and a legend entry "answer surface —
untracked, masked from the agent".

**Figure B — the four golden regimes.** A 2×2 or single-column table-figure keyed by regime with the
engine, datapath expression and compare policy from §3. The point the figure should make visually is
that every arrow ends at an *independent* reference, never at the target's own output.

**Figure C — where a capsule comes from.** A stacked bar per target (radiance / gemmini / atlas /
mx_gemmini) segmented by `source_role`, using §8. This is the figure that shows derivation dominating
hand-authoring.

For all three: the repo's plotting conventions are in `docs/guides/paper-figures.md` and
`figures/paper_plot_style.py`.

---

## 10. What the pipeline does not do

Stated so a paper section does not over-claim:

- **It does not test what the stimulus cannot reach.** `lo=0, hi=3` means no negative operands and no
  saturation or overflow coverage.
- **It does not model residual accelerator state.** No axis expresses what was in the scratchpad
  before an operation, which is how a padded-convolution defect that reads stale scratchpad survived a
  fully covered corpus (see [compiler_verification](compiler_verification.md), 2026-09-07).
- **It does not import foreign corpora.** There is no importer; measuring an external tree works
  because every coverage function takes `corpus_roots`, but registering one needs a new sidecar in the
  merge chain and a new `source_role` (§7).
- **Coverage is not fault detection.** The corpus is measured against a derived requirement over eight
  axes; occupying every cell is necessary, not sufficient.
