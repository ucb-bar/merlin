# Gemmini: Phase 0 — inputs, derivation, and review

Start with [the experiment definition](../experiment.yaml), catalog ID
`gemmini-functional`, runtime target `gemmini`.

**Status: this example is not yet a completed, verified Phase 0.** The
historical selected conformance and synthesis references are useful for
inspection but lack complete application-op provenance. Their original seven
derivation captures contained 245 unresolved declaration-only calls. Fresh
capture bundles now eliminate those raw opaque calls and bind materialized
inputs/weights with producer receipts; a new requirement, exact operation
mapping, generated corpus and reviewed release still have to be selected and
executed. Do not promote the
historical references by editing their YAML or treating an old capsule tree as
source. See the [artifact map](../artifacts/README.md) for where each generated
step belongs.

| Authored here | Why it cannot be read from CIRCT or captures |
| --- | --- |
| `datapath` in [`recipe.yaml`](recipe.yaml) | Software-visible exact-integer oracle policy, integer requant shift and oracle tiers are independent requirements, not structural RTL facts. |
| 50 seed cases and two tile-relative sweeps | They assert specific numerics, placement, host seams and corner behavior. Preserve them until a reviewed generated corpus demonstrates equivalent coverage. They are **not** generated capsules. |
| [`target/descriptor.yaml`](../target/descriptor.yaml) | Declares derivation applications separately from held-out claim models, experiment resources and budgets. Workload intent is an authored choice. |
| [`target/contracts/`](../target/contracts/target_contract.yaml) | Supplies reviewed semantic interpretation and extraction anchors; a structural opcode alone does not prove software behavior. |

The two small, authored whole-model loaders live in [`inputs/`](inputs/README.md).
The tracked copies inside older `merlin/contract/capsules/model/` trees are
historical capsule snapshots; new Phase 0 runs select the example inputs.

Generated RTL facts, application-op inventory, conformance requirement,
synthesis plan, capsules and run receipts live under the configured `out` root.
To inspect a selected capture as it passes through whole-model MLIR and LLVM
lowering, use the separate [whole-model inspection walkthrough](../whole-model/README.md).
That audit explains the IR changes, but does not by itself qualify Gemmini offload.
The recipe is still larger than an ideal minimal software spec: much of its
case roster predates reliable all-op synthesis. Remove a seed only after the
new generator and an independent oracle show that its semantic and placement
checks survive; line-count reduction alone is not evidence of coverage.

Quantization is derived from the target's declared operand format and readout
facts, then applied through TorchAO's public quantizer extension points. The
selected static recipe quantizes only contractions with a floating stored
weight and floating activation in an owned supported module. It does not
quantize an entire network just because the target has int8 instructions:
LSTM recurrence, unsupported layers, and runtime-by-runtime matmuls remain
floating host work unless separately proved and lowered. `quant_recipe` can
inventory multiple declared hardware formats, but this is an evidence list,
not mixed-format capture: a candidate still needs format-specific readout,
framework capture, compiler lowering and numerical agreement before use.
Merlin does not patch TorchAO source. A floating-point path is not itself a
TorchAO quantization format.

When preparing a reviewed corpus release, pass the example's
[`retirements.yaml`](retirements.yaml) as `--retirements` if the historical
baseline still contains generated members omitted by the fresh derivation.
Preparation checks the list exactly and removes only release-local copies;
it does not edit the old source tree or make old capsules count as current
coverage. The review record binds each decision and previous byte digest.

`M2_microvit_gemmini` is an authored model whose loader already performs
integer GEMMs between floating-point host operations. Its
`capture_quantization: already_materialized` declaration tells the generic
capture worker to preserve that graph. The worker records that it applied no
TorchAO recipe, and the capsule writer requires actual integer contractions in
the imported MLIR. Inspect the generated model capsule's `capsule.pytorch.py`,
`capsule.interface.mlir`, and `capsule.yaml` together: the last file records
the capture mode and the measured contraction count. This is a capture check;
the later lowering stages and runtime result need their own run receipts.

`M3_host_island_seam_gemmini` uses the same capture mode: its two int8
contractions are authored around a floating LayerNorm host island. Its active
loader is under [`inputs/`](inputs/host_island_seam.py), while the old capsule
directory remains a historical snapshot. The generated capsule records two
materialized contractions and an `i8` output with exact-integer comparison;
the saved host Torch golden is used even though the output is integer.

The 50 entries break down as follows; this is a map of **authored seeds**, not
a count of generated capsules:

| Seeds | Count | Authored reason |
| --- | ---: | --- |
| `A0`–`A7` (no `A1`), `GS0` | 8 | ISA/ABI trace, accumulation, resident reuse, edge padding, integer readout and spec-origin oracle |
| `B0`–`B2`, `C0`–`C7` | 11 | PyTorch linear, attention and quantized numerical paths |
| `GF1`–`GF5`, `GN0`, `GC1`–`GC6` | 12 | Host-only/non-mesh algorithms and negative placement assertions |
| `GC0`, `GC7`–`GC8`, `GQ0`–`GQ3`, `GB0`, `GM0`–`GM1`, `GP0`–`GP2` | 13 | Conv geometry plus fused numerics, rank, store-pressure and pooling-window corners |
| `M0`–`M4`, `GX0` | 6 | Whole-model, VLA and accelerator–host composition claims |

First **retirement candidates after a newly derived and reviewed synthesis
artifact** are `GB0` versus its generated batched-rank case, and the
geometry-only `GC7`/`GC8` versus generated padded/strided convolution-window
cases. `GS0` could be generated directly from its selected spec source rather
than named here. None is safe to remove today: for each pair compare resolved
shape/layout, source identity, independent golden, required trace/placement
assertions and oracle tier in the actual generated capsule and receipt.
`GM0`/`GM1` intentionally probe the fit/spill *threshold* with one output tile;
the historical synthesized regime cases use different extents and only L2, so
matching their regime labels is not equivalence. Likewise a synthesized family
cell does not replace `GQ0`–`GQ3`'s ordered numerical epilogue and mesh-placement
checks, nor does a model shape replace an independent whole-model oracle.

[`recipe.yaml`](recipe.yaml) is the authored public coverage recipe, not a generated
corpus or evidence of hardware correctness. The definition explicitly names its shared
`performance_template` and generated `conformance_spec`, `synth_profile` /
`smt_profile` inputs. The paths in the example currently select historical
references for inspection; they are not a verified Phase 0 execution set.
Public capsule entries omit `label: public` and the output category when it follows
`kind` (`isa`, `layer`, `model_slice`, or `model`); the Phase 0 loader restores both
before generation. Tile-relative extents such as `tile+1` resolve from the selected
target binding. Workload choices, software-visible numeric semantics (including the
requantization shift), and independent oracle expectations remain explicit: CIRCT
structure alone cannot establish them.
Private `hidden_profile` stays outside examples; never copy holdouts, goldens, weights,
or generated capsules here. Optional sidecars may be absent; frozen runs bind their
presence as well as their bytes.

## Follow one derivation, step by step

The inputs have different owners. The [target contract](../target/contracts/target_contract.yaml)
provides extraction anchors and a reviewed interpretation of the hardware's semantic
capabilities; that interpretation is **not** automatically proved by seeing an RTL opcode.
The [descriptor](../target/descriptor.yaml) declares `workload_spec.applications` (model
captures used to *derive* test shapes) and `workload_spec.models` (held-out models used
to *evaluate* generalization). It also declares a precision preference and a
certification budget; neither can add a capability the hardware contract does not
admit. This [recipe](recipe.yaml) supplies the remaining software-visible choices,
including the integer `requant_shift: 3`, operation semantics, and authored corner
cases. CIRCT cannot infer a rounding rule or an independent golden result.

1. **Extract structural facts from the selected elaboration.** With a Gemmini HW-dialect
   MLIR file and the target's OOT support available, run from the repository root:

   ```sh
   export MERLIN_MLC_DIR=/path/to/mlc
   export MERLIN_EXT_CHIPYARD=/path/to/chipyard
   export MERLIN_TARGET_PATH=/path/to/gemmini-mlir/merlin-support
   python -m merlin.targetgen.rtl.circt_introspect --target gemmini \
     --hw /path/to/gemmini.hw.mlir \
     --out out/artifacts/cache/rtl_introspect/gemmini/facts.json --validate
   python -m merlin.targetgen.rtl.mlc_bridge gemmini --json
   ```

   The first command writes a purgeable `facts.json` with the extracted array,
   memory, interface, and decoder evidence. The second prints a provenance-tagged
   fact brief: each field has `value`, `source`, and `derived`; unavailable facts
   remain unavailable. Do not copy either output into this example as an authored
   input. A different elaboration requires a new extraction and review. A
   successful extraction establishes structural facts and source hashes, **not**
   complete software semantics, capsule coverage, or hardware certification.

2. **Inventory all application operations, then derive the requirement.** The generic
   conformance code intersects semantic family/dtype pairs admitted by the reviewed
   compute-unit contract with captured applications and adds boundary classes from
   extracted facts. It also inventories *every* operation in each application,
   including host work and unresolved external calls. To audit the historical
   selected reference, run:

   ```sh
   python build_tools/scripts/check_conformance_coverage.py --target gemmini \
     --spec experiments/reference-data/phase0/conformance/gemmini.yaml \
     --json --fail-on-unverifiable
   ```

   The selected [conformance reference](../../../experiments/reference-data/phase0/conformance/gemmini.yaml)
   records the derivation formula, inputs read, refusals, and boundary provenance.
   Its snapshot reports `tile_edge: 16`, sourced from RTL `arrays[].rows`, and
   12 required family/dtype/alignment cells. That is **not** application-op
   coverage: the historical reference predates the detailed inventory and the
   strict audit now exits 2. An unavailable declared capture is an error, never
   an empty workload. When all captures and the required lowering/support
   semantics are available, write a *new* requirement under your configured
   `out/artifacts/verification/` with an explicit complete capture roster:

   ```sh
   python build_tools/scripts/check_conformance_coverage.py --target gemmini \
     --application-capture deepjscc_int8_consistent=/capture/deepjscc_int8_consistent/model.mlir \
     --application-capture smolvla_fp32_consistent=/capture/smolvla_fp32_consistent/model.mlir \
     --application-capture smolvla_int8_consistent=/capture/smolvla_int8_consistent/model.mlir \
     --application-capture smolvla_denoise_step_fp32_app=/capture/smolvla_denoise_step_fp32_app/model.mlir \
     --application-capture lstmnetvit_fp8_consistent=/capture/lstmnetvit_fp8_consistent/model.mlir \
     --application-capture lstmnetvit_int8_consistent=/capture/lstmnetvit_int8_consistent/model.mlir \
     --application-capture lstmnetvit_int8_w8a8_consistent=/capture/lstmnetvit_int8_w8a8_consistent/model.mlir \
     --write /configured/out/artifacts/verification/gemmini/REVIEWED.yaml
   ```

   Replace each `/capture/` path with its actual versioned capture directory;
   labels must match the descriptor exactly. An explicit selected bundle needs
   a valid capture-time materialization receipt, not only an MLIR file. `--write`
   also generates an
   adjacent `*.application-demands.json` inventory; review both files, then
   select the new requirement explicitly for a fresh experiment run. Neither
   generated file belongs in `examples/` or in Git. The CLI override shown
   below pins the selection without making a hand-edited definition in `out`.

   Until then, use `--inventory-out PATH` for a diagnostic inventory. It can
   report unresolved calls without publishing a requirement. The historical
   captures contained 201 ATen calls in SmolVLA and 44 TorchAO calls in
   LSTMNetViT. In fresh bundles the calls lower generically; the TorchAO calls
   in particular were caused by an exporter fallback after a range-constraint
   serialization error. The fresh SmolVLA denoise bundle also restores three
   lifted constants that the earlier bundle omitted. Capture-time receipts
   bind files and materialized input ABI, while source dependency closure and
   compiled whole-model execution remain separate checks.

3. **Synthesize proposed cases.** `synth_capsule_corpus.py` reads the *explicitly
   selected* conformance artifact plus the descriptor's workload preferences.
   Its `--json` mode shows proposed entries, application-op obligations and
   refusals without selecting them. With the historical incomplete reference it
   exits 2 after showing a diagnostic plan; `--write` requires a complete
   inventory and creates a review artifact under `out`, not committed source:

   ```sh
   python build_tools/scripts/synth_capsule_corpus.py --target gemmini --json
   python build_tools/scripts/synth_capsule_corpus.py --target gemmini \
     --conformance-spec /configured/out/artifacts/verification/gemmini/REVIEWED.yaml \
     --write
   ```

   The historical synthesis reference records 62 entries but lacks the new exact
   input digests; it is diagnostic and must be regenerated, reviewed and selected
   with a newly frozen run before verified execution. An unresolved case stays
   visible as a refusal, never silently counted as coverage. The authored recipe
   still carries software-visible semantics and corner cases that hardware facts
   cannot derive.

4. **Generate and review the corpus.** A [definition](../experiment.yaml) selects
   the recipe and shared performance template; explicit CLI overrides select a
   newly derived conformance/synthesis pair for this run. `merlin experiment run
   ... --phase 0` resolves tile-relative extents,
   writes capsules and independent goldens into that run, and records the exact
   input identities. Capsules are *only* run artifacts: regenerate them from
   reviewed inputs rather than committing or hand-editing them. Review/seal a
   functional corpus for Phase 1; Phase 2 selects its separate performance
   workloads and the frozen functional compiler. Never silently reuse a
   functional-corpus identity as a performance-corpus identity.
   The review/seal commands below bind a fresh run; changing any selected source
   means freezing a new run, not resuming an old one as verified.

For a concrete thread through these steps: the selected facts give a 16-row tile;
the reviewed contract admits `contraction/int8`; captured workloads show that
family at a partial boundary. The conformance reference therefore requires
`contraction/i8/partial`. Synthesis records
`SY_contraction_i8_partial` with `M: tile`, `K: 2*tile-1`, `N: tile-1` and a
`source_reference` naming that cell. On this reviewed 16-row snapshot the written
shape is 16×31×15. The hardware edge and observed workload justify the *shape*;
the recipe/oracle path still defines how its integer result is judged.

## Inspect a generated run and test source-pool coverage

For the run command below, generated capsules live at
`/configured/out/runs/gemmini/phase0/example-1/phase0/capsules/`. The path is
run-owned, not an authored directory. Its `MANIFEST.yaml` records generated and
authored members, performance-generation decisions, and a count-only
`claim_model_evaluation` obligation. Held-out `workload_spec.models` produce no
public Phase 0 capsule; the owner evaluates them only after freezing Phase 1.
`phase_corpora` names three disjoint selections:
`phase1` functional conformance, `phase2` performance optimization and
`diagnostic` Phase 0-only members. Each selection has a purpose, member list and
selection digest for generated public members; private holdouts have separate
owner-side provenance. The frozen run/release binds the actual bytes. Each member
has a generated `README.md`,
`capsule.yaml` (operation, numeric policy, expectations and source reference),
`capsule.interface.mlir` (compiler input),
`expected_instruction_coverage.yaml`, and an owner-side `golden.yaml` when its
source can provide one. Model captures may have additional sidecars; inspect
the member's actual files rather than assuming every source uses one format.

```sh
CAPSULES=/configured/out/runs/gemmini/phase0/example-1/phase0/capsules
sed -n '1,140p' "$CAPSULES/MANIFEST.yaml"
rg --files --no-ignore "$CAPSULES" | rg '/capsule.yaml$'
sed -n '1,160p' "$CAPSULES/isa/SY_contraction_i8_partial/README.md"
sed -n '1,200p' "$CAPSULES/isa/SY_contraction_i8_partial/capsule.yaml"
sed -n '1,120p' "$CAPSULES/isa/SY_contraction_i8_partial/capsule.interface.mlir"
```

`source_reference` links a synthesized member to its required conformance
cell. Do not expose `golden.yaml`, private sidecars, or owner-only diagnostics
to a compiler candidate. To measure the **actual completed run's public
source pool** against an explicit requirement, rather than auditing a legacy
descriptor-selected corpus, run:

```sh
merlin experiment corpus coverage \
  /configured/out/runs/gemmini/phase0/example-1 \
  --spec /configured/out/artifacts/verification/gemmini/REVIEWED.yaml
```

The JSON names missing and extra cells and reports the other measured axes
(composition, geometry, host-only work, and so on). It verifies the Phase 0
execution receipt and frozen inputs, and hashes the supplied requirement.
Presence in this source pool is **not** numerical correctness, cohort admission,
L3 certification, or evidence that a whole network compiles. Review and seal
the corpus separately, then use Phase 1's functional evidence for those claims.

The authored descriptor lives at
[`target/descriptor.yaml`](../target/descriptor.yaml); its `resources_root`
still selects retained native resources.
This is not yet a self-contained target package. Install Merlin plus
`merlin-experiments`, provision the selected OOT support and the descriptor's declared
inputs, and make any required capture/toolchain dependencies available before running.
An example recipe does not establish simulator, RTL, hardware, or model-download availability.

Inspect first; then use fresh paths under your configured output root. Replace
`/configured/out` below with that root:

```sh
merlin experiment inspect gemmini-functional --phase 0 \
  --phase0-conformance-spec /configured/out/artifacts/verification/gemmini/REVIEWED.yaml \
  --phase0-synth-profile /configured/out/artifacts/verification/gemmini/SYNTH/synth.yaml \
  --phase0-hidden-profile /operator/private/gemmini.hidden.yaml
merlin experiment preflight gemmini-functional --phase 0 \
  --phase0-conformance-spec /configured/out/artifacts/verification/gemmini/REVIEWED.yaml \
  --phase0-synth-profile /configured/out/artifacts/verification/gemmini/SYNTH/synth.yaml \
  --phase0-hidden-profile /operator/private/gemmini.hidden.yaml
merlin experiment run gemmini-functional --phase 0 \
  --phase0-conformance-spec /configured/out/artifacts/verification/gemmini/REVIEWED.yaml \
  --phase0-synth-profile /configured/out/artifacts/verification/gemmini/SYNTH/synth.yaml \
  --phase0-hidden-profile /operator/private/gemmini.hidden.yaml \
  --run-dir /configured/out/runs/gemmini/phase0/example-1
merlin experiment corpus prepare /configured/out/runs/gemmini/phase0/example-1 --output /configured/out/artifacts/protocols/gemmini-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/gemmini-review-1
```

The private profile is an operator-owned input outside examples/Git; omit the flag
only for a deliberately public diagnostic run, not for an admission release
that requires hidden coverage. The source descriptor declares
`grading.release_admission: derive_from_corpus_v1`: release preparation measures
capability exclusions and public/hidden counts from the staged corpus, writes
them to its generated descriptor, and refuses any model not explicitly covered
by its authored resource policy. The source descriptor itself is not a grading
cohort, and a missing hidden corpus blocks formal admission.

Stop for human review of the prepared inputs and owner-only diagnostics. **Only after
review**, acknowledge the exact inspected digest:

```sh
merlin experiment corpus seal /configured/out/artifacts/protocols/gemmini-review-1 \
  --expected-digest DIGEST_FROM_INSPECT --reviewed-by OPERATOR --review-note REVIEW_SUMMARY
```

The seal records a review acknowledgement, not numerical or hardware certification.
It does not automatically approve or start Phase 1. Follow the
[shared reviewed handoff](../../../experiments/README.md#reviewed-phase-0-handoff)
to select the reviewed release explicitly; retain old run receipts unchanged.

The legacy fixed C0–C6 model-slice recipes and golden exporter are target-owned
support, now `gemmini_conformance.model_slices` in the Gemmini OOT repository's
`merlin-support/`. They are not a shared Merlin API or the catalog's default corpus.
The retained performance corpus-authoring scripts require that host-private package
on `PYTHONPATH`, with the same support directory selected by `MERLIN_TARGET_PATH`.
Never grant it to compiler candidates. Generic MLIR emission remains in
`merlin.targetgen.contract.matmul_interface` and requires an explicit target.
