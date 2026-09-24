# Gemmini: Phase 0 — hardware-guided test generation

Start with [the experiment definition](../experiment.yaml), catalog ID
`gemmini-functional`, runtime target `gemmini`.

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
   python -m merlin.targetgen.rtl.circt_introspect --target gemmini \
     --hw /path/to/gemmini.hw.mlir \
     --out out/artifacts/cache/rtl_introspect/gemmini/facts.json --validate
   python -m merlin.targetgen.rtl.mlc_bridge gemmini --json
   ```

   The first command writes a purgeable `facts.json` with the extracted array,
   memory, interface, and decoder evidence. The second prints a provenance-tagged
   fact brief: each field has `value`, `source`, and `derived`; unavailable facts
   remain unavailable. Do not copy either output into this example as an authored
   input. A different elaboration requires a new extraction and review.

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
   `out/artifacts/verification/` with `--write PATH`. This also generates an
   adjacent `*.application-demands.json` inventory; review both files, then
   select the new requirement explicitly in your experiment definition. Neither
   generated file belongs in `examples/` or in Git.

   Until then, use `--inventory-out PATH` for a diagnostic inventory. It can
   report unresolved calls without publishing a requirement. The current seven
   declared captures contain 245 unresolved declaration-only calls: 201 ATen
   calls in SmolVLA and 44 TorchAO calls in LSTMNetViT. These need real generic
   lowering or support semantics and recapture; naming an opaque call as covered
   would not repair Phase 0.

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
   the recipe, shared performance template, conformance artifact and synthesis
   artifact. `merlin experiment run ... --phase 0` resolves tile-relative extents,
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
authored members, performance-generation decisions, and any model-roster cases
that could not be built. `phase_corpora` names three disjoint selections:
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
  --spec experiments/reference-data/phase0/conformance/gemmini.yaml
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
merlin experiment inspect gemmini-functional --phase 0
merlin experiment preflight gemmini-functional --phase 0
merlin experiment run gemmini-functional --phase 0 --run-dir /configured/out/runs/gemmini/phase0/example-1
merlin experiment corpus prepare /configured/out/runs/gemmini/phase0/example-1 --output /configured/out/artifacts/protocols/gemmini-review-1
merlin experiment corpus inspect /configured/out/artifacts/protocols/gemmini-review-1
```

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
