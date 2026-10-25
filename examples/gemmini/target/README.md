# Gemmini target: start with the software spec

**Provide [`software-spec.yaml`](software-spec.yaml) to describe the accelerator's
software-visible functionality.** It is the user-authored semantic entry point.
Keep host support, hardware selections, extracted facts and generated tests separate.

For an existing platform, edit the software spec and select it through Phase 0's
`config.software_spec`. A new platform also needs public RTL/ISA sources and an
explicit execution selection from its integrator; a software spec alone cannot
qualify a simulator or deployment.

## What the user declares

Use `schema: merlin.software_spec.v1`, the target identity, and the named-operation
mapping shown in this example. Declare these properties once:

| Section | What to supply |
| --- | --- |
| `numerical_semantics` | Operand, internal arithmetic, stored accumulator and readout types; overflow, rounding and narrowing behavior. |
| `operations` | Mathematical operation families and accepted types, ranks, layouts, tails, batching and aliasing. A `hardware` request is resolved against selected evidence. |
| `restrictions` | Explicit limits on those declarations, such as an unsupported epilogue. |
| `quantization` | Software-visible formats and scale encoding. `eligible_operations: from_facts` requests derivation from hardware evidence. |
| `transfer_contracts` | Logical typed copies across the host/device boundary and valid-output requirements. |
| `status` | The declaration's actual review state. Review does not establish execution. |

For example, contraction denotes multiply-accumulate:
`C[i,j] = sum_k A[i,k] * B[k,j]`, independently for each admitted batch.
Its exact numerical meaning includes intermediate overflow. This example keeps
its 20-bit MAC result separate from 32-bit accumulator storage; an exact integer
reference requires the stated partial-sum bound. Readout has its own scaling,
rounding and saturation policy.

Use the existing operation-family vocabulary and explicit constraints. Do not
invent a free-text formula field: the current schema does not interpret one.
The shared [software-spec guide](../../../docs/guides/phase0_specification.md)
describes normalization, admission and unresolved declarations. The six accelerator
requests retained here are declarations; some need further evidence to resolve.
Host operation support is declared once in the selected host manifest.

## Where the other files belong

```text
examples/gemmini/
  target/
    software-spec.yaml       user-authored accelerator semantics
    descriptor.yaml          experiment and integration selections
    contracts/               retained prototype contract references
    evidence_concepts.yaml   evidence-discovery vocabulary
  platform/                  RTL, board and host selections
  phase0/                    test-generation policy and instructions
  verification/              operator-side probes
out/                         generated products and execution receipts
```

The [platform guide](../platform/README.md) identifies each integrator-owned input.
[`descriptor.yaml`](descriptor.yaml) selects those files explicitly; it also owns
the iteration/validation separation and experiment policy. The retained contract
references and evidence vocabulary are integration metadata. They are not additional
software-spec fields, generated facts, or a qualified runtime implementation.

## What the tooling derives

Given selected sources and tools, the ordinary Phase 0 flow derives source-bound
RTL facts, checks software declarations against those facts, captures independent
public examples, derives operation/edge requirements and writes capsules, coverage
reports and release evidence. ISA fields and effects need source-backed derivation
or explicit public declarations; missing meaning remains `UNKNOWN`.

Facts, their adjacent `validation.json`, ISA bindings, captures, goldens, capsules
and qualification receipts are generated under `out/`. Do not author or copy these
into this folder. A selected execution route still needs qualification on the exact
selected deployment. No device kernels, packing algorithms or schedules belong in
the software specification or runtime input package.

## How to supply and use the spec

1. Adapt `software-spec.yaml` for the selected accelerator semantics. Mark new or
   changed declarations unreviewed until their review is recorded.
2. Select its path as `phases.0.config.software_spec` in the experiment definition.
   Keep platform paths explicit and freeze their source identities.
3. Follow the [Phase 0 walkthrough](../phase0/README.md). For a fresh AWS run, use
   the [neutral worker preparation guide](../../../docs/guides/phase0_phase1_setup.md).
4. Run the reviewed Phase 0 definition into a fresh owner, inspect its mandatory
   coverage and unresolved decisions, then review and seal the release.

Rows whose evidence is still authored are marked `authored_pending_operator_review`, including
new native component families, widened host signatures, the CPU special-value policy, and the
native-unit declarations. They are not operator-reviewed facts.
`MERLIN_ACCEPT_AUTHORED_PENDING_OPERATOR_REVIEW=1` explicitly accepts these pending declarations
for screening; the default is `1` for compatibility with the existing placement policy. Set it
to `0` to refuse pending declarations. Other values are errors. Unreviewed declarations are
never promoted by this switch. Admission records retain the pending review status and report
`operator_accepted` plus whether the compatibility default or an explicit environment selection
was used. Numerical, source-coverage, compiler and execution gates remain mandatory.

`execution_provider_revision` selects the provider revision used by the pinned E2E gate.
The native-family specification and source-bound provider proof determine placement; performance is deferred to Phase 2.

Run the reviewed definition:

```sh
merlin experiment run "$REVIEWED_PHASE0_DEFINITION" --phase 0 --run-dir "$NEW_RUN"
```

The catalog's retained `gemmini-functional` definition is diagnostic. Its presence
does not establish verified readiness. Phase 1 starts separately after release
review and exact deployment qualification.

For source production, RTL audits and finite numerical/build probes, use the
[verification walkthrough](../verification/README.md). Those probes have one
canonical home; their historical receipts retain their original paths and hashes.

## Seal 4 operator decisions

Loren Hung approved these family/axis changes for seal 4 (2026-10-24). This
review authorizes the declarations; source facts, proofs and execution audits
remain mandatory. The first four comment lines and `execution_provider_revision`
retain their original software-pin meaning. Historical seals are never resealed.

| Decision | Declaration and admission constraint |
| --- | --- |
| S4.1 | `q_pow2_tensor` is the unchanged default/baseline: per-tensor power-of-two observer scales, exact-rational host RNE and single-rounding residuals. New schemes are explicitly selected; they do not change a legacy recipe. |
| S4.2 | `f32_product_rne` means RNE conversion of i32 to f32, one exact-ratio-to-f32 multiplier formation, f32 RNE product, then RNE and saturation. Its complete definition is under `int8_readout.semantics.readout_definitions`. Device admission compares decoded multiplier bits with the formed multiplier. Exact-rational schemes require `prove_fp32_products` equality or retain full-i32/host readout. The old unrestricted int8-readout D6 claim is superseded. Non-unit f32 execution requires the pinned rounding-fixed oracle; a declaration supplies no execution verdict. |
| S4.3 | Arbitrary-f32 tensor, column-block and per-operand schemes use `declared_observer` through scheme-derived recipes; baseline retains `enclosing_power_of_two`. Channel scales are graded only as blocks of `columns_per_scale` derived from the output path (DIM on the selected hardware). Per-column command partitioning is possible but ungraded. Tensor scale is a compatibility choice, not the finest legal readout scale. Per-operand round-and-saturate is offered on every residual add, with each load multiplier in (0, 1]. |
| S4.4 | `contraction_readout` covers bias preload, requantization and ReLU (F1), raw i32 with no readout stage (F2), and convolution/ReLU/max-pool (F4). Readout stage order and limits come from facts. Padded pooling refuses while padding is UNKNOWN. |
| S4.5 | `scaled_operand_sum` covers residual-on-load (F3) under either declared rounding boundary. Exact single-rounding needs proof; per-operand rounding is definitional. |
| S4.6 | `window_reduction_readout` covers max-pool values with dead indices, only in composition truth; the standalone Core ATen selection row remains unchanged. Scratchpad-source pooling requires a fence. Average pooling uses a ones contraction and the reciprocal window count under the selected readout scheme (F5). |
| S4.7 | Output-stationary pointwise contractions (F6) require a constant-weight, all-partial-sums value bound; unknown/runtime weights or an exceeded bound refuse that route. Integer-coefficient residual K-extension (F7) is required when its exact integer coefficient and extended-K bounds hold; noninteger coefficients refuse. |
| S4.8 | Shared-input siblings retain independent exits/scales, one immutable resident input, and a fact-derived capacity bound. Packed weight panels have logical order `[N/w][K][w]`, with provider-declared `w` from facts. No model identity selects either optimization. |
| S4.9 | Composition truth grades every member in the same accelerator region plus instruction audit, including zero outside-kernel instructions and zero host arithmetic consuming intermediates. Bytes alone do not establish fusion. |
| S4.10 | Held-out variants are removed from the active spec. `../phase0/history/software-spec-seals1-3.yaml` retains the complete pre-edit spec for historical builders. `descriptor.yaml` explicitly withholds `phase0/history`, the historical `_model_layers` tree and its shape-bearing provenance manifest via `answer_surfaces.held_out_inputs`. The sandbox masks these beneath broad grants. New seals use the existing generated-only corpus preparation default; the existing reviewed retirement list also excludes the named model layers for historical overlays. No historical capsule or frozen seal is edited. |
