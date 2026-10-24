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
