# Constructing an authored scheduling contract

The selected software specification may describe legal workload interfaces without
providing a dense-scheduling contract. That does not prevent fresh authoring: the
existing `block_schedule` route remains usable with its declared geometry and
normal execution checks. `DenseSourceResult` is an additional checked route whose
specific requirements must actually hold.

Use the same API offline or from the selected compiler library:

```python
from merlin.compile.scheduling.selected_contracts import (
    scheduling_spec_draft, inspect_scheduling_draft, read_scheduling_contracts,
)

# software and optional_isa are explicit public/authored input dictionaries.
# This copies them; it never changes the frozen specification or ISA facts pin.
draft = scheduling_spec_draft(
    software, target=target, isa=optional_isa,
    dense_schema="merlin.dense_schedule.v2",
)
report = inspect_scheduling_draft(
    facts_bytes, yaml.safe_dump(draft).encode(), target=target, require_dense=True,
)
```

`draft` contains the existing schema's keys with nulls for unknown geometry,
formats, resources, instruction bindings, effects, support and completion. Null
configuration is deliberate: `{}` would assert no configuration effects. Fill
fields using explicit facts references and authored semantics; inspect again.
`resolve_fact_integer` in `merlin.targetgen.isa_spec` resolves a supplied reference
against those exact facts. The aid never picks a fact by name similarity, chooses
scratch space, inserts a preferred schedule or changes the ISA's facts hash.

`prepare_scheduling_contract.py` wraps these same APIs:

```sh
python examples/compiler/prepare_scheduling_contract.py draft \
  --facts PUBLIC_FACTS.json --software SOFTWARE.yaml --target TARGET \
  --isa ISA_DECLARATION.json --dense-schema merlin.dense_schedule.v2 \
  --require-dense --output NEW_DRAFT_DIRECTORY

# After editing a separate authored document:
python examples/compiler/prepare_scheduling_contract.py inspect \
  --facts PUBLIC_FACTS.json --software AUTHORED_SOFTWARE.yaml --target TARGET \
  --require-dense --output NEW_INSPECTION_DIRECTORY
```

The report lists exact null-field paths, resolved resources, instructions with
unknown effects/semantics, the ordinary resolver's concrete refusal, and dense
family semantic mismatches. Inputs and report carry their exact byte hashes.
Successful draft inspection means declaration consistency, not execution or
optimization readiness. It returns no selected contract or adapter.

The ordinary `read_scheduling_contracts(facts_path, software_path, target=...)`
already accepts explicit authored paths. Its existing `status: reviewed` marker
and an adapter's `qualification_identity` are assertions/identifiers, not signed
certificates, grader-issued tokens or proof of an independent review. Draft
creation intentionally writes `status: draft`; an author must make the assertion
explicitly when selecting their completed document. There is no invented new
approval step. Normal byte/source freshness, source checks, independent effect
projection and numerical execution remain necessary.

## Logical operations are not raw instruction encodings

The current dense resolver expects distinct logical `lhs_load`, `rhs_load`,
`preload`, `compute` and `store` bindings. It checks their declared memory effects
against local/external resources, coordinate-field representation, selected
formats and ABI. A raw CPU instruction declaration with register-index fields,
`resources: []` and unknown effects only describes transport. It does not provide
those logical operations, an operand-packing implementation or completion.
Do not fill its unknown effects just to make the resolver accept it.

An authored logical adapter must independently match its actual emitted command
sequence: external pointer/stride and tail extents, allocated local addresses,
zero initialization, continuation, output conversion and complete output writes.
A composite logical operation may require several physical instructions; the
binding must represent and check that implementation honestly. Unsupported
logical encoding or effects require an adapter/representation extension, not
made-up hardware facts. Keep transport encoding facts and software packet-layout
assertions distinguishable.

The dense family's supported semantics are exported as `DENSE_SEMANTICS` from
`dense_contracts`; they are requirements of this algorithm, not facts about a
selected accelerator. In particular, `completion: synchronous` means the logical
operation has completed with the required visibility. Queue acceptance is not
completion. An asynchronous target can use an actual synchronous logical wrapper
with proven completion or a separately supported asynchronous route. The draft
tool neither adds a fence nor relabels queued commands synchronous.

Validation of an adapter should include independent emitted-operand decoding,
boundary/tail/reduction cases and numerical execution of the emitted product.
Declaration checks alone establish none of those. See `source_dense_llvm.md` for
how a valid checked result reaches ordinary LLVM code generation.
