---
title: Follow software specifications and RTL facts into compilation
kind: guide
status: current
owner: targetgen
last_verified: 2026-11-09
related: [adding_a_target, target_resolution, cca, kernel_reconstruction, architecture]
code_refs:
  - src/merlin/target/spec/load.py
  - src/merlin/targetgen/software_spec.py
  - src/merlin/targetgen/rtl/introspect.py
  - src/merlin/targetgen/rtl/circt_introspect.py
  - src/merlin/targetgen/rtl/source_selection.py
  - src/merlin/targetgen/rtl/facts.py
  - src/merlin/targetgen/rtl/hw_observations.py
  - src/merlin/targetgen/capability_manifests.py
  - src/merlin/targetgen/spec_fact_drift.py
  - src/merlin/experiments/phase0/evidence.py
  - src/merlin/analysis/rocc_commands.py
  - src/merlin/kernels/cca.py
  - src/merlin/compiler/schedule/derive.py
  - examples/tutorials/01_contracts.ipynb
---

# Follow software specifications and RTL facts into compilation

**Tutorial 1 of 5** | [Start here](../../examples/tutorials/README.md) | [Next: Test generation](test_generation.md)

Open [the contracts notebook](../../examples/tutorials/01_contracts.ipynb) with the
[tutorial launcher](../../examples/tutorials/README.md). It loads a small software
spec and reads a bit slice from HW-dialect text. You can change the request type,
rank, wire width and slice offset and see both results side by side.

The example declares `int8 -> i32` arithmetic. Its HW reader separately reports an
8-bit input, a four-bit slice and an equality comparison with 3. An eight-bit wire
alone does not tell us whether it carries a signed integer, a float or an opcode.
Connecting those meanings is part of target admission.

## Know which file to open

| Input | What it says | Example |
| --- | --- | --- |
| Software spec | Numerical meaning and operation constraints | `examples/gemmini/target/software-spec.yaml` |
| Hardware selection | Which sources and audits to use | `examples/atlas/target/hardware.yaml` |
| Facts artifact | What the selected readers observed, with unknowns and provenance | An explicitly selected `facts.json` |
| Backend capability contract | Compiler/ISA/runtime policy under that selection | An OOT provider's `contracts/target_contract.yaml` |

The retained Gemmini `software-spec.yaml` is reviewed v1 input. Its adjacent
[`spec.yaml`](../../examples/gemmini/target/spec.yaml) is an opt-in, unreviewed compact
v2 candidate, not the default. The Atlas hardware selection is useful reading: it
keeps carrier widths separate from arithmetic-format claims.

`target.spec.load_spec(path)` reads v1 or compact v2 input and retains both the
original source identity and its canonical expansion. `record()` shows those
identities; `verify_sources()` checks the files again. `software_spec()` returns
the existing consumer view when the selected constructs can be projected.
Unsupported projections remain explicit obligations.

The software spec owns numerical types, rounding, accumulation/readout behavior
and supported operation constraints. Application preparation is a separate choice:
`compiler.quantization.capabilities` derives recipe candidates from selected
readout/format information, and the preparation profile selects a supported
framework route. Capture still has to show what arithmetic was actually produced.

Useful software APIs are:

- `numerical_datapath`: read the declared numerical axes.
- `admit_operation`: screen a concrete signature; unsupported and unknown differ.
- `capability_contract`: select explicit same-target backend policy.

A compact spec composes today's numerical, ISA and typed computation records.
It is not yet a way to derive every compiler pass and hardware behavior from one
loop equation.

## Extract facts from a real target

The full path starts with the target's selected configuration:

```text
RTL/configuration -> external elaboration -> FIRRTL + hierarchy + HW
                 -> selected source bundle -> CIRCT/ModelIR readers -> facts.json
```

Prepare the external checkout, elaboration and version-matched tools first. The
census reader needs the declared FIRRTL and hierarchy files. HW discovery also
needs the selected CIRCT/ModelIR environment and target-owned extraction declarations;
`MERLIN_MLC_DIR` selects that external discovery checkout on the MLC route.
These readers do not provision a toolchain or replace a missing elaboration.

Use the actual module entrypoints after those inputs are ready:

```bash
python -m merlin.targetgen.rtl.introspect \
  --target <target> --json-out out/artifacts/cache/rtl_introspect/<target>/facts.json

python -m merlin.targetgen.rtl.circt_introspect \
  --target <target> --source-bundle <selected-source-bundle.json> \
  --out out/artifacts/cache/rtl_introspect/<target>/facts.json --validate
```

`source_selection.load_selection` checks the `core_hw`, `soc_hw`, `firrtl` and
`hierarchy` hashes in the bundle. Production/equivalence receipts check their
correspondence. An explicit `--hw <file>` is another input route; it cannot be
combined with `--source-bundle`.

`circt_introspect.build_facts` collects observations and unknowns; `dump_facts`
writes the result. `--validate` prints agreement, divergence and unknowns against
available declarations/classifiers. The current CLI returns zero after printing
those findings, including divergences, so read the report before admission.

For inspection without extraction, use an explicit path and disable regeneration:

```python
from merlin.targetgen.rtl.facts import load_facts, validate_facts

facts = load_facts(target, explicit=selected_facts_path, regenerate=False)
problems = validate_facts(facts, target=target)
```

The first call reads a file; the second checks its family/schema shape.
`facts_body` rejects a body that grounded nothing. `decode_body` also checks for
the decoder data its caller needs. Keep recorded unknowns visible rather than
turning them into false or zero values. The notebook demonstrates this with an
empty, schema-valid packet.

Be careful when choosing inspection APIs: `load_facts` defaults to regeneration,
`manifest_for` can regenerate a cold RTL-backed selection, and `derive_manifest`
may consult discovery helpers even when it receives a facts argument.

## Follow the facts to the compiler and CCA

`experiments.phase0.evidence.select_evidence` joins the software spec, contract,
facts and source identities. `spec_fact_drift.fact_capabilities` builds the semantic
capability view. `resolve_spec` fills selected `hardware:` declarations and
`eligible_operations: from_facts`; unavailable forms become unknown placement.
The resolved view keeps the authored source identity alongside it.

The shared scheduling library then consumes explicit resource and effect
contracts. `schedule.derive.geometry_from_address_space` and `geometry_from_machine`
project known array/store geometry or refuse a missing/incompatible quantity.
A pass can use that geometry for tiling; command effects and ordering constraints
are still needed for retention, transfer merging and synchronization. The backend
supplies its encoders. See the [tutorial catalog](../../examples/tutorials/README.md)
for runnable pass examples.

CCA comes from an observation of a program, instruction stream or schedule, not
from a direct `software_spec + facts -> CCA` constructor. For RoCC,
`analysis.rocc_commands.funct_table_for` reads the selected fact-derived decode
table. `cca.lift_asm_roles` consumes decoded roles, an endpoint and explicit
observations. The vector `lift_asm` instead reads its decoded vector stream.
The caller binds those observations to the selected program and target.

Keep these checks separate: source consistency, software admission, implemented
lowering, output correctness, actual accelerator use and measured time. A valid
file or matching carrier width cannot stand in for them. The [CCA guide](cca.md)
continues from observations to comparison and a proposed compiler edit.
