# Declared library-pass integration

This is an opt-in author ABI. Existing arbitrary emitters need an adapter; the
framework does not infer local addresses, instruction semantics, completion or
hardware permissions. No adapter or winning target implementation is supplied.

`tools/library_pass_adapter.py` exports:

```python
def source(case, *, package):
    # Optional locator only; --source OWN_SOURCE.mlir can replace it.
    return package / "OWN_SOURCE.mlir"

def prepare(source, *, package, mechanism):
    # Parse CURRENT source, derive selected facts, and build a fresh session.
    # None selects incumbent; a declared ID selects a shared registered pass path.
    # Return these existing typed objects (not previously emitted text):
    return dict(session=session, facts=facts, registry=registry,
                policy=policy, output="OWN_EMITTED_PRODUCT")
```

Types are `PipelineSession` from `merlin.transforms.pipeline_state`, and
`PolicyFacts`, `RegisteredPass`, `PipelinePolicy`, `PipelineVariant` from
`merlin.transforms.pipeline_policy`. Both paths must have identical initial
IR/context stamps, facts and registered pass contracts. The last registered
pass must be the same actual target emitter and return nonempty text/bytes in
the named current-session analysis. Independent pass verification still applies.
Registered actions and mechanism mappings remain authored: declared contracts
and an ID do not independently prove use of a catalogue implementation.
Do not treat a performance predicate as a legality permission.

For compact counted IR, supply its fingerprint, clone and verification adapters
to the existing session. Use the existing checked scheduling result's
`emit_index_program(control, lower_effect, complete_epoch=...)` bridge: retain
your original effect encoder and real completion callback. Keep counted loops;
do not flatten a large program to integrate the pass. See `heavy-pass-recipes`
for the shared APIs and executable emitter examples. A mechanism catalogue ID
alone cannot create a source/effect mapping.

Your actual declared compiler entrypoint must read the variant's
`LIBRARY_PASS_SELECTION.json`, then use the SAME adapter and registered emitter:

```python
selection = package / "LIBRARY_PASS_SELECTION.json"
mechanism = json.loads(selection.read_text())["mechanism"] if selection.exists() else None
binding = prepare(source, package=package, mechanism=mechanism)
product = execute_policy(**binding).artifact
# The declared stdout-producing emit command writes this actual product.
```

The file has schema `merlin.author_library_pass_selection.v1`, the relative
adapter path, and the requested mechanism. The CLI writes it only in a new
variant. The baseline stays unchanged. Use source-derived applicability and
explicit UNKNOWN fallback; `--case` is only a source locator and provenance.

## Running and interpreting application

The default `--command lower_target_to_llvm` reads its `argv` from manifest.yaml.
Use `--command OWN_DECLARED_STDOUT_COMMAND` for another declared text/bytes
emitter. Arguments support `{input_mlir}`; no shell is used. `--timeout` bounds
each actual emit process (default120s, maximum900s). The tool runs each side's
declared emission once, avoiding full compiler builds or self-check runs here.
Selected compiler-library source is pinned, reused by link, and verified
unchanged; build/output/evidence caches are not copied. Other safe source links
are materialized as regular files. Links outside the source or selected library
are refused.

`APPLIED_EMITTED_CHANGE` requires actual baseline/proposal stdout to match its
respective shared pipeline product byte for byte and the products to differ.
It proves that the pass is connected to actual emission. Numerics remain
`NOT_CHECKED`, timing `UNKNOWN`: then run normal same-source paired self-checks.
An emitter which ignores the selection is `DISCONNECTED_ADAPTER`; identical
products are `UNCHANGED`. Neither may proceed to A/B evaluation. Missing hooks
or mechanism mappings return `CONSTRUCTION_REQUIRED` (exit2); other refusals
return exit1. Receipts retain actual command, executable, source, adapter,
pipeline, output and library identities. Partial/failed variants stay for
diagnosis: use a new variant name for retry; do not grade a failed application.
The CLI creates `variants/.library-pass-pending-NAME` before exposing any
variant files. Idle tuning skips that name during construction and after any
refusal. Only a successful application receipt releases it for discovery;
do not remove the marker to bypass a failed application.
