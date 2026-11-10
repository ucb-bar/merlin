`host-capabilities-attention-v2.yaml` is a retained host view; the default
descriptor's `host_lane.capability_spec` selects the reviewed
`host-capabilities.yaml`. This view's prospective successor is
`attention-host-placement-v2-20261019`. The historical `host-capabilities.yaml` and immutable scalar compiler package are
unchanged. The nine added declarations cover closed typed attention preparation
and boolean negation; they do not move supported INT8 contractions onto the host.
Their numerical contracts remain unreviewed, and parsed source, linked-image,
offload and full-output checks remain separate obligations.

Fresh test generation selection automatically calls
`HostLane.resolve_capabilities` in `merlin.development.testgen.evidence.select_evidence`.
That observes the selected declaration and package bytes, and
`evidence_artifacts` exports `software/host-capabilities.json`. A derivation using
`--evidence-input` deliberately retains the old frozen view. To select this
successor, prepare a new evidence/corpus root from the current descriptor; never
replace or relabel an existing release, private specification or sealed host view.

The smallest host-view-only preparation, with the selected immutable host package
already present under `SOURCE_ROOT`, is:

```sh
PYTHONPATH=src:packages/merlin-experiments/src python build_tools/deploy/derive_host_view.py \
  --descriptor examples/compiler-generation/gemmini/target/descriptor.yaml \
  --source-root "$SOURCE_ROOT" --output "$NEW_HOST_VIEW_ROOT"
```

The destination must be new. The producer uses the same descriptor/lane resolver
as test generation and records descriptor, host declaration, compiler package and exported
view hashes. This command derives only the host view; it neither captures models
nor seals a corpus or claims execution qualification. A later corpus/private
specification must pin that new view under its own identity. Existing sealed declarations and numerical selections retain their original
identities; a successor requires a new explicit input selection.
