# Muon reference metadata

Muon runtime, code generation, introspection and Cyclotron oracle support now
have one owner: the separate local `muon-support` repository recorded in
[`target_support.json`](../../../build_tools/upstreams/target_support.json).
No remote has been selected and no companion changes have been pushed.

Select its root explicitly through `MERLIN_TARGET_PATH`. The provider identity
is `muon`, not the experiment identity `radiance`; preserve that distinction.
This reference contract has no executable plugin declarations.

The [Radiance descriptor](../../../examples/radiance/target/descriptor.yaml)
still declares the legacy metadata/pin location here. Neither this tree nor the
new provider supplies qualified RTL facts or IRDL pins. Their provisioning and
descriptor migration remain unfinished; moving source must not fabricate facts,
change hardware identity, or silently reuse a different provider's evidence.

The OOT provider includes the native carrier C resource beside its backend and
reuses shared Merlin machinery. Keep its entire root host-private during compiler
evaluation. Pure import, source-identity and mask tests do not qualify native
tools, simulator execution, compiler candidates or historical frozen runs.
