# Muon reference metadata

Muon runtime, code generation, introspection and Cyclotron oracle support now
have one owner: the provider vendored at [`../support`](../support), a byte-identical
copy of the local `muon-support` companion commit recorded in [`../SOURCE.yaml`](../SOURCE.yaml)
and [`target_support.json`](../../../../build_tools/dependencies/target_support.json). That
companion had no remote, so this tracked copy is its only published home.

With `MERLIN_TARGET_PATH` unset it is the selected support; an explicit value replaces
it. The provider identity
is `muon`, not the experiment identity `radiance`; preserve that distinction.
This reference contract has no executable plugin declarations.

The [Radiance descriptor](../../../compiler-generation/radiance/target/descriptor.yaml)
still declares the legacy metadata/pin location here. Neither this tree nor the
new provider supplies qualified RTL facts or IRDL pins. Their provisioning and
descriptor migration remain unfinished; moving source must not fabricate facts,
change hardware identity, or silently reuse a different provider's evidence.

The OOT provider includes the native carrier C resource beside its backend and
reuses shared Merlin machinery. Keep its entire root host-private during compiler
evaluation. Pure import, source-identity and mask tests do not qualify native
tools, simulator execution, compiler candidates or historical frozen runs.
