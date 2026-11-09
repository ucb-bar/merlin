# Draft target artifact ABI: structural inspection only

Experiment ABI `0.1` and its default four-command protocol remain supported.
Its existing `emit_target_artifact` alias still means the legacy text artifact.
Explicit ABI `0.2` replaces the required fourth command with
`emit_target_artifact`. The common manifest definition is shared with `0.1`;
the new schema inherits it and changes only the version and required commands.

There is **no ABI `0.2` executor or certification adapter**. Normal device builds,
capsule execution and certification must refuse this version before invoking
package commands. They may not interpret its stdout as legacy LLVM text.
`run_entrypoint` and command resolution admit it only with an explicit typed
`SelectedArtifactProfile` for structural emission. Even that permission never
authorizes execution or supplies a numerical, ABI or RTL certificate.

`schemas/artifact_bundle.schema.json` carries file identities, logical whole
buffer identities, artifact/unit/issuer labels and explicit predecessor edges.
`verify_bundle(document, root, profile=SelectedArtifactProfile.capture(path))`
checks the independent current profile identity, contained file identities,
unique IDs, references, initialization and transitive read/write ordering.
The profile and files are rechecked before returning. The module CLI requires
`--profile` and performs the same structural checks.

Artifact kinds, entries, ABIs, unit/issuer capabilities, physical buffer aliasing,
memory-space access and the actual realization of dependency edges remain OOT
provider obligations. A declared full write is not a proved initialization.
Partial writes need a richer ownership contract. The core verifier checks the
declared graph only; it does not launch, link or execute artifacts, and no new
normal runtime path or Phase 1 certificate is available from this draft.

A future executor must explicitly consume this carrier and bind the selected
OOT linker/runtime, actual storage/ABI/capability proofs, synchronization and
complete numerical gate. Adding such an adapter is a separate change.
