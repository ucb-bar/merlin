# Gemmini platform selections

The accelerator user starts with [`../target/software-spec.yaml`](../target/software-spec.yaml),
using the [software-spec instructions](../target/README.md).
These files select the hardware and host environment. Their owner is the platform
or host-compiler integrator; they do not add mathematical accelerator functionality.

| File | Owner and purpose |
| --- | --- |
| [hardware.yaml](hardware.yaml) | Hardware integrator: source-production policy and assertions checked against the selected RTL. Expected types are audit questions, not facts. |
| [host-capabilities.yaml](host-capabilities.yaml) | Host-compiler owner: exact typed host-operation declarations bound to an immutable compiler package. Review is separate from execution qualification. |
| [board-catalog.yaml](board-catalog.yaml) | Platform integrator: selected board, DTS and host-ISA identities. Unknown loaders or capacities remain unknown. |
| [scalar-host-recipe.yaml](scalar-host-recipe.yaml) | Host integrator: minimal ISA/codegen request for Merlin's generic scalar package minter. Minting does not qualify operation support. |

[`../target/descriptor.yaml`](../target/descriptor.yaml) explicitly selects the board
catalog and host capability manifest. [`../experiment.yaml`](../experiment.yaml)
selects the hardware policy for the retained diagnostic Phase 0 example. Path moves
must update these selectors; there are no compatibility symlinks.

The tooling derives RTL facts, source audits, ISA bindings and runtime/build
receipts from the selected sources. An audit writes `validation.json` beside the
selected `facts.json`, under `out/`. It is not a user input to the software spec.
Execution routes and public ISA sources still need explicit selections. No
compiler-bearing support package is supplied by these data files.

See the [Phase 0 instructions](../phase0/README.md) and
[operator verification commands](../verification/README.md) for generation and
qualification. Keep all generated products outside this directory.
