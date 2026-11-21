# Target inputs and dependencies

These targets provide declarations or retained support without a catalog-selected
compiler-generation workflow:

| Target | Contents |
| --- | --- |
| [Toy NPU](toy_npu/target) | Small authored target semantics, ISA descriptions and interface programs. |
| [Muon](muon) | Target declarations and a source-pinned support provider. |
| [Saturn](saturn) | Target declarations and source-pinned vector/matrix support. |
| [Voyager](voyager_accel/target) | Retained capability residual and target notes. |
| [RVV](rvv) | Source-pinned support metadata for the external RVV implementation. |

A target's `support/` and `SOURCE.yaml` remain together. The source record pins
provider bytes and companion provenance; edit the actual OOT implementation and
record a new provider selection rather than silently changing a frozen snapshot.
Support implementations are withheld from candidate compiler workspaces.

Use [compiler generation](../compiler-generation/README.md) for the targets with
configured authoring workflows, or [compilation](../compilation/README.md) to use
an existing compiler.
