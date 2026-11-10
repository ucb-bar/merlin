# Optional kernel-mining inputs

These are **user-authored policy files**, not Phase 0 RTL facts or generated
capsules. Select them explicitly when mining external kernels:

- [feature-extraction.yaml](feature-extraction.yaml) adds Gemmini vocabulary to
  Merlin's generic kernel indexer.
- [autocomp-framework.yaml](autocomp-framework.yaml) records caller-side layout
  and calling-convention assumptions for mined Autocomp kernels.

The indexer generates its index and receipts in the selected output directory.
Neither file claims that a mined kernel is correct or that a compiler lowers it.
See the [kernel-mining guide](../../../docs/guides/09-verification.md).
