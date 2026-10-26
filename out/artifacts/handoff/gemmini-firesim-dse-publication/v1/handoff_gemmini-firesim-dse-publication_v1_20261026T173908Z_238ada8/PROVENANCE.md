# Publication provenance

This product publishes the prepared inputs from
`gemmini-firesim-dse-round2/v1/handoff_gemmini-firesim-dse-round2_v1_20261026T103920Z_ee4031d`.
Its original archive SHA256 is
`0f4abd85fae9fa3528c151ad4f73cd54a940e62b6bdbad4e92ec21b78ef6b95a`.
Splitting that archive preserves every original byte. The historical preparation
reports, source patches, qualification fields and paths are not rewritten.

The numerical-source implementation and FireSim supervisor are committed in
`8e424a270` and `238ada8d3`, respectively. The latter is the
`source_code_commit` in the transfer manifest. The publication commit adds
this product and updates entry-point documentation.

The full ResNet ELF is copied byte-for-byte from the original baseline guest.
Its independent fixture checks every one of 1,000 outputs; retained Spike
receipts establish positive success and wrong-oracle failure. These receipts
are functional evidence, not FireSim timing or fixed-set accuracy.

The companion TVM source remains at
`3423bd6818ab752ddd9cdb5d5773743e48c65f1d`. The archive records the actual
native-library hash and its incremental build receipt. Its configured Git hash
is older metadata; native source/build identities are recorded separately.
No new TVM source edit accompanies this publication.

New TinyLlama calibrated exports, complete quality evaluation and actual
FireSim live-observer qualification remain pending. Publication does not
convert those statuses into passed checks.
