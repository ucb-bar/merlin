# Disabled transpose-pack draft

These payloads preserve paused source bytes. They are plain `.txt` files, are
outside the Python package and backend discovery paths, and include no active
SDK addition, candidate registration or executable test module.

The optimization is **untested and disabled**. No tests, compiler, native run,
trace, target job or source/effect qualification ran for these draft bytes.
There is no performance claim. `manifest.json` records each original intended
path, exact byte count and SHA-256 so the complete draft can be recovered.

Known issue: the draft test compares raw MVIN3 source pointers even though the
hoisted pack changes zero-seed allocation order. A later reviewed implementation
must compare independently established zero provenance and output extent rather
than assume pointer identity. This issue was recorded before any test execution;
the stored test payload has deliberately not been fixed or run.

The intended generic component emits a bounded dense i8 transpose. The OOT
component proposes packing an immutable RHS once into private dense K-by-N
storage before the existing contraction loops, with a 64 KiB compiler workspace
policy and exact fallback for unsupported semantics. Candidate preparation was
intended to carry exact shared source bytes as ordinary compiler code, avoiding
an undeclared import from the frozen public SDK. None of these custody,
immutability, alias, resource or output obligations is established by preserving
this draft. Original complete source/native/ELF/packet owners remain mandatory.

Intended composition order is layout preparation, then panel scheduling, then
configuration/effect cleanup. Wider raw transfers are not implemented: selected
hardware/software support does not widen the old packet owner's DIM-column
limit. Do not activate these payloads by copying them into live paths without a
new review, meaningful tests and separately authorized original admission.
