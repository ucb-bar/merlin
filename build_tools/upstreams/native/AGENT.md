# Native upstream build support

Retain thin opt-in patches and public source pins for independently selected
native build tools. Keep external checkouts, SDKs, plugin binaries and execution
receipts under the caller's `out/` owner. This directory is not a runtime default,
target provider, compiler baseline or author input grant. Do not vendor external
source trees or add target interface, memory, boot/loading or scheduling code.

Each patch names its exact public revision, affected source hash, rationale and
replay limits. Source and SDK header correspondence is narrower than a reproduced
toolchain build or semantic qualification. Original runtime/numeric controls and
all independent hardware, ELF, source, loading, reset and timer gates remain with
their existing owners. Never infer completion from process exit or numeric rows
when the selected engine exposes a distinct termination request.
