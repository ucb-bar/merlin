---
title: Projected entry ABIs in multi-program sessions
kind: design
status: current
owner: core
last_verified: 2026-10-18
---

# Projected entry ABIs in multi-program sessions

`llvmlower.session_projection` carries explicitly selected prepared entry
projections into generated session callers. It supports the immutable-parameter
removal in `entry_weight_projection`; it does not infer a smaller argument
table from LLVM, object symbols, or a saved JSON remap.

`read_program_entry_projection` rederives the captured/prepared argument table,
rechecks all source and optional-file pins, recomputes every removal from typed
prepared SSA, and checks the exact selected module and receipt. New receipts
also record their original source filename and file digest. Legacy receipts
require the caller to supply their checked original source explicitly.

`rename_projected_entry` first rechecks the original complete proof, then
renames the entry and its verified direct callers. Address/symbol escapes,
recursive or unknown callers, invalid names, and symbol collisions refuse.
It replans the projection under the new symbol and requires unchanged removal,
arithmetic DCE, and caller counts. The original module is never mutated.
The target linker separately owns the actual compiled-symbol rename and
compiled-object correspondence; source renaming supplies no binary proof.

`session_bundle.generate(..., entry_projections=...)` accepts typed per-program
selections. It uses each renamed proof in ordinary C generation, preserving
the checked compact weight layout and stream/state descriptor remaps. Its
stage adapter keeps the public session input indices in capture coordinates:
logical input access maps to the retained descriptor table, while deleted
immutable parameter slots remain inaccessible. Cross-program bindings and
output ordinals retain their captured meanings. The saved renamed MLIR has
the projected ABI, matching the generated caller rather than the old capture.

`materialize_shared_weights` requires the same explicit selections. It reopens
every projected program proof before sharing storage, compares each manifest's
map/table identity, regenerates the selected weight producer, and verifies the
actual packed bytes. Different nonempty layouts still refuse sharing. No
projection or prepared-source selection remains the existing session route.

All generated callers must be owned and regenerated, immutable weight bytes
must remain readonly, and weight addresses must be unobserved. Inputs, mutable
state and outputs cannot be removed. This mechanism changes neither numerical
tolerances nor accelerator placement. Timing, physical execution, binary
admission and profitability remain separate caller obligations.

Tests in `merlin/tests/runtime/test_session_projection.py` cover exact source
and receipt rechecking, namespaced direct calls, collision refusal, remapped
state/stream access, shared compact layouts, and mutation of a nonselected
weight-producing stage. Existing entry projection and session tests continue
to cover ownership, state contracts, streams and unprojected behavior.
