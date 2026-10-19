---
title: Generic assembly read views
kind: design
status: current
owner: core
last_verified: 2026-10-18
code_refs: [src/merlin/frontends/linalg_mlir.py, src/merlin/llvmlower/generic_form.py, merlin/tests/ir/test_custom_form_parse_fallback.py]
---

# Generic assembly read views

xDSL implements only part of MLIR's custom assembly syntax. A native generic
printer view lets readers inspect the same module without teaching every
consumer another custom parser. Generic printing is selected without compiler
rewrite passes; it preserves the original captured file.

`parse_mlir_file` first tries the original text. On a parse failure it uses the
existing native generic printer in temporary scratch, respecting `TMPDIR`.
It writes nothing beside the captured source. An explicit `work` directory
retains the printer output there. The compatibility reader delegates this
same choice instead of retrying in the source directory.

An explicit `create_generic_read_view` request retains a separate view and
receipt in a fresh workspace. The receipt records source and view paths, sizes
and hashes, plus the selected interpreter and printer-script identities.
`recheck_generic_read_view` reopens those artifacts and the original printer
selection before the view is consumed. Source changes during printing refuse.
The original captured model remains the executable input; a caller can use the
view for text-only source inventories and bind its policy to the view hash.
Callers retain both identities so printing cannot masquerade as a new capture.

These checks establish artifact custody for the selected printing operation.
They do not establish dependency closure, arithmetic equivalence of compiler
rewrites, numerical quality or runtime behavior. Ordinary source-policy,
compiled-object, final-image and execution gates remain separate obligations.
