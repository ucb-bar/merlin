The frozen qualification and exact actual selections are in qualification-v1/PLAN.json and originals/*/selection.json.
Use the pinned Lean executable/library, retained fixed-core selection and source/profile bytes; do not replace a changed standard-object binding.
One fresh-process batch already ran. To independently reproduce later, use a NEW output directory and:
PYTHONPATH=src .venv/bin/python -m merlin.verify.lean_graph_state_scalar_receipt --receipt <receipt.json> --destination <new-output> --reference /home/jeremy/Projects/merlin-graph-proof-intake --original <source.mlir> [--original <middle.mlir>] --original <actual-after.mlir>
Original selection is external authority; saved JSON is not. Std.Time.Format.Basic is not admitted by this profile. Unknown native failure/unconfirmed dependency stops the context without retries.
The owned.patch is additive and is delivered for review; nothing was staged/committed/pushed.
