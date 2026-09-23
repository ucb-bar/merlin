# merlin_assisted_public_v0

Input bundle for the **Merlin-assisted** arm. Same task / capsules / grading as the raw baseline,
plus read access to Merlin's **authoring** tools (`targetgen/synthesize`, `targetgen/generate`,
`xdsl_dialects`, `interface_emit`). The runtime **reference/simulator** and **all prior backends** are
denied so the agent cannot read the answer or copy a finished package; the submitted package must
still pass the non-exempt integrity scan (no `import merlin`, no reference reads, no copied/called
kernels). Authoritative spec: `input_bundle_manifest.yaml`.

**Oracle-callable helpers are denied even inside allowed dirs (deny-wins):**
`targetgen/generate/runtime_adapter.py` and `xdsl_dialects/lowering/` both expose a *callable* route to
the reference/simulator oracle (richer than the redacted QA verdict), so they are excluded — the
launcher stages the allowed tool dirs **minus** these. See `ALLOWED_MERLIN_TOOLS.md` for the exact
allowed/forbidden tool surface, `TASK_ADDENDUM.md` for the merlin-specific agent instructions, and
`MERLIN_PROVENANCE_TEMPLATE.md` for the post-hoc provenance the agent fills.
