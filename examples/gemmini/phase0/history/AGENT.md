# Historical software declaration

`software-spec-seals1-3.yaml` retains the exact pre-seal-4 declaration for
historical builders. Select it explicitly only when replaying those semantics.
The current descriptor excludes this directory from agent inputs through
`answer_surfaces.held_out_inputs`. New releases select the active target spec
and generated-only corpus preparation; never republish this historical leaf.
