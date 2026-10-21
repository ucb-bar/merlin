# Prospective bounded affine source copies

`checked_native_integer_affine_copy_v2` explicitly selects the source arithmetic
proof extension `merlin.connected_integer_affine_copy.v2`. Public source recipes
request it with `merlin.phase0.public_program_selection.v3` and the
`source_integer_proof` field. The existing V1 source proof and V2 public recipe
retain their original projected-permutation guard and refusals.

The extension accepts a single-input, single-output, all-parallel generic whose
entire scalar body yields the unchanged input cell. The output map must be the
complete identity. Every input index is a nonnegative constant linear expression
of output-bounded loop variables, and its full interval must fit its typed source
extent. Initializer reads, reductions, dynamic extents, symbols, division,
remainder, negative coefficients and additional scalar effects refuse.

This proves gather geometry and propagates the exact source interval. It does
not alter recurrence arithmetic, captured inputs, numerical policy, complete
outputs or target realization obligations. Each issued writer qualification and
each later checked execution recomputes its selected source proof. A saved
qualification is not reused as an execution verdict.
