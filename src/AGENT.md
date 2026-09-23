# Python source

`merlin/` is the canonical core package. Optional research distributions belong in
`packages/`; experiment definitions belong in `experiments/`. The old
`merlin/python/merlin` path is a migration symlink, never a second implementation.
Preserve sandbox module identities and run the access-registry tests after moves.
