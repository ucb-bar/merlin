# Python source

`merlin/`, `merlin_experiments/` and `merlin_dse/` are owned by one Merlin distribution.
Compiler primitives remain independent of optional research workflows and heavy dependencies.
Importing a package or inspecting definitions must not launch agents, simulators or hardware.
Experiment definitions belong in `experiments/`; generated state belongs under `out/`.
The old `merlin/python/merlin` path is a compatibility symlink, never a second implementation.
Preserve logical access roles, source custody, command/import identities and frozen replay inputs.
