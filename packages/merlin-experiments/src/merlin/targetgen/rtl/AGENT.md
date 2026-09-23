# Trusted RTL replay generation

`gen_rocc_replay` constructs replay inputs and independent expected outputs for
the existing native harness. It is host-owned grading tooling, not structure-only
RTL inspection. Its source identity must remain withheld in `common.access`.

Core owns the `merlin.targetgen.rtl` initializer. Do not add one here. Keep the
stable module CLI and existing numerical/layout behavior; target generalization
is separate from this ownership move.
