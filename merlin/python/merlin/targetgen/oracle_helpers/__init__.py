"""Helper scripts for the generic program-oracle that run inside a TARGET MODEL's own venv (not merlin's)
— e.g. ``npu_emit.py`` (assembles a target's emitted kernel via the model's OWN ISA/assembler). Kept in
the package so they ship with the wheel; invoked by path as a subprocess, never imported into merlin."""
