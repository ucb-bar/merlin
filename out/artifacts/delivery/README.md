# Delivery packages

Zipped image sets handed to a board owner for bring-up. Each package carries **two builds of every
image**: the plain one to run for a number, and a `_debug_` twin of the same model that explains
itself while it runs — so one hand-off answers both questions and the recipient never has to come
back for the other half.

Build one with `build_tools/scripts/make_delivery.py`. Everything that varies per board — the models,
hart counts, console and baud, reset clock, vector length, and the linker's DRAM base — is read from
the board descriptor (`merlin.runtime.boards`), not written per recipient.

Packages themselves are generated output and are gitignored; this file and `AGENT.md` are the only
tracked entries. A round's recipients and per-chip bring-up notes travel with the hand-off, not in
this repo — it is public, and the boards are frequently unreleased silicon.
