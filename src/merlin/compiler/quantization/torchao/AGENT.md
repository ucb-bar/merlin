# AGENT.md — src/merlin/compiler/quantization/torchao

These files own the existing recipe application, configurations, annotations,
graph rewrites, effects and calibration. Import framework packages lazily.
Generic numerical records remain outside this framework-specific package.
The direct-file capture entry uses the same selected source definitions through
the closed portable loader, without requiring installed Merlin or xDSL.
Preserve arithmetic, provenance, original comparison gates and exact source pins.

`recipes.py` projects saved admitted candidates into prospective capture recipes.
Observer/calibration defaults are framework adapter policy, not hardware facts.
Import its application policy only when projecting; unknown numerical extensions
remain non-executable. Keep this owner outside the portable application helper
roster: an isolated capture consumes its already-selected JSON recipe.
