
## Starting from a seed compiler

This workspace starts from an existing compiler, the seed. Treat it as a baseline to beat, not a design to preserve: replace any part of it whenever a library route or a new construct is faster. gSIM confirmation is optional and not part of the performance loop; performance is measured on the Spike cycle-model estimates, and nothing here waits for gSIM.

Your first task is to read what the seed run left, all at the workspace root and outside `submission/`:

* `SEED_EVIDENCE/`, when present: the seed's baseline performance map and its structure diff. Rank the seed's slowest parts on the large model layers and connected programs from it and start replacing them.
* `OPTIMIZATION_RULES.json`: the seed run's rules with their evidence. Keep maintaining it.
* `SEED_EXPERIMENTS/`: the seed run's A/B records and variant outcomes, including rejected experiments. Do not repeat a rejected experiment unless you have a new reason, and say what the reason is.
* `PREVIOUS_SESSION.md`: where the seed session left off.
* `SEED_NOTES/`: the seed's own reports and iteration notes. They describe an earlier harness and may be stale. They are kept for reference only.

From minute one, give each helper one operation family with a performance objective: the family's slowest large cases first, compared by cycles against the seed's numbers. Integrate each win as soon as it keeps numerics.

The seed's correctness is already established. A byte-identical executable keeps its earlier verdict and is not certified again, so a grade re-checks numerics only where your edits change the emitted program. Spend your time making cases faster, not re-proving cases that already pass. Check the cases your change affects (`--capsules` with their names or `--panel FAMILY`) instead of sweeping everything.
