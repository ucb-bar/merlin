# shape_choice_table.py: inputs, classes and how to read the table

## Inputs

**`--knobs`**: your `merlin.tuning_knobs.v1` declaration. Every declared value is a candidate; the
candidate space is the product over the knobs you select with `--knob` (all knobs otherwise, at most
256 candidates). The `settings_file` next to it gives the current setting. A result row whose other
knobs differ from the current setting is set aside, so vary one family of knobs at a time.

```json
{"schema": "merlin.tuning_knobs.v1", "settings_file": "tuning_settings.json",
 "knobs": {"block_policy": {"values": ["baseline", "retain-rhs", "wide-panels"], "mechanism": "..."}}}
```

**`--cases`**: the geometry of each case you judge by. One of:

```json
[{"case": "A", "m": 64, "k": 256, "n": 512},
 {"case": "B", "operand_shapes": [[1, 2048], [2048, 2048]]},
 {"case": "C", "conv": {"n": 1, "c": 24, "h": 3, "w": 3, "k": 24, "r": 3, "s": 3, "stride": 1, "pad": 1}},
 {"case": "D", "m": 196, "k": 512, "n": 128, "model": true},
 {"case": "E", "m": 48, "k": 48, "n": 48, "class": "boundary"}]
```

A conv geometry includes its output shape, so stride and padding are part of its identity; the
capacity predicates use its implicit contraction (M = N·OH·OW, K = C·R·S, N = output channels). Cases
with the same geometry are aliases: they merge into the first one (`aliases` in `--json`), because a
`ValidationPartition` refuses one geometry on both sides of a split.

**`--results`**: one of

* normalized rows (write these from any source, e.g. a script that collects `.qa_channel` responses):
  ```json
  {"schema": "merlin.author_shape_results.v1", "rows": [
    {"case": "A", "settings": {"block_policy": "retain-rhs"}, "cycles": 412345,
     "basis": "modeled_roi", "domain": "timing model sha/mode/epoch", "numerics": true,
     "program_sha256": "optional: sha256 of the emitted program"}]}
  ```
  `basis` is `measured_roi` (simulator cycles), `modeled_roi` (timing-model estimate) or
  `static_emitted_ir` (a static count you chose as the objective, e.g. operand bytes from
  `explore-schedules`); `cycles` is that one objective. Missing or non-positive `cycles` is unknown.
  A row may instead carry `"observation"`: an `asdict(CompletedObservation)` from
  `merlin.transforms.evaluated_exploration` (plus optional `implementation_sha256`). Its own rule
  applies (passing numerics, complete timing over the complete ROI, known cycles), and its domain is
  the `PerformanceDomain` fields of its identity, so the rows you give `shape_policy` read the same here.
* self-check responses (`agent_selfcheck.py --attach ID --out FILE`), as a list of paths or documents,
  or `{"responses": [...]}`. A knob-override response carries its own settings. For a response of
  `variants/NAME`, wrap it: `{"response": "ab/NAME.json", "settings": {"block_policy": "wide-panels"}}`.
  Measured certifying-tier cycles are used when present, else the timing-model estimate, each with its
  own identity as the domain.

Only one (basis, domain) is ranked: the one with the most known observations (measured first on a
tie). Rows from other domains are counted under `set aside`; re-run them in the chosen domain rather
than mixing domains.

**`--facts`** (default `rtl_facts/facts.json` when present): the RTL facts. They give the array edge
(classes, `ragged`) and the operand and accumulator stores (capacity predicates). `--tile EDGE` gives
the edge without facts; capacity predicates then stay unknown.

## Classes

Derived from structure, in this order: an explicit `class`; `model-layer` (`"model": true`); `m1` (one
LHS row); `ragged` (an extent that is not a multiple of the array edge); then `long-m`, `long-n` or
`square` by comparing the M and N tile counts. No aspect-ratio number is involved. Give a case an
explicit `class` when your lowering already distinguishes something else (a decode step, a
boundary case you constructed on purpose).

## Ranking and hold-outs

The incumbent is the current setting. A candidate's score on a set of cases is the geometric mean of
its cycles divided by the incumbent's on each case, the score `assess_shape_policy` trains on; a case
without a passing pair (candidate and incumbent) makes the score unknown and the candidate is listed
as unranked, never ranked as zero. `rank_observed_metrics` orders the scores (`cycle_ratio`); the
`cycles` column is the class total for reading, `vs current` the score.

For each class C a `ValidationPartition(training=other classes, held_out=C)` is frozen; the candidate
ranked first on the training classes is looked up on C:

* `holds-out yes`: one choice serves C and the other classes; a single default is supported.
* `holds-out no (other classes pick X)`: C needs its own choice. Find the capacity guard that
  separates it (below) instead of adding a size test.
* `regresses +x% on a case here`: the other classes' winner is slower than the incumbent on one of
  C's cases. `assess_shape_policy` vetoes exactly this (`max_regression_ratio`) and keeps C's incumbent.

## Capacity predicates

Per case, with the derived stores (row-granular, `address_space.working_set_rows`):

| flag | meaning |
|---|---|
| L | the whole LHS (M x K) fits the operand store |
| R | the whole RHS (K x N) fits the operand store |
| O | both operands fit together (the `capacity_fit` working set) |
| A | the whole output (M x N) fits the accumulator |
| S | one M stripe of output (M x one array edge of N) fits the accumulator |
| T | resident tiles that cover the layer (`merlin.compile.capacity._capacity_fit_tile`) |

The flags come from integer facts listed per case under `resources` (`--json`), named for
`PolicyFacts.resources`: `<set>_copies` (whole copies the store holds: 0 does not fit, 2 leaves room to
double-buffer) and `<set>_stores_needed` (2 or more overflows), for the sets `operand_lhs`,
`operand_rhs`, `operand_both`, `accumulator_output` and `accumulator_m_stripe`, plus
`resident_tiles_cover`. Compute the same values in your lowering from the same facts.

`Capacity predicates that separate the per-case winners` lists each knob whose winning value is a
function of one flag over every case with a unique winner, e.g.
`block_policy: wide-panels where output_fits (2 cases), retain-rhs where not (5 cases)`, followed by
its two guards, `ResourcePredicate('accumulator_output_copies', 1)` and
`ResourcePredicate('accumulator_output_stores_needed', 2)`, each with the boundary it still needs:
`assess_shape_policy` keeps the incumbent until the panel holds the guard's value and the adjacent
outside value (1 and 0 copies, 2 and 1 stores). That is your cutoff, as `shape_policy` consumes it
(`reference/shape-policy.md`). A cutoff copied from the
measured extents (the first M where the winner changed) moves with the cases you happened to measure;
the predicate moves only with the device. The case counts per side are the support: a predicate seen
on one or two cases per side may be coincidence (several predicates can agree on few cases), so add
cases just inside and just outside it and re-run before relying on it.

When nothing separates: the classes may need a fact the script does not compute (bank count,
double-buffered windows, a reduction depth). Derive it from the facts the same way, add cases on both
sides, or keep the per-class choice as a knob value selected by class.

## From explore_candidates to rows

`explore_candidates(module, space, facts, output=..., emission=..., feedback=...)` yields each real
compiled candidate once per distinct emitted program. Its `feedback` callback receives the emitted
program and may return your observation joined to `program_sha256`; write one row per (case,
candidate) with that sha. Identical programs for two settings on a class mean the knob is inert there
(`distinct_programs` in `--json`), which is evidence too: no timing is needed to choose between them.

## Reading a shape_policy report

`shape_choice_table.py --shape-policy policy.json` reads `assess_shape_policy(...).report`
(`merlin.shape_schedule_policy.v1`) and prints, per class, its status
(`empirically_supported_declared_panel` or `incumbent_retained`), the selected and incumbent paths,
and per alternative its training score and what keeps it from being supported: panel geometries
without feedback, a cutoff without its boundary cases, incomplete pairs, mixed performance domains,
or a regression on a matched case. `--json` prints the report unchanged.
