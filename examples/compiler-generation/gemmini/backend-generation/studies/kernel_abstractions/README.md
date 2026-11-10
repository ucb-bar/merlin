# Compiler abstraction studies

These studies test whether kernel analysis, reusable transformations and checked
scheduling help a coding agent improve an existing accelerator compiler. A
construction study measures effort to implement a change. A frozen-compiler
study checks whether the resulting rule applies to additional input shapes.
Those are separate outcomes, even when two authors produce identical code.

## Source entry points

| Task | Source |
| --- | --- |
| Describe the experimental factors and held-out counts | [`protocol.json`](protocol.json) |
| Prepare the selected seed, public inputs, tools and author workspace | [`prepare_runs.py`](prepare_runs.py) |
| Run an explicitly prepared author with its selected budget | [`run.py`](run.py) |
| Inspect or evaluate frozen compiler variants | [`evaluate_panels.py`](evaluate_panels.py), [`transfer_panel.py`](transfer_panel.py) |
| Build and collect a separately selected physical comparison | [`physical_panel.py`](physical_panel.py) |
| Read-only collection and descriptive plotting | [`paper_analysis/`](paper_analysis/README.md) |

`experiments/cca.md`, `experiments/library.md` and `experiments/scheduling.md`
contain task instructions. Supplying a file is an experimental treatment and
must be recorded in the author-input inventory. Selecting a configuration or
running a preparation command does not establish that an author or device ran.
The runner and physical-panel commands require explicit deployment selections.

## Factors and measurements

CCA studies vary access to decoded expert-kernel descriptions and applicability
queries. Library studies vary access to reusable transformation modules while
keeping the oracle implementation withheld. Scheduling studies compare existing
compiler support with a checked schedule-construction interface. Preserve the
specific seed, exposed source files, prompts, model, reasoning effort, CPU
allocation, helpers, budgets and reference access for each arm. Some historical
studies change several of these together. They cannot isolate a single factor.

Measure numerical correctness, actual emitted changes, elapsed author time and
observed model-token usage separately. A modeled cycle estimate is not a FireSim
cycle count. A static command count does not establish dynamic traffic or engine
concurrency. Freeze the compiler before evaluating the held-out panel and keep
unchanged fallbacks, refusals and failures in the denominator. Never pool results
from different seeds, numerical contracts or shape selections without declaring
those differences.

The dense projection panel varies M, K and N one axis at a time around
16 by 2048 by 2048. The convolution panel varies batch, spatial extent and
channel dimensions. QK and PV attention use different output numerical contracts.
The exact public/held-out split belongs to each study's recorded configuration.
Do not infer that an input was previously unseen solely because a new study
labels it held out.

## Historical runs

Original configurations, delivered prompts, source-input inventories, retirement
records and dated analysis reports are in the checksum-verified
historical study archive retained under ignored `out/` with its original source revision.
They retain original run identities and original machine paths. They are
scientific evidence, not portable presets. The archive includes superseded and
invalidated pilots so their exclusion remains auditable.

For new experiments, install the optional experiment package and select inputs
through the [artifact evaluation guide](../../../../../../docs/guides/12-experiments.md).
For a reported result, first select its exact frozen compiler and measurement
records through the compiler repository's
[run index](https://github.com/ucb-bar/gemmini-mlir/tree/paper/index).
Reusing the current library instead of the recorded one changes the experiment.
