# Public scoped observer acceptance

The selected existing public queue-capacity-validation probe was replayed with
one rebuilt observer, first disabled and then enabled. The original selected
observer estimated **354 cycles**; disabled and enabled each estimate **354**.
Feature counts, functional stdout, selected ELF/input/counter bindings are equal.
The enabled report has a complete scoped capture of 19 named model resources.
No timing coefficients were changed. The model engine source identity was
explicitly rebound to the compiled diagnostic source; diagnostics themselves
change model hashes. This grants no calibration or physical timing qualification.

`source-overlay.json` lists the exact original selected observer source leaves
and two compiled candidate leaves. All five original leaves were checked against
the original selected build receipt. `build.json` retains commands, base archive
hash and candidate binary hash. The unchanged engine headers preserve the loaded
provider ABI. Unselected conditional-MMIO changes from another worktree were not
imported. Spike companion commit: `472bd25`.

`acceptance.json` records the exact two invocations and wall times: 17.91s off,
21.82s on (single sequential samples, not a throughput benchmark). Full reports
and common offline/automatic projections are retained alongside it. The binary,
models, stdout/stderr, projection and replay helper are in the operator artifact
folder `out/artifacts/scoped-resources-20261023` of task
`el4-memory-reference-20261023`. Build commands require their explicitly pinned
base archives; this is an incremental same-machine acceptance, not a portable
binary package. A fresh companion `timing/build.py` build remains the ordinary
source build path.

Focused checks: C++ clipping/overlap/queue/unknown/bounded-buffer cases and Python
scope/source binding, capacity arithmetic, missing metrics, compact publication,
automatic feedback and terminal rendering. Four unrelated broader fixture tests
still omit `selection.facts.sha256` and fail in untouched `joint_performance.py`;
no thresholds or source contracts were relaxed to make those fixtures pass.

This capture includes model controller/storage bookings, not physical busy cycles.
Host preparation/dispatch/wait timing, true ROB occupancy and physical bandwidth
remain unavailable. The public latency objective and calibration authority do not
change, and no experiment or FPGA run was launched.

The actual retained probe has 24 counter windows and explicitly selects window0.
Offline projection uses that exact selected window. The automatic Phase1 wrapper
retains its existing single-window guard and refuses the unmodified multi-window
probe. Its shared projection path is tested separately with a clearly synthetic
single-window transport fixture; that fixture grants no native execution credit.

## Current selected WS runtime follow-up

The exact current `ws-accumulate-operands-v1` model and its unchanged provider
were then tested on the same public probe with original, rebuilt-disabled and
rebuilt-enabled observers. All three predict **354 cycles**, preserve functional
stdout and every observed feature/count/window, and the enabled capture is
complete. `ws-acceptance.json` records actual invocations. No hardware execution
was added. The full Phase1 `_runtime` freeze passes the original RTL facts,
additional source alignment, library order/hashes, ISA, scope and model/source
identity checks; `ws-runtime-freeze.json` is its recorded result.

`ws-selection-delta.json` pins the old and new runtime/model/performance selection
and records the only model changes: `domain.feature_engine_sha256` and optional
`diagnostics.scoped_resources`. Timing parameters, target scheduling settings,
provider, original functional Spike and scope policy remain exactly unchanged.
The new selection is prepared under task-local
`out/artifacts/ws-scoped-runtime-20261023/performance-selection.json`; the operator
may explicitly choose its pinned JSON via `MERLIN_JOINT_PERFORMANCE_SELECTION`.
No other environment change is needed. This adds diagnostics only to existing
optional performance-observer executions, not to functional Spike itself.
It does not launch a campaign, alter an active run, fit coefficients or qualify
physical accuracy. The exact source/build/replay helpers and models remain in
that artifact directory for local reproduction.

## Publication redaction of the WS follow-up

The curated `ws-*.json` records are explicitly non-executable projections.
Absolute machine paths have been replaced by consistent opaque
`<machine-path-NNN>` placeholders, including paths within command arguments and
embedded environment JSON. Numerical values and original identity hash strings
are unchanged. Embedded file pins still identify the original raw inputs; they
do not authenticate the redacted projection or authorize executing it.

`ws-redaction.json` records each original byte hash, sanitized byte hash and
relative generated-artifact location. Exact original receipts are preserved in
`out/artifacts/ws-scoped-runtime-20261023/curated-original`; actual executable
runtime selections remain in their original generated-artifact directory.
The preceding historical baseline files are unchanged. The original 354-cycle
neutrality result is not evidence of calibration accuracy: independent held-out
coverage, every-case error qualification, deployed memory/configuration identity,
and missing physical/host attribution remain separate requirements.
