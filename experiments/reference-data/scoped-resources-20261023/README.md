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
