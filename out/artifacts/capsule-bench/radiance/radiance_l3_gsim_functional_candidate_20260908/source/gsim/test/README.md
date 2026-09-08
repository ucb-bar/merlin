# Test Inputs

- Any `*.fir` file in this directory is auto-discovered by `make fir-tests` and by the GitHub CI `fir-regression` job.
- repro-usefulreset.fir: Minimized FIR reproducer for GSIM issue #106, used to guard against ConstantAnalysis hangs and OOM regressions.
- stop-witness.fir: Guards the positive completion record emitted immediately before a successful FIRRTL `stop`; the enclosing harness cannot report after the generated model calls `exit(0)` from inside `step()`.
- printf-filter.fir: Guards the default suppression of high-volume hardware diagnostic prints while timeout/assertion/completion markers remain visible; set `GSIM_HW_PRINTF=1` to restore every hardware print while debugging.
