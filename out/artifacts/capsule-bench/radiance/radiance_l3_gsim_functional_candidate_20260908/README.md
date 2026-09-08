# Radiance L3 GSIM functional candidate

This package records the first bounded Radiance L3 GSIM run with a **positive
hardware completion witness**.  It is a functional simulator candidate, not a
kernel-correctness or performance certification.

## What changed

- Successful FIRRTL `stop(..., 0)` sites now print `GSIM model finished
  execution.` immediately before `exit(0)`.
- Non-zero stops retain their failure exit and never print a success witness.
- High-volume hardware `printf` traffic is suppressed by default, while
  assertions, timeout notices, cycle lines, and completion notices remain.
  `GSIM_HW_PRINTF=1` restores all diagnostic prints.
- `make stop-witness-check printf-filter-check` passes in the source tree.
- Merlin's adapter accepts the new positive marker, but still rejects a
  non-zero emulator exit even if a stale marker is present.

The exact modified GSIM sources and the three generated/harness files that
required target-specific intervention are retained under `source/`.  The full
1.93 GB FIRRTL input and 225 generated C++ translation units are not duplicated;
their source paths and hashes are in `receipt.json`.

## Bounded smoke result

The packaged emulator ran the known `R0_gemm_fp32` SoC capsule and exited zero
after 82.22 seconds with the positive completion marker.  It executed meaningful
hardware activity (`317` DRAM reads, `1,268` read beats, and `19,786` CVFPU
accepts), but emitted no DRAM writes, result-page writes, or UART characters.
The capsule currently self-verifies through the SoC completion path, so this is
enough to qualify GSIM observability; it is **not** independent numerical proof.

## Remaining qualification

Before calling Radiance L3 kernel-ready, run a capsule whose result page is
independently checked and require non-zero output traffic plus a golden match.
Before calling this emulator reproducible, regenerate it from a committed GSIM
revision containing the retained source changes; the source checkout used here
was based on commit `65a1f89af195c62b365cd9a827fa4f5f2ca71d1f` with a recorded dirty diff.

Run `python verify.py` from this directory to validate the retained payload.
