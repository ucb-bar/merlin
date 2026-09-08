# Radiance L3 GSIM: bounded RP10 numerical readback

Status: **PASS for this instrumented 32-element capsule; not a full L3-suite certification.**

This artifact closes the ambiguity in the earlier completion-only `R0` smoke.
It runs the existing `RP10_gemv_batched_fp16_pt` fork-free kernel on the rebuilt
Radiance GSIM emulator, places its 32 output words on a fixed Muon result page,
and makes the rv64 Rocket carrier read and compare every word against the
capsule's checked-in golden under the capsule's declared policy:

- comparison: `tolerance_float`
- `atol = 0.03125`
- `rtol = 0.015625`
- result: `32/32` within bounds

The carrier contains distinct retained symbols for the two outcomes.  The final
and steady Rocket PC is `0x80000086`, exactly `rp10_numeric_pass`; the fail symbol
is `0x800000c6` and is never observed.  The run reaches the deliberate
120,000-cycle observation bound in 40.76 s simulator time (40.88 s process wall)
with 81,408 KiB maximum RSS.  This bound is an observation window, not a kernel
cycle measurement: the Muon manager is intentionally parked so GPU-idle cannot
terminate GSIM before Rocket records the comparison.

The readback also causes non-vacuous output traffic: `dram_aw=1`, `dram_w=4`,
and the write-back address is `0x110010060`, inside the chosen result page.

## Why the previous R0 smoke reported zero result-page writes

The stock capsule harness declares `_out_Y0` as an automatic array, so it lives
on the Muon stack (the observed physical stack range is around `0x17effffe0`),
not at the instrumentation's hard-coded `0x110001000` page.  In fact local
`0x10001000`, mirrored to `0x110001000`, is the ELF's `.tohost` segment.  The
counter was therefore watching a page no output used.  The ordinary Muon print
aperture is also unmapped from the SoC console, so `uart_chars=0` is expected.
Finally, dirty output lines need not reach DRAM before GPU-idle.  Here Rocket's
coherent read of the explicit result page makes the data observable and causes
the recorded write-back.

## Reproduce

From the repository root:

```sh
.venv/bin/python out/artifacts/capsule-bench/radiance/rp10_gsim_numeric_readback_20260908/build_readback.py
/usr/bin/time -v timeout --signal=TERM --kill-after=10s 180s \
  /scratch/agustin/tmp/gsim-radiance-l3-v6-20260907/emulator \
  out/artifacts/capsule-bench/radiance/rp10_gsim_numeric_readback_20260908/build/rp10.readback.soc.elf \
  +loadmem=out/artifacts/capsule-bench/radiance/rp10_gsim_numeric_readback_20260908/build/rp10.readback.soc.elf \
  +max_core_cycles=0 +max-cycles=120000
```

Run `python verify.py` in this directory to verify hashes and the PASS witness.
The build records the outcome-symbol addresses in `build/readback_symbols.txt`
and strips the fuse helper's randomly named temporary symbols, making the SoC
ELF byte-identical across repeated builds.
The two logs under `diagnostics/` retain the rejected UART and HTIF approaches;
neither is evidence for the final PASS.

## Exact limitation

This proves independent result readback and golden agreement for one small
compiled capsule on these exact emulator bytes.  It does not certify all
Radiance kernels, does not establish a performance number, and does not repair
the production L3 adapter.  The production fix is to allocate a declared result
page, have Rocket grade or export it, and derive the watched range from that
layout instead of hard-coding `0x110001000`.
