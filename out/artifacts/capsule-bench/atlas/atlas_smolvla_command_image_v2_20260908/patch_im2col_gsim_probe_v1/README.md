# Patch-im2col GSIM cap probe

This directory records the bounded GSIM attempt for the exact SmolVLA
patch-embedding contraction `atlas_p0000`. The source boundary is image
`[1,3,512,512]` plus kernel `[768,3,16,16]`; host preprocessing materializes
`A0=[768,768]` and `W=[768,1024]`, and the saved Atlas command computes
`Y0=[768,1024]` before host NCHW reshape and f32 bias addition.

The normal harness did not halt within its explicit 20,000,000-cycle cap and
raised `ProgramDidNotHalt`. The assertion-enabled engine was then run directly
with `assertion_probe_spec.json` and exited successfully after the spec's
1,000,000-cycle cap with `halted=false`, `halt_reason=-1`, 206,822 reads, 256
writes, and 207,080 wrap hits. There was no assertion diagnostic. This is an
assertion-clean bounded-progress observation, not a completed numeric result.
It does not qualify the kernel variant, the physical capture partition, or
SmolVLA end to end.

The hashes and exact limits are in `receipt.json`. The large preload spec and
generated build products are intentionally ignored; the tracked static command
contract and tests are the durable source/ABI proof.
