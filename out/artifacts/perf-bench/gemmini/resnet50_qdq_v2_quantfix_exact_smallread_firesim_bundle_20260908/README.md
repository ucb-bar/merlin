# Merlin ResNet-50 exact-small-read bundle

> **Hardware qualification failed.** FireSim queue job 533 executed the exact
> staged ELF on the intended Universal U250 image, but 999/1,000 logits differed
> (top-1 749 rather than 258). Do not use the 3,633,774,166 measured cycles as a
> performance result and do not submit the warm arm. The failure receipt is
> `validation/firesim_cold_job_533_failed/receipt.json`.

This bundle contains one compiler-generated TorchAO PT2E W8A8 ResNet-50 kernel for
Jack's `FireSimUniversalResNet50GemminiRocketConfig`.  The hardware has no full-width
accumulator read.  The compiler reconstructs exact i32 contraction results using four
legal narrow radix-128 reads; `compiler/accumulator_readout_audit.json` independently
rejects any forbidden full-width read.

Two harnesses share the exact same kernel object and input:

- `cold`: 0 warmups, 1 measured invocation (official comparison protocol).
- `warm`: 10 warmups, 5 measured invocations (steady-state protocol).

Both validate all 1,000 logits against the independent PT2E integer reference and
require top-1 class 258.  Build with `./build_elf.sh`, then qualify with
`./run_spike.sh cold` before submitting either ELF to FireSim.

The postmortem found that Spike was not a sufficient functional oracle for the
four-pass readout. Jack's RTL sets `hardcode_d_to_garbage_addr=true`, while the
radix reconstruction loads each prior digit's correction through D. The frozen
target YAML contained that fact, but the generated target-profile loader ignored
it. The recovered compiler now reads it and refuses this unsupported lowering.
