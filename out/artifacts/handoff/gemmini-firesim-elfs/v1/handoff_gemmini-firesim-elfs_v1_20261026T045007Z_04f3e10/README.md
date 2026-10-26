# Committed FireSim ELFs

Run these binaries with the existing DDR3 FireSim bitstream and ELF launcher.
No TVM checkout, compiler, Python environment or memory-model setup is required
to execute them. Check SHA256SUMS from this directory before transfer/execution.

- ELFS/resnet-baseline.elf: full unprofiled ResNet with default schedules.
- ELFS/resnet-tuned.elf: full unprofiled ResNet with the current validated winner.
- ELFS/resnet-tuned-profiled.elf: optional operator/stage profiling.
- ELFS/tinyllama-decode-kv.elf: M=1, K=2048, N=256 K/V projection pair.
- ELFS/tinyllama-prefill-kv.elf: M=32, K=2048, N=256 K/V projection pair.
- ELFS/tinyllama-decode-down-panel.elf: M=1, full K=5632, N=128 output panel.

TinyLlama chunks include quantization/scaling and all-output checks. They are
trained-weight W8A8 probes, not full TinyLlama E2E. Full TinyLlama baseline and
tuned E2E ELFs must be built/reused from actual remote calibrated exports.

Order: TinyLlama baseline plus ResNet baseline/current tuned, then TinyLlama
chunk ML DSE, then tuned TinyLlama E2E. Run independent jobs on available slots.
For DSE abort when active measured cycles reach the current best, excluding
startup/loading/initialization/warmup. Do not infer runtime or F2 specifications.

All six binaries match the previously Spike-checked payloads. No new FireSim
performance result is claimed. Compiler fork details concern generating new
candidates, not execution of these ELFs. See the source BRANCH_HANDOFF.md guide.
