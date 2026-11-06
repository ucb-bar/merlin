# Kodiak aligned-weight int8 schedule

This package copies the schedule and knobs from `impr_tuned_wholemodel_vf_int8` and adds only
`prepack_weight_layout`. The parent package's K1 certification does not transfer to this variant.

The two-layer TinyLlama 64-token FireSim pair using this schedule passed the exact Spike gates
and the pinned one-core OPU bitstream: device job 3866 took 1,101,993,218 target compute cycles,
RVV control job 3867 took 13,689,726,593, and both full 2,048,000-value hashes were 2628075425.
The portable result and utilization receipt is `firesim_evidence.json`; source receipt SHA-256
values are in both that file and `manifest.yaml`. That pair used
Merlin before the later generated-kernel fill-elision optimization; test any new source head as a
new image. No K1 or physical Kodiak measurement is claimed.

Select this package with `./run.sh firesim --models tinyllama_l2_s64 --rvv-package
rvv/kodiak_prepack_w8a8_v1`. For a bounded board delivery, use `./run.sh package --opu --models
tinyllama_l2_s64:<bundle-dir> --harts 1 --rvv-package rvv/kodiak_prepack_w8a8_v1
--no-spike-backends matrix`. That delivery marks its exact chipyard OPU ELF ungated; the separate
FireSim result does not grade those bytes.
