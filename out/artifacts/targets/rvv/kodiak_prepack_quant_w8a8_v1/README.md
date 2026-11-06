# Kodiak aligned-prepack int8 schedule with exact quantization fusion

This opt-in RVV package copies `kodiak_prepack_w8a8_v1` and adds only
`fuse_quantize_round_convert`. It removes the 23 scalar `roundevenf` quantization call sites
observed in the two-layer TinyLlama 64-token image. The broader
`fuse_quantize_round_convert_vec` schedule remains off. The parent package and the default
K1-certified schedule retain their original identities.

The byte-identical schedule and knobs passed an independent OPU/RVV Spike gate and a one-core
FireSim pair for that TinyLlama section. The OPU image ran in 1,022,361,106 target compute cycles,
versus 1,064,676,843 for the matched fill-only prepack image (3.9745% fewer). The plain-RVV
control ran in 13,646,684,998 cycles, giving a 13.3482× paired speedup. Both images matched their
exact Spike full 2,048,000-element output hashes and prefixes; their 4,096-element hardware W8A8
prefix gates passed. Full W8A8 numerical comparison of the FireSim output was not available from
the capped console. The source SHA values, jobs, bitstream digest, and utilization are in
`firesim_evidence.json`.

The OPU device routed all 19 contractions and 9,865,003,008 valid MACs, with 100% static tile
fill. It achieved 9.649235 valid MACs per target compute cycle, or **0.942308% of the
1,024-MAC/cycle theoretical peak across the whole inference**. This is not physical array
occupancy: the bitstream has no active-MAC counter. The archived bitstream digest matched, while
the live Saturn checkout was off its source pin, so clean bitstream reproduction is unverified.

Build a new exact pair with:

```sh
./run.sh firesim --models tinyllama_l2_s64 --rvv-package rvv/kodiak_prepack_quant_w8a8_v1
```

The package manifest has a distinct identity from the measured s64 trial artifact. Its rebuilt
s64 OPU and independent plain-RVV images have therefore passed their own exact Spike full-hash
gate; the rebuilt s64 images have not run on FireSim. See `model_gates.json` for all immutable
plan, capture, ELF, output, and routing hashes.

The rebuilt s128 OPU and plain-RVV Spike images agree on the full 4,096,000-value output hash.
The exact OPU ELF additionally passes a complete W8A8 comparison of all logits, plus all 128
per-token argmaxes, using a debugger dump taken after computation and before the capped UART
output. Its matched FireSim pair is queued as jobs 3891/3892. The rebuilt Spectformer OPU and
plain-RVV Spike images agree on the full 1,000-value output hash; the OPU image passes complete
W8A8 numerical and argmax gates. Its matched FireSim pair 3889/3890 is **verified**: 2,578,301,427
OPU versus 5,667,528,436 plain-RVV target compute cycles, or 2.1982× speedup. The OPU routes
99.6546% of the valid MACs with 73.8223% static tile fill. Its 1,853,155,328 valid routed MACs
divided by 1,024 MACs per target cycle and 2,578,301,427 target compute cycles give **0.0701905%**
whole-inference effective peak; physical array occupancy is unmeasured. Compared with the prior
prepack pair on the same capture, routing, and bitstream, OPU cycles fall 5.6915%, control cycles
fall 2.9556%, and paired speedup rises 2.9010%. The compiler source revisions differ, so this is
an observed package A/B and does not isolate a single knob. The archived bitstream digest matches;
the live Saturn checkout is off its source pin. Exact hashes and denominator are in
`spectformer_hardware_evidence.json`.

The s128 FireSim jobs 3891/3892 await final grading. No K1 or physical Kodiak certification is
claimed.
