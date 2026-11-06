# Same-source SpectFormer round-even FireSim comparison

This comparison tests the `lower_roundeven_to_intrinsic` package knob on the pinned
one-core Kodiak OPU FireSim image. The two packages use a byte-identical schedule,
the same capture SHA `6ba976f5c848018a2872c959eada50280a081f317b9cd83feeb624030af443c3`,
the same compiler source commit `ac314735f66349ab270b5caa2934351b33fc2ebc`,
and the same routing report SHA
`2cf24c59ea9aa8529bed811bf028cbfc578942b19aa34633b62c3e5b1c8265fc`.
Their `knobs.yaml` files differ only by the added `lower_roundeven_to_intrinsic` feature.
All four exact ELFs passed their Spike gates before submission.

| Package | FireSim jobs, OPU / plain RVV | Target compute cycles, OPU / RVV | RVV / OPU | Whole-inference effective OPU peak |
| --- | --- | ---: | ---: | ---: |
| Baseline | 3895 / 3896 | 2,733,902,458 / 5,840,140,166 | 2.136191856× | 0.06619556% |
| Round-even intrinsic | 3897 / 3898 | 2,539,617,917 / 5,660,152,596 | 2.228741795× | 0.07125962% |

Every job completed with `DONE`, queue rc 0, and a verified grade. Each entire
1,000-value output matched its own exact Spike full-output hash (`FNV1a32 3242232599`)
and printed prefix; all four images had the same identity. All 1,000 values passed the
earlier Merlin-host W8A8 **numerical** gate. That golden is close rather than
byte-identical (`out_hash_matches_w8a8=false` in each collected result). The one-core
bitstream artifact SHA was
`f671be16ae5850c2765d897593010f4144654fe49bed0d5fe0d3ebdbd49d9462`.
The original plan preflight marks source-pin reproduction incomplete; the tested
artifact bytes and hardware config were independently verified at collection.

Both OPU images route 90 contractions and perform **1,853,155,328 valid MACs**. The
other contractions perform 6,422,528 MACs, giving **99.6546% routed MAC share**.
Their 2,510,290,944 padded logical tile slots give **73.8223% static tile fill**.
The board's capacity bound is 1,024 signed-int8 MACs per target cycle. Dividing
valid routed MACs by `OPU target compute cycles × 1,024` gives the two whole-inference
effective peak fractions in the table. The intrinsic device saves **194,284,541
cycles (7.10649%)**; the matched RVV control saves 179,987,570 cycles (3.08190%).
The paired speedup ratio rises 4.33247%.

These unprofiled ELFs emit no `PROF` call rows, and the bitstream has no physical
MAC-cell enable counter. Their OPU call-window cycles and physical array occupancy
are therefore **unmeasured**; static tile fill and whole-inference throughput are
different quantities. Do not borrow the separate profiled SpectFormer image's
call-window result for these four ELFs.

The capture stores torchAO weight-only int8 Linear weights, then Merlin applies its
own W8A8 transform. This A/B measures those exact images and **does not establish
parity with torchAO's source-exported W8A8 program**. The full-output TorchAO-source
numerical gate remains separate.

The SHA-bound collector receipt lives in managed task
`spectformer-ab-utilization-collector/.tmp/spectformer_roundintrinsic_ab_receipt.json`
(SHA256 `8932f945de4fc41a8c327ad3d9e337736fd7e8e05a9b492f4bf5b6c75dfd51d3`).
It binds plan, ELF, UART, package, bitstream, routing, and collected-result hashes.
