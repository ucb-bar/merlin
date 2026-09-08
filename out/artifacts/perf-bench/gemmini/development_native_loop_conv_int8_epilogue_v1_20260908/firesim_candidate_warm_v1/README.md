# ResNet-50 native LOOP_CONV hybrid, warm-measured FireSim candidate

This candidate replaces all 53 TVM convolution calls with uniquely named
Merlin-generated native `LOOP_CONV` kernels.  The TVM-generated host runner
still owns the 20 residual rescale/add/ReLU operations, first maxpool, global
average pool/flatten, dense layer, activation arena, and call graph.

Local Spike qualification is exact against Jack's independently generated TVM
INT8 reference: 1000/1000 logits match, maximum absolute difference is zero,
and top-1 is 258.  The program performs one complete untimed warm inference,
restores its input, and times one complete measured inference.  Its local Spike
window is 365,212,240 cycles.  The 338,090,625-cycle Jack TVM number is a cold
single run, so the arithmetic 1.0802 ratio is recorded but not treated as a
protocol-matched performance comparison.

FireSim queue job 536 passed on the pinned Universal Gemmini U250 image at
555,991,472 measured cycles.  All 1000 UART logits match the independent Jack
TVM reference byte-for-byte and top-1 is 258.  This is 2.765x faster than q534
(63.84% fewer cycles) and 2.368x faster than q535 (57.77% fewer cycles).  The
queue completed its leading kill, infrasetup, runworkload, and trailing kill in
260.1 seconds with an invariant HWDB hash.  The staged ELF hash matches this
bundle's ELF hash.

`verify_bundle.sh` checks the exact ELF, bitstream, queue wrappers, local gates,
and q536 receipt.  `run_firesim.sh` is now blocked by the sealed readiness file
to prevent an unchanged resubmission.

The local working artifact retains the 44 MB pinned bitstream archive used by
q536.  It is reused immutable hardware collateral rather than newly generated
work, so the public Git snapshot records its SHA-256 but does not track that
archive.  `verify_bundle.sh` checks it when present and always verifies the
sealed hardware receipt.

No full golden/logit vector or UART answer-bearing file is packaged.  Receipts
contain only counts, hashes, mismatch totals, and top-1; the immutable queue
authority retains the raw q536 UART at the path recorded in the receipt.
