# Response: define_verification_ladder

## Summary
Verification ladder for Gemmini MX:

1. Smoke: `./merlin build --profile gemmini` succeeds.
2. Compile: `./merlin compile <matmul.mlir> --target gemmini_mx` produces
   a non-empty `.vmfb`.
3. Sim: `./merlin chipyard run gemmini-mx <elf>` exits 0 with the
   `gemmini-rocc-tests/matmul_ws_mx_generic` reference binary.
4. FireSim: bitstream build + workload run on U250.

## Conclusion
Ladder recorded. No mutation required. Task should reach `completed`.
