# Host rotary qualification

Option B uses captured source f32 arithmetic on the host and the existing
Merlin unary-math route, with the candidate policy in
host-unary-math-policy.json. CORE-MATH revision
a055b9b9a86b857e49573e339cf68306ad92c831 provides correctly rounded sinf/cosf.
The two unmodified source files and MIT LICENSE are vendored under third_party;
no upstream build scripts or Makefiles run. The archive is a generated artifact,
not checked in. Its source hashes, compiler flags, symbol roster and archive pin
are in out/core-math/build-receipt.json. This is a source arithmetic policy, not an Atlas
rotary implementation or a claim of f32 framework equivalence.

Run `bash scripts/run_rotary_gate.sh` here. Set RISCV to the pinned RV64GC/lp64d GCC 13.2 toolchain prefix and
TORCH_PYTHON to the source torch interpreter. The script checks the
archive hash, runs an existing Spike/pk binary, and checks all 262,144 words.
The result is PASS: zero BF16 mismatches over positions 0..2047. Implicit FMA
contraction is disabled; explicit source __builtin_fmaf stays fused (fnmsub.s).
The helper round resolves from toolchain libm and __ctzdi2 from libgcc; the link
map records the selected archive as sole supplier of sinf/cosf. The unobserved
flags/errno contract permits -fno-math-errno. No simulator is built.

scripts/inv_freq.hex retains the captured 128-byte inverse-frequency buffer.
Every f32 angle agrees with torch's K=1 product; transpose/cat duplicate its
lanes, then source trig, attention scaling by f32 1.0, and BF16 RNE compose the
rotary words. Independent bounded Python arithmetic is checked separately by
reference v3. Positions beyond 2047 are unqualified. Chipyard newlib was rejected without tolerance or domain reduction because
sine changed a BF16 word at position 1897/frequency 24.
