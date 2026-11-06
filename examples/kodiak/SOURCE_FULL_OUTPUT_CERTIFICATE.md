# Source TinyLlama full-output certificate

## Present evidence

The old exact-ELF Spike extraction in `out/artifacts/targets/rvv/kodiak_prepack_quant_w8a8_v1/s128_full_w8a8_gate.json` records the ELF, debugger command and log, memory dump, output address, full FNV1a32 hash, and a 4,096,000-element numerical gate. `examples/kodiak/UTILIZATION.md` also records the 2,048,000-element s64 extraction. Those gates used Merlin's `golden_w8a8.npy`. They establish a workable extraction method, **not** fidelity to an independent TorchAO reference. The s64 JSON at task `.tmp/tinyllama_s64_full_gate.json` is a grade without the extraction provenance of the s128 receipt.

The source FireSim bridge (`examples/lib/firesim_flow.py`) correctly refuses whole-output acceptance when the console contains only a prefix. No true-source TinyLlama target extraction has been presented yet. Host source mismatch is a separate open issue. Do not amend `target_output_complete`, `all_verified`, or `acceptance` from an old receipt or from device/control hash identity.

## Evidence to obtain for each source leg

Use each **staged exact ELF** from a source FireSim plan. In a Spike debug run, stop after the harness prints `HASH fnv1a32 ...` and before it prints `OUT`; at that point the whole output buffer is resident. Read `MERLIN_OUTPUT_PTR[0]`, dump exactly `4 * golden.size` bytes starting there, resume the *same run* to `DONE`, and retain its complete console, debugger command file, debugger log, raw buffer, Spike binary identity, and output address. Record the halt PC and its location in the ELF disassembly. A debugger script should resolve the pointer from the symbol in this ELF, rather than use an address copied from an older build. Keep each leg's raw dump separate. The device image and RVV control need independent extractions.

The resulting per-leg certificate should carry these immutable fields:

| Group | Required values |
| --- | --- |
| Plan | schema, job identifier, leg, board name, bitstream configuration, harts, capture SHA-256, framework-result SHA-256, reference kind `torchao_w8a8_framework`, `golden.npy` SHA-256 |
| Image | staged ELF SHA-256, build hash, exact Spike `out_hash` and prefix SHA-256 from the plan |
| Extraction | Spike executable/version and SHA-256, debugger command SHA-256, debugger log SHA-256, same-run console SHA-256, raw buffer SHA-256, output pointer address, byte count, halt PC and symbol/disassembly evidence |
| Verification | raw FNV1a32 and element count, raw leading-prefix SHA-256 and count, full `_gate` result with `min_coverage=1`, full tensor shape/dtype, per-token argmax comparison |

Hash the raw little-endian f32 bytes *before* NumPy conversion. Verify the byte count equals `golden.size * 4`, then reinterpret as `<f4` and reshape to the golden shape. Reject missing bytes, mismatched shape, non-finite values, a changed capture, framework result, reference, staged ELF, debugger artifacts, or console. Parse the same-run console with `_parse_console`; require `DONE`, matching build hash, the exact planned whole-output HASH and prefix SHA-256, then require the raw buffer's FNV1a32 and leading-prefix SHA-256 to match that console. Run `_gate(full, {"w8a8": golden}, min_coverage=1)` and require `ok`, `tier_ok == "w8a8"`, `comparison_complete`, and `n_compared == n_reference == golden.size`. Compare each TinyLlama token's argmax, rather than relying on the gate's single global argmax. Do not accept a `golden_w8a8.npy` from the older Merlin-derived bundles in this path.

Compute a stable plan identity from the exact fields in the first two table rows, with canonical JSON, instead of hashing the mutable `firesim_job.json` file: FireSim submission may update its `jobs` map. Recheck every pin while collecting a result. Preserve the raw buffer and evidence files alongside the certificate; a JSON statement of their hashes without the files cannot be reverified. Certificate creation and verification should be separate commands, and a synthetic test must demonstrate rejection of a corrupted raw tail, wrong ELF, wrong capture/reference, mismatched same-run console, missing debugger artifact, and prefix-only gate.

## Certificate command

`examples/kodiak/source_output_certificate.py` implements the separate
`create` and `verify` steps without changing the FireSim plan. Stage the raw
dump, complete same-run console, debugger commands and log, exact ELF
disassembly, and Spike version text under `JOB/full_output/LEG/` first. The
Spike executable itself may live elsewhere; its bytes are hashed. For each
leg, provide the observed buffer address and halt PC from that exact run:

```sh
JOB=/path/to/exact/source/fireSim/run_123
LEG=device
EVIDENCE="$JOB/full_output/$LEG"
PYTHONPATH=src python examples/kodiak/source_output_certificate.py create "$JOB" \
  --leg "$LEG" --raw "$EVIDENCE/raw.bin" \
  --console "$EVIDENCE/console.txt" \
  --debugger-commands "$EVIDENCE/commands.txt" \
  --debugger-log "$EVIDENCE/debugger.log" \
  --disassembly "$EVIDENCE/disassembly.txt" \
  --spike-version "$EVIDENCE/spike_version.txt" \
  --spike /path/to/exact/spike --output-address 0x80002000 --halt-pc 0x80001000
PYTHONPATH=src python examples/kodiak/source_output_certificate.py verify \
  "$EVIDENCE/certificate.json"
```

The hexadecimal addresses above are placeholders, not a pointer recipe.
The debugger command file must resolve `MERLIN_OUTPUT_PTR[0]` from the staged
ELF, and its log and disassembly must identify the observed addresses. The
tool refuses a short dump, mismatched console or image, altered evidence,
non-finite values, a failed complete TorchAO W8A8 gate, or a per-token
argmax mismatch. The certificate stays in its original job directory so the
verifier can reopen every referenced file. Queue `jobs` changes do not
invalidate its stable plan identity. A certificate's FNV check is still
non-cryptographic, as described below. The tool validates the staged evidence
and its consistency with the console; it cannot independently attest that an
operator captured the debugger files from the claimed physical process.

## Acceptance boundary

Only a validated certificate from **both** exact source Spike legs can set `target_output_complete_on_spike=true`. The existing FireSim hardware check must still verify its board/bitstream artifact, staged ELF, full HASH and prefix against the corresponding certified Spike run, plus normal completion and cycles. Its current FNV1a32 is 32-bit and non-cryptographic: this proves whole-buffer *consistency under the FNV check*, not cryptographic equality of the unprinted hardware tail. Report that limit explicitly. For a cryptographic whole-output hardware claim, add a target-produced SHA-256 over the output bytes or an independently captured full FireSim memory buffer from the same run; bind it to the certificate and hardware job. Never mark source hardware whole-output fidelity as proved by matching device/control FNV hashes alone.

Do not wire certificate acceptance into `build_legs` or `collect` until a true-source staged ELF has produced the extraction above and the independent TorchAO host gate is passing. The current `diagnostic_only` flag should remain set until then.
