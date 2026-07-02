# raw_baseline_public_v0

Input bundle for the **raw baseline** arm of `gemmini-mlir-oot-capsule`: an un-tooled agent must
produce a `mlir_oot_target_backend` package from the frozen contract + public/dev capsules + public
Gemmini ISA header + the LLVM/MLIR toolchain only. Merlin internals, the reference/simulator,
all prior backends, the hidden capsules+goldens, and prior submissions are **denied** (see
`denied_files.txt` / the `denied` block in `input_bundle_manifest.yaml`). The submitted package must
pass the non-exempt integrity scan. Authoritative spec: `input_bundle_manifest.yaml`.
