# Gemmini certification ledger — GENERATED from recorded AET run manifests.
#
# Do NOT edit by hand. Regenerate with:
#   python -m merlin.eval.gemmini_dispatcher --summary-only --runs-root runs/gemmini_cert
# Each row is backed by an isolated run dir under runs/gemmini_cert/ with run_manifest.yaml,
# origin-tagged artifact_manifest.json (command_buffer/mlir/llvm_ir/object/harness/console),
# toolchain SHAs, and metrics. Spike = bootstrap (cycle_accurate False); Verilator = cert gate.
#
# Acceptance: this table is reproducible from manifests ALONE — no hand-authored YAML.

| rung | backend | oracle.kind | derived_from_rtl | cycle_accurate | correct | cycles | run_id | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 308 | C0_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C1 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 308 | C1_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C4 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 1006 | C4_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C4e | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 636 | C4e_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C5 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 657 | C5_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q0 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 250 | Q0_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q1 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 250 | Q1_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q1t | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 940 | Q1t_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q2 | mlir_inline_asm_rocc | rtl_verilator | True | True | True | 274 | Q2_verilator_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C0 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 47 | C0_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C1 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 47 | C1_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C4 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 89 | C4_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C4e | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 70 | C4e_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| C5 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 73 | C5_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q0 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 51 | Q0_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q1 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 51 | Q1_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q1t | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 100 | Q1t_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
| Q2 | mlir_inline_asm_rocc | spike_gemmini_functional | False | False | True | 52 | Q2_spike_mlir_inline_asm_rocc_seed000 | command_buffer:generated, mlir:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, harness:compiler_generated, log:oracle_output |
