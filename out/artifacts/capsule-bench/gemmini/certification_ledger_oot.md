# Gemmini conformance battery — RTL-certified THROUGH THE EXPERIMENT-ABI CONTRACT.
#
# CANONICAL, regenerable from recorded run manifests alone:
#   python -m merlin.eval.gemmini_dispatcher --summary-only --runs-root runs/gemmini_cert_oot
# Each cell flowed through merlin.targetgen.oot_runner against the merlin_native_v0 package
# (subprocess + file contract). 9 RTL-certified (verilator) + 9 bootstrap (spike). DO NOT EDIT.

| rung | backend | oracle.kind | derived_from_rtl | cycle_accurate | correct | cycles | run_id | artifacts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C0 | oot_package | rtl_verilator | True | True | True | 308 | C0_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C1 | oot_package | rtl_verilator | True | True | True | 308 | C1_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C4 | oot_package | rtl_verilator | True | True | True | 1006 | C4_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C4e | oot_package | rtl_verilator | True | True | True | 636 | C4e_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C5 | oot_package | rtl_verilator | True | True | True | 657 | C5_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q0 | oot_package | rtl_verilator | True | True | True | 250 | Q0_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q1 | oot_package | rtl_verilator | True | True | True | 250 | Q1_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q1t | oot_package | rtl_verilator | True | True | True | 940 | Q1t_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q2 | oot_package | rtl_verilator | True | True | True | 274 | Q2_verilator_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C0 | oot_package | spike_gemmini_functional | False | False | True | 47 | C0_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C1 | oot_package | spike_gemmini_functional | False | False | True | 47 | C1_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C4 | oot_package | spike_gemmini_functional | False | False | True | 89 | C4_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C4e | oot_package | spike_gemmini_functional | False | False | True | 70 | C4e_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| C5 | oot_package | spike_gemmini_functional | False | False | True | 73 | C5_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q0 | oot_package | spike_gemmini_functional | False | False | True | 51 | Q0_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q1 | oot_package | spike_gemmini_functional | False | False | True | 51 | Q1_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q1t | oot_package | spike_gemmini_functional | False | False | True | 100 | Q1t_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
| Q2 | oot_package | spike_gemmini_functional | False | False | True | 52 | Q2_spike_oot_native | interface_mlir:generated, target_mlir:compiler_generated, command_buffer:compiler_generated, llvm_ir:compiler_generated, object:compiler_generated, log:oracle_output |
