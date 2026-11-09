
// RUN: %merlin-opt %s -p merlin-apply-schedule | %filecheck %s
//
// OBLIGATION target transformation -- a reused, immutable RHS must be hoisted, made resident, and
// its layout preserved. These are the schedule decisions residency depends on.

// CHECK-DAG: schedule.hoist_pack
// CHECK-DAG: schedule.place
// CHECK-DAG: schedule.preserve_layout
// CHECK-DAG: schedule.select_interface
builtin.module {
  func.func @repeated_rhs_matmul(%0: tensor<64x128xi8>, %1: tensor<64x128xi8>, %2: tensor<64x128xi8>, %3: tensor<64x128xi8>, %4: tensor<128x64xi8>) -> (tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>) {
    %5 = "contract.capability"() <{sym_name = "toy_npu", features = ["resident_packed_tensor", "accumulator_commit", "command_buffer", "metrics"], runtime = ["simulator", "zephyr"]}> : () -> !contract.capability<"toy_npu">
    "contract.assume"(%4) <{kind = "immutable", lifetime = #contract<lifetime within_region>}> : (tensor<128x64xi8>) -> ()
    "contract.fact"(%4) <{role = #contract<memory_role reusable_weight>, reuse_count = 4 : i64, layout = #contract<layout_role canonical>}> : (tensor<128x64xi8>) -> ()
    "contract.require"() <{feature = "resident_packed_tensor", requires = ["rhs_immutable", "capacity_fit"]}> : () -> ()
    %6 = "contract.prove"(%4) <{requirement = "rhs_immutable", producer_pass = "merlin-infer-contract-facts"}> : (tensor<128x64xi8>) -> !contract.proof<"rhs_immutable">
    "contract.check"(%4, %6) <{requirement = "rhs_immutable"}> : (tensor<128x64xi8>, !contract.proof<"rhs_immutable">) -> ()
    %7 = "contract.prove"(%4) <{requirement = "capacity_fit", producer_pass = "merlin-infer-contract-facts"}> : (tensor<128x64xi8>) -> !contract.proof<"capacity_fit">
    "contract.check"(%4, %7) <{requirement = "capacity_fit"}> : (tensor<128x64xi8>, !contract.proof<"capacity_fit">) -> ()
    %8 = arith.constant 0 : i32
    %9 = tensor.empty() : tensor<64x64xi32>
    %10 = linalg.quantized_matmul ins(%0, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%9 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %11 = tensor.empty() : tensor<64x64xi32>
    %12 = linalg.quantized_matmul ins(%1, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%11 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %13 = tensor.empty() : tensor<64x64xi32>
    %14 = linalg.quantized_matmul ins(%2, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%13 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %15 = tensor.empty() : tensor<64x64xi32>
    %16 = linalg.quantized_matmul ins(%3, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%15 : tensor<64x64xi32>) -> tensor<64x64xi32>
    func.return %10, %12, %14, %16 : tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>
  }
}
