
// RUN: %merlin-opt %s -p merlin-materialize-interface | %filecheck %s
//
// OBLIGATION boundary materialization, and the structural half of the manifest obligations
// `must_prove_rhs_immutable_for_residency` and `must_commit_accumulator_before_reuse`.
//
// A reused resident weight must be packed ONCE, used by every matmul, and evicted after the LAST
// use. Each accumulator must be committed exactly once. CHECK lines are ordered, so a pack emitted
// after a use, or an evict before the last matmul, fails here in milliseconds -- the same defect
// would otherwise surface only as a wrong number at the RTL tier.
//
// The CHECK-NOT BETWEEN the two commits is what makes commit-once a real assertion: without it a
// program that commits the same accumulator twice satisfies this sequence (measured 2026-09-04 --
// the `duplicate_commit` mutation passed this file while the derived per-target check caught it).

// CHECK:     interface.resident_pack
// CHECK-NOT: interface.resident_pack
// CHECK:     interface.matmul
// CHECK:     interface.commit
// CHECK-NOT: interface.commit
// CHECK:     interface.matmul
// CHECK:     interface.commit
// CHECK:     interface.resident_evict
// CHECK-NOT: interface.matmul
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
    "schedule.hoist_pack"(%4) <{outside = "@loop_i", layout = "packed_rhs"}> : (tensor<128x64xi8>) -> ()
    "schedule.preserve_layout"(%4) <{layout = "packed_rhs", scope = "region"}> : (tensor<128x64xi8>) -> ()
    "schedule.place"(%4, %5) <{state = #schedule<memory_state resident>, lifetime = "region"}> : (tensor<128x64xi8>, !contract.capability<"toy_npu">) -> ()
    "schedule.select_interface"(%4) <{interface = "resident_packed_tensor", reason = "reuse_count >= 2 and capacity_fit proven", visibility = #schedule<visibility software_visible>}> : (tensor<128x64xi8>) -> ()
    %8 = arith.constant 0 : i32
    %9 = tensor.empty() : tensor<64x64xi32>
    %10 = linalg.quantized_matmul ins(%0, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%9 : tensor<64x64xi32>) -> tensor<64x64xi32>
    "schedule.keep_accumulator_live"(%10) <{until = "epilogue_commit"}> : (tensor<64x64xi32>) -> ()
    %11 = tensor.empty() : tensor<64x64xi32>
    %12 = linalg.quantized_matmul ins(%1, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%11 : tensor<64x64xi32>) -> tensor<64x64xi32>
    "schedule.keep_accumulator_live"(%12) <{until = "epilogue_commit"}> : (tensor<64x64xi32>) -> ()
    %13 = tensor.empty() : tensor<64x64xi32>
    %14 = linalg.quantized_matmul ins(%2, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%13 : tensor<64x64xi32>) -> tensor<64x64xi32>
    "schedule.keep_accumulator_live"(%14) <{until = "epilogue_commit"}> : (tensor<64x64xi32>) -> ()
    %15 = tensor.empty() : tensor<64x64xi32>
    %16 = linalg.quantized_matmul ins(%3, %4, %8, %8 : tensor<64x128xi8>, tensor<128x64xi8>, i32, i32) outs(%15 : tensor<64x64xi32>) -> tensor<64x64xi32>
    "schedule.keep_accumulator_live"(%16) <{until = "epilogue_commit"}> : (tensor<64x64xi32>) -> ()
    "schedule.group_dispatch"(%10, %12, %14, %16) <{granularity = #schedule<dispatch_granularity command_buffer>}> : (tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>) -> ()
    func.return %10, %12, %14, %16 : tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>, tensor<64x64xi32>
  }
}
