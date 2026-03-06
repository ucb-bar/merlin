// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: iree-compile %s \
// RUN:   --iree-hal-target-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=riscv64-unknown-linux-gnu \
// RUN:   --iree-llvmcpu-target-abi=lp64d \
// RUN:   --iree-llvmcpu-target-cpu-features=+m,+a,+f,+d,+c,+v,+buddyext \
// RUN:   --iree-llvmcpu-enable-gemmini-linalg-lowering \
// RUN:   --iree-hal-dump-executable-files-to=%t.dir \
// RUN:   -o %t.dir/test_gemmini.vmfb
// RUN: FileCheck %s --input-file=%t.dir/module_matmul_dispatch_0_embedded_elf_riscv_64.s

// CHECK: config_ex
// CHECK: loop_ws_config_bounds
// CHECK: loop_ws
// CHECK: flush

module {
  func.func @matmul(%lhs: tensor<4x4xf32>, %rhs: tensor<4x4xf32>, %acc: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<4x4xf32>, tensor<4x4xf32>)
      outs(%acc : tensor<4x4xf32>)
      -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}
