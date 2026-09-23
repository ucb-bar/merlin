module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi32>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<16xi32>
  %Y0 = merlin_iface.bias_add %X, %B {name = "Y0", output_dtype = "i32"} : (tensor<16x16xi32>, tensor<16xi32>) -> tensor<16x16xi32>
}
