module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<16x16xi8>
  %X1 = merlin_iface.tensor {name = "X1", role = "input"} : tensor<16x16xi8>
  %Y0 = merlin_iface.residual_add %A0, %X1 {name = "Y0", epilogue = [], lhs_scale = 1.0 : f32, rhs_scale = 1.0 : f32, bound_lsb = 0 : i64, output_dtype = "i8"} : (tensor<16x16xi8>, tensor<16x16xi8>) -> tensor<16x16xi8>
}
