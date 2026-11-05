module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<2x3xi8>
  %X1 = merlin_iface.tensor {name = "X1", role = "input"} : tensor<2x3xi8>
  %Y0 = merlin_iface.concat %X0, %X1 {name = "Y0", axis = 0 : i64, output_dtype = "i8"} : (tensor<2x3xi8>, tensor<2x3xi8>) -> tensor<4x3xi8>
}
