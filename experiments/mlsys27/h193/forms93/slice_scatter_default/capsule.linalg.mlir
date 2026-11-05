module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X0 = merlin_iface.tensor {name = "X0", role = "input"} : tensor<2x5xi8>
  %X1 = merlin_iface.tensor {name = "X1", role = "input"} : tensor<2x2xi8>
  %Y0 = merlin_iface.insert_slice %X0, %X1 {name = "Y0", offsets = [0, 1], strides = [1, 2], output_dtype = "i8"} : (tensor<2x5xi8>, tensor<2x2xi8>) -> tensor<2x5xi8>
}
