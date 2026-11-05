module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x3xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0", form = "repeat", repeats = [2, 1], output_dtype = "i8"} : (tensor<2x3xi8>) -> tensor<4x3xi8>
}
