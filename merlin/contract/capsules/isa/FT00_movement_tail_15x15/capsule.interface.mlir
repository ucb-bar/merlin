module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<15x15xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0", semantic = "mvin_mvout", output_dtype = "i32"} : (tensor<15x15xi8>) -> tensor<15x15xi32>
}
