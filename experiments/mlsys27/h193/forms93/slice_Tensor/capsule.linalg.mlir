module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<2x5xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0", form = "slice", offsets = [0, 1], sizes = [2, 2], strides = [1, 2], output_dtype = "i8"} : (tensor<2x5xi8>) -> tensor<2x2xi8>
}
