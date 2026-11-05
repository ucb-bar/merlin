module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi8>
  %sel = merlin_iface.movement %X {name = "sel", form = "slice", offsets = [8, 0], sizes = [1, 16], strides = [1, 1], output_dtype = "i8"} : (tensor<16x16xi8>) -> tensor<1x16xi8>
  %Y0 = merlin_iface.movement %sel {name = "Y0", form = "reshape", output_dtype = "i8"} : (tensor<1x16xi8>) -> tensor<16xi8>
}
