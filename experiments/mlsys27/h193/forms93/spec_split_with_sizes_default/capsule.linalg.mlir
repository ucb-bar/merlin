module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0", form = "slice", offsets = [0, 0], sizes = [3, 16], strides = [1, 1], output_dtype = "i8"} : (tensor<16x16xi8>) -> tensor<3x16xi8>
  %Y1 = merlin_iface.movement %X {name = "Y1", form = "slice", offsets = [3, 0], sizes = [5, 16], strides = [1, 1], output_dtype = "i8"} : (tensor<16x16xi8>) -> tensor<5x16xi8>
  %Y2 = merlin_iface.movement %X {name = "Y2", form = "slice", offsets = [8, 0], sizes = [8, 16], strides = [1, 1], output_dtype = "i8"} : (tensor<16x16xi8>) -> tensor<8x16xi8>
}
