module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %X = merlin_iface.tensor {name = "X", role = "input"} : tensor<16x16xi8>
  %Y0 = merlin_iface.movement %X {name = "Y0", form = "as_strided", sizes = [16], strides = [17], storage_offset = 0 : i64, output_dtype = "i8"} : (tensor<16x16xi8>) -> tensor<16xi8>
}
