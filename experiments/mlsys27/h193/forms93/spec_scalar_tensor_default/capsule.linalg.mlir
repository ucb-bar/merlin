module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Y0 = merlin_iface.fill  {name = "Y0", value_code = 5 : i64, output_dtype = "i8"} : () -> tensor<i8>
}
