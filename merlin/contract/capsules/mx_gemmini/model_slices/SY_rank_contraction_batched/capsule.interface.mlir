module attributes {merlin_iface.version = "0.1", merlin_iface.target = "mx_gemmini", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<2x16x32xbf16>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<2x32x16xbf16>
  %Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", batch = 2 : i64, output_dtype = "f32"} : (tensor<2x16x32xbf16>, tensor<2x32x16xbf16>) -> tensor<32x16xf32>
}
