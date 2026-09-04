module attributes {merlin_iface.version = "0.1", merlin_iface.target = "atlas", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<2x32x64xf8E4M3FN>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<2x64x32xf8E4M3FN>
  %Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", batch = 2 : i64, output_dtype = "bf16"} : (tensor<2x32x64xf8E4M3FN>, tensor<2x64x32xf8E4M3FN>) -> tensor<64x32xbf16>
}
