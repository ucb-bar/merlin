module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %A0 = merlin_iface.tensor {name = "A0", role = "input"} : tensor<2x16x32xf8E4M3FN>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<2x32x16xf8E4M3FN>
  %A0_scale = merlin_iface.tensor {name = "A0_scale", role = "scale", scale_of = "A0", block = 32 : i64} : tensor<2x1x16xi8>
  %W_scale = merlin_iface.tensor {name = "W_scale", role = "scale", scale_of = "W", block = 32 : i64} : tensor<2x1x16xi8>
  %Y0 = merlin_iface.matmul_batched %A0, %W {name = "Y0", batch = 2 : i64, block_scale = "e8m0", output_dtype = "bf16"} : (tensor<2x16x32xf8E4M3FN>, tensor<2x32x16xf8E4M3FN>) -> tensor<32x16xbf16>
}
