// K_conv_1x1_16x16x32_32 (1x1 conv (pointwise = matmul-like))
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x16x16x32xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<32x32xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<32x32xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [1, 1, 32, 32], stride = [1, 1], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"} : (tensor<1x16x16x32xi8>, !merlin_iface.resident) -> tensor<256x32xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
