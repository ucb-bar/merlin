// K_conv_3x3_stride2_16x16x16_32 (strided 3x3 (downsample))
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x16x16x16xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<144x32xi8>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<144x32xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 16, 32], stride = [2, 2], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "i32", layout = "nhwc"} : (tensor<1x16x16x16xi8>, !merlin_iface.resident) -> tensor<49x32xi32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
