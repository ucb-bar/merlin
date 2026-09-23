// GQ3: 3x3 stride-2 int8 conv over 8x8x8 -> 3x3x16 with the quantized epilogue fused.
module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x8x8x8xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<72x16xi8>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<16xi32>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<72x16xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 8, 16], stride = [2, 2], padding = [0, 0, 0, 0], dilation = [1, 1], name = "Y0", epilogue = ["bias_add", "acc_scale", "relu"], output_dtype = "i8", acc_scale = 0.25 : f32, bias = "B", layout = "nhwc"} : (tensor<1x8x8x8xi8>, !merlin_iface.resident) -> tensor<9x16xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
