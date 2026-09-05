module attributes {merlin_iface.version = "0.1", merlin_iface.target = "radiance", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x4x4x4xf16>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<36x16xf16>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<36x16xf16>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 4, 16], stride = [1, 1], padding = [1, 1, 1, 1], dilation = [1, 1], name = "Y0", epilogue = [], output_dtype = "f32", layout = "nhwc"} : (tensor<1x4x4x4xf16>, !merlin_iface.resident) -> tensor<16x16xf32>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
