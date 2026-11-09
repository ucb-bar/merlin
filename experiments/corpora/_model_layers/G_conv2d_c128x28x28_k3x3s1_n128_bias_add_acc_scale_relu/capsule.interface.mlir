module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %IFM = merlin_iface.tensor {name = "IFM", role = "input"} : tensor<1x28x28x128xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<1152x128xi8>
  %B = merlin_iface.tensor {name = "B", role = "bias"} : tensor<128xi32>
  %W_res = merlin_iface.resident_pack %W {layout = "packed_conv_rhs"} : (tensor<1152x128xi8>) -> !merlin_iface.resident
  %Y0 = merlin_iface.conv2d %IFM, %W_res {kernel = [3, 3, 128, 128], stride = [1, 1], padding = [1, 1, 1, 1], dilation = [1, 1], name = "Y0", epilogue = ["bias_add", "acc_scale", "relu"], output_dtype = "i8", acc_scale = 0.005765077542770927 : f32, bias = "B", layout = "nhwc"} : (tensor<1x28x28x128xi8>, !merlin_iface.resident) -> tensor<784x128xi8>
  merlin_iface.evict %W_res : (!merlin_iface.resident) -> ()
}
