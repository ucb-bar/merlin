module attributes {merlin_iface.version = "0.1", merlin_iface.target = "gemmini", merlin_iface.abi_version = "0.1"} {
  %Xcol = merlin_iface.tensor {name = "Xcol", role = "input"} : tensor<16x36xi8>
  %dY = merlin_iface.tensor {name = "dY", role = "input"} : tensor<16x16xi8>
  %W = merlin_iface.tensor {name = "W", role = "weight"} : tensor<36x16xi8>
  %xt = merlin_iface.movement %Xcol {name = "xt", form = "permute", perm = [1, 0], output_dtype = "i8"} : (tensor<16x36xi8>) -> tensor<36x16xi8>
  %dY_res = merlin_iface.resident_pack %dY {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc_w = merlin_iface.matmul %xt, %dY_res : (tensor<36x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %grad_weight = merlin_iface.commit %acc_w {name = "grad_weight", epilogue = [], output_dtype = "i32", prov.aten = "aten.convolution_backward.default"} : (!merlin_iface.acc<i32>) -> tensor<36x16xi32>
  %dyt = merlin_iface.movement %dY {name = "dyt", form = "permute", perm = [1, 0], output_dtype = "i8"} : (tensor<16x16xi8>) -> tensor<16x16xi8>
  %dyt_res = merlin_iface.resident_pack %dyt {layout = "packed_rhs"} : (tensor<16x16xi8>) -> !merlin_iface.resident
  %acc_i = merlin_iface.matmul %W, %dyt_res : (tensor<36x16xi8>, !merlin_iface.resident) -> !merlin_iface.acc<i32>
  %gic = merlin_iface.commit %acc_i {name = "gic", epilogue = ["acc_scale"], output_dtype = "i8", acc_scale = 0.00048828125 : f32, prov.aten = "aten.convolution_backward.default"} : (!merlin_iface.acc<i32>) -> tensor<36x16xi8>
  %gic3 = merlin_iface.movement %gic {name = "gic3", form = "reshape", output_dtype = "i8"} : (tensor<36x16xi8>) -> tensor<1x36x16xi8>
  %grad_input = merlin_iface.window_sum %gic3 {name = "grad_input", mode = "fold", kernel = [3, 3], stride = [1, 1], padding = [1, 1], dilation = [1, 1], layout = "nchw", readout_scale = 1.0 : f32, output_dtype = "i32", prov.aten = "aten.convolution_backward.default"} : (tensor<1x36x16xi8>) -> tensor<1x4x4x4xi32>
  %grad_bias = merlin_iface.window_sum %dY {name = "grad_bias", mode = "reduce", dims = [0], average = false, layout = "nchw", readout_scale = 1.0 : f32, output_dtype = "i32", prov.aten = "aten.convolution_backward.default"} : (tensor<16x16xi8>) -> tensor<1x16xi32>
}
