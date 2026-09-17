builtin.module attributes {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors", prov.quantization = "int8_dyn_act_int8_weight"} {
  func.func @forward(%0: tensor<64x3x7x7xf32>, %1: tensor<64xf32>, %2: tensor<64xf32>, %3: tensor<64x64x1x1xf32>, %4: tensor<64xf32>, %5: tensor<64xf32>, %6: tensor<64x64x3x3xf32>, %7: tensor<64xf32>, %8: tensor<64xf32>, %9: tensor<256x64x1x1xf32>, %10: tensor<256xf32>, %11: tensor<256xf32>, %12: tensor<256x64x1x1xf32>, %13: tensor<256xf32>, %14: tensor<256xf32>, %15: tensor<64x256x1x1xf32>, %16: tensor<64xf32>, %17: tensor<64xf32>, %18: tensor<64x64x3x3xf32>, %19: tensor<64xf32>, %20: tensor<64xf32>, %21: tensor<256x64x1x1xf32>, %22: tensor<256xf32>, %23: tensor<256xf32>, %24: tensor<64x256x1x1xf32>, %25: tensor<64xf32>, %26: tensor<64xf32>, %27: tensor<64x64x3x3xf32>, %28: tensor<64xf32>, %29: tensor<64xf32>, %30: tensor<256x64x1x1xf32>, %31: tensor<256xf32>, %32: tensor<256xf32>, %33: tensor<128x256x1x1xf32>, %34: tensor<128xf32>, %35: tensor<128xf32>, %36: tensor<128x128x3x3xf32>, %37: tensor<128xf32>, %38: tensor<128xf32>, %39: tensor<512x128x1x1xf32>, %40: tensor<512xf32>, %41: tensor<512xf32>, %42: tensor<512x256x1x1xf32>, %43: tensor<512xf32>, %44: tensor<512xf32>, %45: tensor<128x512x1x1xf32>, %46: tensor<128xf32>, %47: tensor<128xf32>, %48: tensor<128x128x3x3xf32>, %49: tensor<128xf32>, %50: tensor<128xf32>, %51: tensor<512x128x1x1xf32>, %52: tensor<512xf32>, %53: tensor<512xf32>, %54: tensor<128x512x1x1xf32>, %55: tensor<128xf32>, %56: tensor<128xf32>, %57: tensor<128x128x3x3xf32>, %58: tensor<128xf32>, %59: tensor<128xf32>, %60: tensor<512x128x1x1xf32>, %61: tensor<512xf32>, %62: tensor<512xf32>, %63: tensor<128x512x1x1xf32>, %64: tensor<128xf32>, %65: tensor<128xf32>, %66: tensor<128x128x3x3xf32>, %67: tensor<128xf32>, %68: tensor<128xf32>, %69: tensor<512x128x1x1xf32>, %70: tensor<512xf32>, %71: tensor<512xf32>, %72: tensor<256x512x1x1xf32>, %73: tensor<256xf32>, %74: tensor<256xf32>, %75: tensor<256x256x3x3xf32>, %76: tensor<256xf32>, %77: tensor<256xf32>, %78: tensor<1024x256x1x1xf32>, %79: tensor<1024xf32>, %80: tensor<1024xf32>, %81: tensor<1024x512x1x1xf32>, %82: tensor<1024xf32>, %83: tensor<1024xf32>, %84: tensor<256x1024x1x1xf32>, %85: tensor<256xf32>, %86: tensor<256xf32>, %87: tensor<256x256x3x3xf32>, %88: tensor<256xf32>, %89: tensor<256xf32>, %90: tensor<1024x256x1x1xf32>, %91: tensor<1024xf32>, %92: tensor<1024xf32>, %93: tensor<256x1024x1x1xf32>, %94: tensor<256xf32>, %95: tensor<256xf32>, %96: tensor<256x256x3x3xf32>, %97: tensor<256xf32>, %98: tensor<256xf32>, %99: tensor<1024x256x1x1xf32>, %100: tensor<1024xf32>, %101: tensor<1024xf32>, %102: tensor<256x1024x1x1xf32>, %103: tensor<256xf32>, %104: tensor<256xf32>, %105: tensor<256x256x3x3xf32>, %106: tensor<256xf32>, %107: tensor<256xf32>, %108: tensor<1024x256x1x1xf32>, %109: tensor<1024xf32>, %110: tensor<1024xf32>, %111: tensor<256x1024x1x1xf32>, %112: tensor<256xf32>, %113: tensor<256xf32>, %114: tensor<256x256x3x3xf32>, %115: tensor<256xf32>, %116: tensor<256xf32>, %117: tensor<1024x256x1x1xf32>, %118: tensor<1024xf32>, %119: tensor<1024xf32>, %120: tensor<256x1024x1x1xf32>, %121: tensor<256xf32>, %122: tensor<256xf32>, %123: tensor<256x256x3x3xf32>, %124: tensor<256xf32>, %125: tensor<256xf32>, %126: tensor<1024x256x1x1xf32>, %127: tensor<1024xf32>, %128: tensor<1024xf32>, %129: tensor<512x1024x1x1xf32>, %130: tensor<512xf32>, %131: tensor<512xf32>, %132: tensor<512x512x3x3xf32>, %133: tensor<512xf32>, %134: tensor<512xf32>, %135: tensor<2048x512x1x1xf32>, %136: tensor<2048xf32>, %137: tensor<2048xf32>, %138: tensor<2048x1024x1x1xf32>, %139: tensor<2048xf32>, %140: tensor<2048xf32>, %141: tensor<512x2048x1x1xf32>, %142: tensor<512xf32>, %143: tensor<512xf32>, %144: tensor<512x512x3x3xf32>, %145: tensor<512xf32>, %146: tensor<512xf32>, %147: tensor<2048x512x1x1xf32>, %148: tensor<2048xf32>, %149: tensor<2048xf32>, %150: tensor<512x2048x1x1xf32>, %151: tensor<512xf32>, %152: tensor<512xf32>, %153: tensor<512x512x3x3xf32>, %154: tensor<512xf32>, %155: tensor<512xf32>, %156: tensor<2048x512x1x1xf32>, %157: tensor<2048xf32>, %158: tensor<2048xf32>, %159: tensor<1000x2048xf32>, %160: tensor<1000xf32>, %161: tensor<64xf32>, %162: tensor<64xf32>, %163: tensor<i64>, %164: tensor<64xf32>, %165: tensor<64xf32>, %166: tensor<i64>, %167: tensor<64xf32>, %168: tensor<64xf32>, %169: tensor<i64>, %170: tensor<256xf32>, %171: tensor<256xf32>, %172: tensor<i64>, %173: tensor<256xf32>, %174: tensor<256xf32>, %175: tensor<i64>, %176: tensor<64xf32>, %177: tensor<64xf32>, %178: tensor<i64>, %179: tensor<64xf32>, %180: tensor<64xf32>, %181: tensor<i64>, %182: tensor<256xf32>, %183: tensor<256xf32>, %184: tensor<i64>, %185: tensor<64xf32>, %186: tensor<64xf32>, %187: tensor<i64>, %188: tensor<64xf32>, %189: tensor<64xf32>, %190: tensor<i64>, %191: tensor<256xf32>, %192: tensor<256xf32>, %193: tensor<i64>, %194: tensor<128xf32>, %195: tensor<128xf32>, %196: tensor<i64>, %197: tensor<128xf32>, %198: tensor<128xf32>, %199: tensor<i64>, %200: tensor<512xf32>, %201: tensor<512xf32>, %202: tensor<i64>, %203: tensor<512xf32>, %204: tensor<512xf32>, %205: tensor<i64>, %206: tensor<128xf32>, %207: tensor<128xf32>, %208: tensor<i64>, %209: tensor<128xf32>, %210: tensor<128xf32>, %211: tensor<i64>, %212: tensor<512xf32>, %213: tensor<512xf32>, %214: tensor<i64>, %215: tensor<128xf32>, %216: tensor<128xf32>, %217: tensor<i64>, %218: tensor<128xf32>, %219: tensor<128xf32>, %220: tensor<i64>, %221: tensor<512xf32>, %222: tensor<512xf32>, %223: tensor<i64>, %224: tensor<128xf32>, %225: tensor<128xf32>, %226: tensor<i64>, %227: tensor<128xf32>, %228: tensor<128xf32>, %229: tensor<i64>, %230: tensor<512xf32>, %231: tensor<512xf32>, %232: tensor<i64>, %233: tensor<256xf32>, %234: tensor<256xf32>, %235: tensor<i64>, %236: tensor<256xf32>, %237: tensor<256xf32>, %238: tensor<i64>, %239: tensor<1024xf32>, %240: tensor<1024xf32>, %241: tensor<i64>, %242: tensor<1024xf32>, %243: tensor<1024xf32>, %244: tensor<i64>, %245: tensor<256xf32>, %246: tensor<256xf32>, %247: tensor<i64>, %248: tensor<256xf32>, %249: tensor<256xf32>, %250: tensor<i64>, %251: tensor<1024xf32>, %252: tensor<1024xf32>, %253: tensor<i64>, %254: tensor<256xf32>, %255: tensor<256xf32>, %256: tensor<i64>, %257: tensor<256xf32>, %258: tensor<256xf32>, %259: tensor<i64>, %260: tensor<1024xf32>, %261: tensor<1024xf32>, %262: tensor<i64>, %263: tensor<256xf32>, %264: tensor<256xf32>, %265: tensor<i64>, %266: tensor<256xf32>, %267: tensor<256xf32>, %268: tensor<i64>, %269: tensor<1024xf32>, %270: tensor<1024xf32>, %271: tensor<i64>, %272: tensor<256xf32>, %273: tensor<256xf32>, %274: tensor<i64>, %275: tensor<256xf32>, %276: tensor<256xf32>, %277: tensor<i64>, %278: tensor<1024xf32>, %279: tensor<1024xf32>, %280: tensor<i64>, %281: tensor<256xf32>, %282: tensor<256xf32>, %283: tensor<i64>, %284: tensor<256xf32>, %285: tensor<256xf32>, %286: tensor<i64>, %287: tensor<1024xf32>, %288: tensor<1024xf32>, %289: tensor<i64>, %290: tensor<512xf32>, %291: tensor<512xf32>, %292: tensor<i64>, %293: tensor<512xf32>, %294: tensor<512xf32>, %295: tensor<i64>, %296: tensor<2048xf32>, %297: tensor<2048xf32>, %298: tensor<i64>, %299: tensor<2048xf32>, %300: tensor<2048xf32>, %301: tensor<i64>, %302: tensor<512xf32>, %303: tensor<512xf32>, %304: tensor<i64>, %305: tensor<512xf32>, %306: tensor<512xf32>, %307: tensor<i64>, %308: tensor<2048xf32>, %309: tensor<2048xf32>, %310: tensor<i64>, %311: tensor<512xf32>, %312: tensor<512xf32>, %313: tensor<i64>, %314: tensor<512xf32>, %315: tensor<512xf32>, %316: tensor<i64>, %317: tensor<2048xf32>, %318: tensor<2048xf32>, %319: tensor<i64>, %320: tensor<1x3x224x224xf32>) -> tensor<1x1000xf32> {
    %321 = tensor.empty() : tensor<1000x2048xf32>
    %322 = tensor.empty() : tensor<1000x2048xi8>
    %323 = tensor.empty() : tensor<1000x2048xi8>
    %324 = tensor.empty() : tensor<1000xf32>
    %325 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} 0.000000e+00 : f32
    %326 = tensor.splat %325 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<1x3x230x230xf32>
    %327 = "tensor.insert_slice"(%320, %326) <{static_offsets = array<i64: 0, 0, 3, 3>, static_sizes = array<i64: 1, 3, 224, 224>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : (tensor<1x3x224x224xf32>, tensor<1x3x230x230xf32>) -> tensor<1x3x230x230xf32>
    %328 = tensor.empty() : tensor<3x7x7x1x112x112xf32>
    %329 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%327 : tensor<1x3x230x230xf32>) outs(%328 : tensor<3x7x7x1x112x112xf32>) attrs =  {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} {
    ^bb0(%330: f32, %331: f32):
      linalg.yield %330 : f32
    } -> tensor<3x7x7x1x112x112xf32>
    %332 = tensor.collapse_shape %329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<3x7x7x1x112x112xf32> into tensor<1843968xf32>
    %333 = tensor.expand_shape %332 [[0 : i64, 1 : i64]] output_shape [147, 12544] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<1843968xf32> into tensor<147x12544xf32>
    %334 = tensor.collapse_shape %0 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<64x3x7x7xf32> into tensor<9408xf32>
    %335 = tensor.expand_shape %334 [[0 : i64, 1 : i64]] output_shape [64, 147] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<9408xf32> into tensor<64x147xf32>
    %336 = arith.constant {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} 0.000000e+00 : f32
    %337 = tensor.splat %336 {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<64x12544xf32>
    %338 = linalg.matmul {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} ins(%335, %333 : tensor<64x147xf32>, tensor<147x12544xf32>) outs(%337 : tensor<64x12544xf32>) -> tensor<64x12544xf32>
    %339 = tensor.collapse_shape %338 [[0 : i64, 1 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<64x12544xf32> into tensor<802816xf32>
    %340 = tensor.expand_shape %339 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 112, 112] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<802816xf32> into tensor<64x1x112x112xf32>
    %341 = tensor.collapse_shape %340 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<64x1x112x112xf32> into tensor<802816xf32>
    %342 = tensor.expand_shape %341 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 112, 112] {prov.region_id = "conv_0", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.conv1"} : tensor<802816xf32> into tensor<1x64x112x112xf32>
    %343 = tensor.empty() : tensor<1x64x112x112xf32>
    %344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%342, %161 : tensor<1x64x112x112xf32>, tensor<64xf32>) outs(%343 : tensor<1x64x112x112xf32>) attrs =  {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} {
    ^bb1(%345: f32, %346: f32, %347: f32):
      %348 = arith.subf %345, %346 : f32
      linalg.yield %348 : f32
    } -> tensor<1x64x112x112xf32>
    %349 = arith.constant {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} 1.000000e-05 : f32
    %350 = tensor.splat %349 {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} : tensor<64xf32>
    %351 = tensor.empty() : tensor<64xf32>
    %352 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%162, %350 : tensor<64xf32>, tensor<64xf32>) outs(%351 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} {
    ^bb2(%353: f32, %354: f32, %355: f32):
      %356 = arith.addf %353, %354 : f32
      linalg.yield %356 : f32
    } -> tensor<64xf32>
    %357 = tensor.empty() : tensor<64xf32>
    %358 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%352 : tensor<64xf32>) outs(%357 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} {
    ^bb3(%359: f32, %360: f32):
      %361 = math.rsqrt %359 : f32
      linalg.yield %361 : f32
    } -> tensor<64xf32>
    %362 = tensor.empty() : tensor<1x64x112x112xf32>
    %363 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%344, %358 : tensor<1x64x112x112xf32>, tensor<64xf32>) outs(%362 : tensor<1x64x112x112xf32>) attrs =  {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} {
    ^bb4(%364: f32, %365: f32, %366: f32):
      %367 = arith.mulf %364, %365 : f32
      linalg.yield %367 : f32
    } -> tensor<1x64x112x112xf32>
    %368 = tensor.empty() : tensor<1x64x112x112xf32>
    %369 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%363, %1 : tensor<1x64x112x112xf32>, tensor<64xf32>) outs(%368 : tensor<1x64x112x112xf32>) attrs =  {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} {
    ^bb5(%370: f32, %371: f32, %372: f32):
      %373 = arith.mulf %370, %371 : f32
      linalg.yield %373 : f32
    } -> tensor<1x64x112x112xf32>
    %374 = tensor.empty() : tensor<1x64x112x112xf32>
    %375 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%369, %2 : tensor<1x64x112x112xf32>, tensor<64xf32>) outs(%374 : tensor<1x64x112x112xf32>) attrs =  {prov.region_id = "batch_norm_0", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.bn1"} {
    ^bb6(%376: f32, %377: f32, %378: f32):
      %379 = arith.addf %376, %377 : f32
      linalg.yield %379 : f32
    } -> tensor<1x64x112x112xf32>
    %380 = tensor.empty() : tensor<1x64x112x112xf32>
    %381 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%375 : tensor<1x64x112x112xf32>) outs(%380 : tensor<1x64x112x112xf32>) attrs =  {prov.region_id = "minmax_0", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.relu"} {
    ^bb7(%382: f32, %383: f32):
      %384 = arith.constant 0.000000e+00 : f32
      %385 = arith.maximumf %382, %384 : f32
      linalg.yield %385 : f32
    } -> tensor<1x64x112x112xf32>
    %386 = arith.constant {prov.region_id = "max_pool2d_0", prov.family = "pool", prov._pattern_hint = "max_pool2d", prov.op = "max_pool2d", prov.aten = "aten.max_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.maxpool"} 0xff800000 : f32
    %387 = tensor.splat %386 {prov.region_id = "max_pool2d_0", prov.family = "pool", prov._pattern_hint = "max_pool2d", prov.op = "max_pool2d", prov.aten = "aten.max_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.maxpool"} : tensor<1x64x114x114xf32>
    %388 = "tensor.insert_slice"(%381, %387) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 64, 112, 112>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "max_pool2d_0", prov.family = "layout", prov._pattern_hint = "max_pool2d", prov.op = "slice_scatter", prov.aten = "aten.max_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.maxpool"} : (tensor<1x64x112x112xf32>, tensor<1x64x114x114xf32>) -> tensor<1x64x114x114xf32>
    %389 = tensor.splat %386 {prov.region_id = "max_pool2d_0", prov.family = "pool", prov._pattern_hint = "max_pool2d", prov.op = "max_pool2d", prov.aten = "aten.max_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.maxpool"} : tensor<1x64x56x56xf32>
    %390 = tensor.empty() : tensor<3x3xf32>
    %391 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, ((d2 * 2) + d4), ((d3 * 2) + d5))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%388, %390 : tensor<1x64x114x114xf32>, tensor<3x3xf32>) outs(%389 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "max_pool2d_0", prov.family = "pool", prov._pattern_hint = "max_pool2d", prov.op = "max_pool2d", prov.aten = "aten.max_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.maxpool"} {
    ^bb8(%392: f32, %393: f32, %394: f32):
      %395 = arith.maximumf %392, %394 : f32
      linalg.yield %395 : f32
    } -> tensor<1x64x56x56xf32>
    %396 = tensor.empty() : tensor<64x1x1x1x56x56xf32>
    %397 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%391 : tensor<1x64x56x56xf32>) outs(%396 : tensor<64x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} {
    ^bb9(%398: f32, %399: f32):
      linalg.yield %398 : f32
    } -> tensor<64x1x1x1x56x56xf32>
    %400 = tensor.collapse_shape %397 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<64x1x1x1x56x56xf32> into tensor<200704xf32>
    %401 = tensor.expand_shape %400 [[0 : i64, 1 : i64]] output_shape [64, 3136] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<200704xf32> into tensor<64x3136xf32>
    %402 = tensor.collapse_shape %3 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<64x64x1x1xf32> into tensor<4096xf32>
    %403 = tensor.expand_shape %402 [[0 : i64, 1 : i64]] output_shape [64, 64] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<4096xf32> into tensor<64x64xf32>
    %404 = arith.constant {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} 0.000000e+00 : f32
    %405 = tensor.splat %404 {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<64x3136xf32>
    %406 = linalg.matmul {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} ins(%403, %401 : tensor<64x64xf32>, tensor<64x3136xf32>) outs(%405 : tensor<64x3136xf32>) -> tensor<64x3136xf32>
    %407 = tensor.collapse_shape %406 [[0 : i64, 1 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<64x3136xf32> into tensor<200704xf32>
    %408 = tensor.expand_shape %407 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 56, 56] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<200704xf32> into tensor<64x1x56x56xf32>
    %409 = tensor.collapse_shape %408 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<64x1x56x56xf32> into tensor<200704xf32>
    %410 = tensor.expand_shape %409 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 56, 56] {prov.region_id = "conv_1", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv1"} : tensor<200704xf32> into tensor<1x64x56x56xf32>
    %411 = tensor.empty() : tensor<1x64x56x56xf32>
    %412 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%410, %164 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%411 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} {
    ^bb10(%413: f32, %414: f32, %415: f32):
      %416 = arith.subf %413, %414 : f32
      linalg.yield %416 : f32
    } -> tensor<1x64x56x56xf32>
    %417 = arith.constant {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} 1.000000e-05 : f32
    %418 = tensor.splat %417 {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} : tensor<64xf32>
    %419 = tensor.empty() : tensor<64xf32>
    %420 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%165, %418 : tensor<64xf32>, tensor<64xf32>) outs(%419 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} {
    ^bb11(%421: f32, %422: f32, %423: f32):
      %424 = arith.addf %421, %422 : f32
      linalg.yield %424 : f32
    } -> tensor<64xf32>
    %425 = tensor.empty() : tensor<64xf32>
    %426 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%420 : tensor<64xf32>) outs(%425 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} {
    ^bb12(%427: f32, %428: f32):
      %429 = math.rsqrt %427 : f32
      linalg.yield %429 : f32
    } -> tensor<64xf32>
    %430 = tensor.empty() : tensor<1x64x56x56xf32>
    %431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%412, %426 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%430 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} {
    ^bb13(%432: f32, %433: f32, %434: f32):
      %435 = arith.mulf %432, %433 : f32
      linalg.yield %435 : f32
    } -> tensor<1x64x56x56xf32>
    %436 = tensor.empty() : tensor<1x64x56x56xf32>
    %437 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%431, %4 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%436 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} {
    ^bb14(%438: f32, %439: f32, %440: f32):
      %441 = arith.mulf %438, %439 : f32
      linalg.yield %441 : f32
    } -> tensor<1x64x56x56xf32>
    %442 = tensor.empty() : tensor<1x64x56x56xf32>
    %443 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%437, %5 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%442 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_1", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn1"} {
    ^bb15(%444: f32, %445: f32, %446: f32):
      %447 = arith.addf %444, %445 : f32
      linalg.yield %447 : f32
    } -> tensor<1x64x56x56xf32>
    %448 = tensor.empty() : tensor<1x64x56x56xf32>
    %449 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%443 : tensor<1x64x56x56xf32>) outs(%448 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "minmax_1", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.relu"} {
    ^bb16(%450: f32, %451: f32):
      %452 = arith.constant 0.000000e+00 : f32
      %453 = arith.maximumf %450, %452 : f32
      linalg.yield %453 : f32
    } -> tensor<1x64x56x56xf32>
    %454 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} 0.000000e+00 : f32
    %455 = tensor.splat %454 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<1x64x58x58xf32>
    %456 = "tensor.insert_slice"(%449, %455) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 64, 56, 56>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : (tensor<1x64x56x56xf32>, tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32>
    %457 = tensor.empty() : tensor<64x3x3x1x56x56xf32>
    %458 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%456 : tensor<1x64x58x58xf32>) outs(%457 : tensor<64x3x3x1x56x56xf32>) attrs =  {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} {
    ^bb17(%459: f32, %460: f32):
      linalg.yield %459 : f32
    } -> tensor<64x3x3x1x56x56xf32>
    %461 = tensor.collapse_shape %458 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<64x3x3x1x56x56xf32> into tensor<1806336xf32>
    %462 = tensor.expand_shape %461 [[0 : i64, 1 : i64]] output_shape [576, 3136] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<1806336xf32> into tensor<576x3136xf32>
    %463 = tensor.collapse_shape %6 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<64x64x3x3xf32> into tensor<36864xf32>
    %464 = tensor.expand_shape %463 [[0 : i64, 1 : i64]] output_shape [64, 576] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<36864xf32> into tensor<64x576xf32>
    %465 = arith.constant {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} 0.000000e+00 : f32
    %466 = tensor.splat %465 {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<64x3136xf32>
    %467 = linalg.matmul {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} ins(%464, %462 : tensor<64x576xf32>, tensor<576x3136xf32>) outs(%466 : tensor<64x3136xf32>) -> tensor<64x3136xf32>
    %468 = tensor.collapse_shape %467 [[0 : i64, 1 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<64x3136xf32> into tensor<200704xf32>
    %469 = tensor.expand_shape %468 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 56, 56] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<200704xf32> into tensor<64x1x56x56xf32>
    %470 = tensor.collapse_shape %469 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<64x1x56x56xf32> into tensor<200704xf32>
    %471 = tensor.expand_shape %470 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 56, 56] {prov.region_id = "conv_2", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv2"} : tensor<200704xf32> into tensor<1x64x56x56xf32>
    %472 = tensor.empty() : tensor<1x64x56x56xf32>
    %473 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%471, %167 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%472 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} {
    ^bb18(%474: f32, %475: f32, %476: f32):
      %477 = arith.subf %474, %475 : f32
      linalg.yield %477 : f32
    } -> tensor<1x64x56x56xf32>
    %478 = arith.constant {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} 1.000000e-05 : f32
    %479 = tensor.splat %478 {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} : tensor<64xf32>
    %480 = tensor.empty() : tensor<64xf32>
    %481 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%168, %479 : tensor<64xf32>, tensor<64xf32>) outs(%480 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} {
    ^bb19(%482: f32, %483: f32, %484: f32):
      %485 = arith.addf %482, %483 : f32
      linalg.yield %485 : f32
    } -> tensor<64xf32>
    %486 = tensor.empty() : tensor<64xf32>
    %487 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%481 : tensor<64xf32>) outs(%486 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} {
    ^bb20(%488: f32, %489: f32):
      %490 = math.rsqrt %488 : f32
      linalg.yield %490 : f32
    } -> tensor<64xf32>
    %491 = tensor.empty() : tensor<1x64x56x56xf32>
    %492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%473, %487 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%491 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} {
    ^bb21(%493: f32, %494: f32, %495: f32):
      %496 = arith.mulf %493, %494 : f32
      linalg.yield %496 : f32
    } -> tensor<1x64x56x56xf32>
    %497 = tensor.empty() : tensor<1x64x56x56xf32>
    %498 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%492, %7 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%497 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} {
    ^bb22(%499: f32, %500: f32, %501: f32):
      %502 = arith.mulf %499, %500 : f32
      linalg.yield %502 : f32
    } -> tensor<1x64x56x56xf32>
    %503 = tensor.empty() : tensor<1x64x56x56xf32>
    %504 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%498, %8 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%503 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_2", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn2"} {
    ^bb23(%505: f32, %506: f32, %507: f32):
      %508 = arith.addf %505, %506 : f32
      linalg.yield %508 : f32
    } -> tensor<1x64x56x56xf32>
    %509 = tensor.empty() : tensor<1x64x56x56xf32>
    %510 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%504 : tensor<1x64x56x56xf32>) outs(%509 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "minmax_2", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.relu"} {
    ^bb24(%511: f32, %512: f32):
      %513 = arith.constant 0.000000e+00 : f32
      %514 = arith.maximumf %511, %513 : f32
      linalg.yield %514 : f32
    } -> tensor<1x64x56x56xf32>
    %515 = tensor.empty() : tensor<64x1x1x1x56x56xf32>
    %516 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%510 : tensor<1x64x56x56xf32>) outs(%515 : tensor<64x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} {
    ^bb25(%517: f32, %518: f32):
      linalg.yield %517 : f32
    } -> tensor<64x1x1x1x56x56xf32>
    %519 = tensor.collapse_shape %516 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<64x1x1x1x56x56xf32> into tensor<200704xf32>
    %520 = tensor.expand_shape %519 [[0 : i64, 1 : i64]] output_shape [64, 3136] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<200704xf32> into tensor<64x3136xf32>
    %521 = tensor.collapse_shape %9 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<256x64x1x1xf32> into tensor<16384xf32>
    %522 = tensor.expand_shape %521 [[0 : i64, 1 : i64]] output_shape [256, 64] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<16384xf32> into tensor<256x64xf32>
    %523 = arith.constant {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} 0.000000e+00 : f32
    %524 = tensor.splat %523 {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<256x3136xf32>
    %525 = linalg.matmul {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} ins(%522, %520 : tensor<256x64xf32>, tensor<64x3136xf32>) outs(%524 : tensor<256x3136xf32>) -> tensor<256x3136xf32>
    %526 = tensor.collapse_shape %525 [[0 : i64, 1 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<256x3136xf32> into tensor<802816xf32>
    %527 = tensor.expand_shape %526 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 56, 56] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<802816xf32> into tensor<256x1x56x56xf32>
    %528 = tensor.collapse_shape %527 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<256x1x56x56xf32> into tensor<802816xf32>
    %529 = tensor.expand_shape %528 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 56, 56] {prov.region_id = "conv_3", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.conv3"} : tensor<802816xf32> into tensor<1x256x56x56xf32>
    %530 = tensor.empty() : tensor<1x256x56x56xf32>
    %531 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%529, %170 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%530 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} {
    ^bb26(%532: f32, %533: f32, %534: f32):
      %535 = arith.subf %532, %533 : f32
      linalg.yield %535 : f32
    } -> tensor<1x256x56x56xf32>
    %536 = arith.constant {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} 1.000000e-05 : f32
    %537 = tensor.splat %536 {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} : tensor<256xf32>
    %538 = tensor.empty() : tensor<256xf32>
    %539 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%171, %537 : tensor<256xf32>, tensor<256xf32>) outs(%538 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} {
    ^bb27(%540: f32, %541: f32, %542: f32):
      %543 = arith.addf %540, %541 : f32
      linalg.yield %543 : f32
    } -> tensor<256xf32>
    %544 = tensor.empty() : tensor<256xf32>
    %545 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%539 : tensor<256xf32>) outs(%544 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} {
    ^bb28(%546: f32, %547: f32):
      %548 = math.rsqrt %546 : f32
      linalg.yield %548 : f32
    } -> tensor<256xf32>
    %549 = tensor.empty() : tensor<1x256x56x56xf32>
    %550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%531, %545 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%549 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} {
    ^bb29(%551: f32, %552: f32, %553: f32):
      %554 = arith.mulf %551, %552 : f32
      linalg.yield %554 : f32
    } -> tensor<1x256x56x56xf32>
    %555 = tensor.empty() : tensor<1x256x56x56xf32>
    %556 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%550, %10 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%555 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} {
    ^bb30(%557: f32, %558: f32, %559: f32):
      %560 = arith.mulf %557, %558 : f32
      linalg.yield %560 : f32
    } -> tensor<1x256x56x56xf32>
    %561 = tensor.empty() : tensor<1x256x56x56xf32>
    %562 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%556, %11 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%561 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_3", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.bn3"} {
    ^bb31(%563: f32, %564: f32, %565: f32):
      %566 = arith.addf %563, %564 : f32
      linalg.yield %566 : f32
    } -> tensor<1x256x56x56xf32>
    %567 = tensor.empty() : tensor<64x1x1x1x56x56xf32>
    %568 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%391 : tensor<1x64x56x56xf32>) outs(%567 : tensor<64x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} {
    ^bb32(%569: f32, %570: f32):
      linalg.yield %569 : f32
    } -> tensor<64x1x1x1x56x56xf32>
    %571 = tensor.collapse_shape %568 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<64x1x1x1x56x56xf32> into tensor<200704xf32>
    %572 = tensor.expand_shape %571 [[0 : i64, 1 : i64]] output_shape [64, 3136] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<200704xf32> into tensor<64x3136xf32>
    %573 = tensor.collapse_shape %12 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<256x64x1x1xf32> into tensor<16384xf32>
    %574 = tensor.expand_shape %573 [[0 : i64, 1 : i64]] output_shape [256, 64] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<16384xf32> into tensor<256x64xf32>
    %575 = arith.constant {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} 0.000000e+00 : f32
    %576 = tensor.splat %575 {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<256x3136xf32>
    %577 = linalg.matmul {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} ins(%574, %572 : tensor<256x64xf32>, tensor<64x3136xf32>) outs(%576 : tensor<256x3136xf32>) -> tensor<256x3136xf32>
    %578 = tensor.collapse_shape %577 [[0 : i64, 1 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<256x3136xf32> into tensor<802816xf32>
    %579 = tensor.expand_shape %578 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 56, 56] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<802816xf32> into tensor<256x1x56x56xf32>
    %580 = tensor.collapse_shape %579 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<256x1x56x56xf32> into tensor<802816xf32>
    %581 = tensor.expand_shape %580 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 56, 56] {prov.region_id = "conv_4", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.0"} : tensor<802816xf32> into tensor<1x256x56x56xf32>
    %582 = tensor.empty() : tensor<1x256x56x56xf32>
    %583 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%581, %173 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%582 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} {
    ^bb33(%584: f32, %585: f32, %586: f32):
      %587 = arith.subf %584, %585 : f32
      linalg.yield %587 : f32
    } -> tensor<1x256x56x56xf32>
    %588 = arith.constant {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} 1.000000e-05 : f32
    %589 = tensor.splat %588 {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} : tensor<256xf32>
    %590 = tensor.empty() : tensor<256xf32>
    %591 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%174, %589 : tensor<256xf32>, tensor<256xf32>) outs(%590 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} {
    ^bb34(%592: f32, %593: f32, %594: f32):
      %595 = arith.addf %592, %593 : f32
      linalg.yield %595 : f32
    } -> tensor<256xf32>
    %596 = tensor.empty() : tensor<256xf32>
    %597 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%591 : tensor<256xf32>) outs(%596 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} {
    ^bb35(%598: f32, %599: f32):
      %600 = math.rsqrt %598 : f32
      linalg.yield %600 : f32
    } -> tensor<256xf32>
    %601 = tensor.empty() : tensor<1x256x56x56xf32>
    %602 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%583, %597 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%601 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} {
    ^bb36(%603: f32, %604: f32, %605: f32):
      %606 = arith.mulf %603, %604 : f32
      linalg.yield %606 : f32
    } -> tensor<1x256x56x56xf32>
    %607 = tensor.empty() : tensor<1x256x56x56xf32>
    %608 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%602, %13 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%607 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} {
    ^bb37(%609: f32, %610: f32, %611: f32):
      %612 = arith.mulf %609, %610 : f32
      linalg.yield %612 : f32
    } -> tensor<1x256x56x56xf32>
    %613 = tensor.empty() : tensor<1x256x56x56xf32>
    %614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%608, %14 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%613 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_4", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.downsample.1"} {
    ^bb38(%615: f32, %616: f32, %617: f32):
      %618 = arith.addf %615, %616 : f32
      linalg.yield %618 : f32
    } -> tensor<1x256x56x56xf32>
    %619 = tensor.empty() : tensor<1x256x56x56xf32>
    %620 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%562, %614 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%619 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "add_0", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0"} {
    ^bb39(%621: f32, %622: f32, %623: f32):
      %624 = arith.addf %621, %622 : f32
      linalg.yield %624 : f32
    } -> tensor<1x256x56x56xf32>
    %625 = tensor.empty() : tensor<1x256x56x56xf32>
    %626 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%620 : tensor<1x256x56x56xf32>) outs(%625 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "minmax_3", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.0.relu"} {
    ^bb40(%627: f32, %628: f32):
      %629 = arith.constant 0.000000e+00 : f32
      %630 = arith.maximumf %627, %629 : f32
      linalg.yield %630 : f32
    } -> tensor<1x256x56x56xf32>
    %631 = tensor.empty() : tensor<256x1x1x1x56x56xf32>
    %632 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%626 : tensor<1x256x56x56xf32>) outs(%631 : tensor<256x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} {
    ^bb41(%633: f32, %634: f32):
      linalg.yield %633 : f32
    } -> tensor<256x1x1x1x56x56xf32>
    %635 = tensor.collapse_shape %632 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<256x1x1x1x56x56xf32> into tensor<802816xf32>
    %636 = tensor.expand_shape %635 [[0 : i64, 1 : i64]] output_shape [256, 3136] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<802816xf32> into tensor<256x3136xf32>
    %637 = tensor.collapse_shape %15 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<64x256x1x1xf32> into tensor<16384xf32>
    %638 = tensor.expand_shape %637 [[0 : i64, 1 : i64]] output_shape [64, 256] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<16384xf32> into tensor<64x256xf32>
    %639 = arith.constant {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} 0.000000e+00 : f32
    %640 = tensor.splat %639 {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<64x3136xf32>
    %641 = linalg.matmul {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} ins(%638, %636 : tensor<64x256xf32>, tensor<256x3136xf32>) outs(%640 : tensor<64x3136xf32>) -> tensor<64x3136xf32>
    %642 = tensor.collapse_shape %641 [[0 : i64, 1 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<64x3136xf32> into tensor<200704xf32>
    %643 = tensor.expand_shape %642 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 56, 56] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<200704xf32> into tensor<64x1x56x56xf32>
    %644 = tensor.collapse_shape %643 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<64x1x56x56xf32> into tensor<200704xf32>
    %645 = tensor.expand_shape %644 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 56, 56] {prov.region_id = "conv_5", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv1"} : tensor<200704xf32> into tensor<1x64x56x56xf32>
    %646 = tensor.empty() : tensor<1x64x56x56xf32>
    %647 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%645, %176 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%646 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} {
    ^bb42(%648: f32, %649: f32, %650: f32):
      %651 = arith.subf %648, %649 : f32
      linalg.yield %651 : f32
    } -> tensor<1x64x56x56xf32>
    %652 = arith.constant {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} 1.000000e-05 : f32
    %653 = tensor.splat %652 {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} : tensor<64xf32>
    %654 = tensor.empty() : tensor<64xf32>
    %655 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%177, %653 : tensor<64xf32>, tensor<64xf32>) outs(%654 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} {
    ^bb43(%656: f32, %657: f32, %658: f32):
      %659 = arith.addf %656, %657 : f32
      linalg.yield %659 : f32
    } -> tensor<64xf32>
    %660 = tensor.empty() : tensor<64xf32>
    %661 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%655 : tensor<64xf32>) outs(%660 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} {
    ^bb44(%662: f32, %663: f32):
      %664 = math.rsqrt %662 : f32
      linalg.yield %664 : f32
    } -> tensor<64xf32>
    %665 = tensor.empty() : tensor<1x64x56x56xf32>
    %666 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%647, %661 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%665 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} {
    ^bb45(%667: f32, %668: f32, %669: f32):
      %670 = arith.mulf %667, %668 : f32
      linalg.yield %670 : f32
    } -> tensor<1x64x56x56xf32>
    %671 = tensor.empty() : tensor<1x64x56x56xf32>
    %672 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%666, %16 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%671 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} {
    ^bb46(%673: f32, %674: f32, %675: f32):
      %676 = arith.mulf %673, %674 : f32
      linalg.yield %676 : f32
    } -> tensor<1x64x56x56xf32>
    %677 = tensor.empty() : tensor<1x64x56x56xf32>
    %678 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%672, %17 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%677 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_5", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn1"} {
    ^bb47(%679: f32, %680: f32, %681: f32):
      %682 = arith.addf %679, %680 : f32
      linalg.yield %682 : f32
    } -> tensor<1x64x56x56xf32>
    %683 = tensor.empty() : tensor<1x64x56x56xf32>
    %684 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%678 : tensor<1x64x56x56xf32>) outs(%683 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "minmax_4", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.relu"} {
    ^bb48(%685: f32, %686: f32):
      %687 = arith.constant 0.000000e+00 : f32
      %688 = arith.maximumf %685, %687 : f32
      linalg.yield %688 : f32
    } -> tensor<1x64x56x56xf32>
    %689 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} 0.000000e+00 : f32
    %690 = tensor.splat %689 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<1x64x58x58xf32>
    %691 = "tensor.insert_slice"(%684, %690) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 64, 56, 56>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : (tensor<1x64x56x56xf32>, tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32>
    %692 = tensor.empty() : tensor<64x3x3x1x56x56xf32>
    %693 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%691 : tensor<1x64x58x58xf32>) outs(%692 : tensor<64x3x3x1x56x56xf32>) attrs =  {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} {
    ^bb49(%694: f32, %695: f32):
      linalg.yield %694 : f32
    } -> tensor<64x3x3x1x56x56xf32>
    %696 = tensor.collapse_shape %693 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<64x3x3x1x56x56xf32> into tensor<1806336xf32>
    %697 = tensor.expand_shape %696 [[0 : i64, 1 : i64]] output_shape [576, 3136] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<1806336xf32> into tensor<576x3136xf32>
    %698 = tensor.collapse_shape %18 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<64x64x3x3xf32> into tensor<36864xf32>
    %699 = tensor.expand_shape %698 [[0 : i64, 1 : i64]] output_shape [64, 576] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<36864xf32> into tensor<64x576xf32>
    %700 = arith.constant {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} 0.000000e+00 : f32
    %701 = tensor.splat %700 {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<64x3136xf32>
    %702 = linalg.matmul {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} ins(%699, %697 : tensor<64x576xf32>, tensor<576x3136xf32>) outs(%701 : tensor<64x3136xf32>) -> tensor<64x3136xf32>
    %703 = tensor.collapse_shape %702 [[0 : i64, 1 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<64x3136xf32> into tensor<200704xf32>
    %704 = tensor.expand_shape %703 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 56, 56] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<200704xf32> into tensor<64x1x56x56xf32>
    %705 = tensor.collapse_shape %704 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<64x1x56x56xf32> into tensor<200704xf32>
    %706 = tensor.expand_shape %705 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 56, 56] {prov.region_id = "conv_6", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv2"} : tensor<200704xf32> into tensor<1x64x56x56xf32>
    %707 = tensor.empty() : tensor<1x64x56x56xf32>
    %708 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%706, %179 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%707 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} {
    ^bb50(%709: f32, %710: f32, %711: f32):
      %712 = arith.subf %709, %710 : f32
      linalg.yield %712 : f32
    } -> tensor<1x64x56x56xf32>
    %713 = arith.constant {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} 1.000000e-05 : f32
    %714 = tensor.splat %713 {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} : tensor<64xf32>
    %715 = tensor.empty() : tensor<64xf32>
    %716 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%180, %714 : tensor<64xf32>, tensor<64xf32>) outs(%715 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} {
    ^bb51(%717: f32, %718: f32, %719: f32):
      %720 = arith.addf %717, %718 : f32
      linalg.yield %720 : f32
    } -> tensor<64xf32>
    %721 = tensor.empty() : tensor<64xf32>
    %722 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%716 : tensor<64xf32>) outs(%721 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} {
    ^bb52(%723: f32, %724: f32):
      %725 = math.rsqrt %723 : f32
      linalg.yield %725 : f32
    } -> tensor<64xf32>
    %726 = tensor.empty() : tensor<1x64x56x56xf32>
    %727 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%708, %722 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%726 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} {
    ^bb53(%728: f32, %729: f32, %730: f32):
      %731 = arith.mulf %728, %729 : f32
      linalg.yield %731 : f32
    } -> tensor<1x64x56x56xf32>
    %732 = tensor.empty() : tensor<1x64x56x56xf32>
    %733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%727, %19 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%732 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} {
    ^bb54(%734: f32, %735: f32, %736: f32):
      %737 = arith.mulf %734, %735 : f32
      linalg.yield %737 : f32
    } -> tensor<1x64x56x56xf32>
    %738 = tensor.empty() : tensor<1x64x56x56xf32>
    %739 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%733, %20 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%738 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_6", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn2"} {
    ^bb55(%740: f32, %741: f32, %742: f32):
      %743 = arith.addf %740, %741 : f32
      linalg.yield %743 : f32
    } -> tensor<1x64x56x56xf32>
    %744 = tensor.empty() : tensor<1x64x56x56xf32>
    %745 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%739 : tensor<1x64x56x56xf32>) outs(%744 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "minmax_5", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.relu"} {
    ^bb56(%746: f32, %747: f32):
      %748 = arith.constant 0.000000e+00 : f32
      %749 = arith.maximumf %746, %748 : f32
      linalg.yield %749 : f32
    } -> tensor<1x64x56x56xf32>
    %750 = tensor.empty() : tensor<64x1x1x1x56x56xf32>
    %751 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%745 : tensor<1x64x56x56xf32>) outs(%750 : tensor<64x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} {
    ^bb57(%752: f32, %753: f32):
      linalg.yield %752 : f32
    } -> tensor<64x1x1x1x56x56xf32>
    %754 = tensor.collapse_shape %751 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<64x1x1x1x56x56xf32> into tensor<200704xf32>
    %755 = tensor.expand_shape %754 [[0 : i64, 1 : i64]] output_shape [64, 3136] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<200704xf32> into tensor<64x3136xf32>
    %756 = tensor.collapse_shape %21 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<256x64x1x1xf32> into tensor<16384xf32>
    %757 = tensor.expand_shape %756 [[0 : i64, 1 : i64]] output_shape [256, 64] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<16384xf32> into tensor<256x64xf32>
    %758 = arith.constant {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} 0.000000e+00 : f32
    %759 = tensor.splat %758 {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<256x3136xf32>
    %760 = linalg.matmul {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} ins(%757, %755 : tensor<256x64xf32>, tensor<64x3136xf32>) outs(%759 : tensor<256x3136xf32>) -> tensor<256x3136xf32>
    %761 = tensor.collapse_shape %760 [[0 : i64, 1 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<256x3136xf32> into tensor<802816xf32>
    %762 = tensor.expand_shape %761 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 56, 56] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<802816xf32> into tensor<256x1x56x56xf32>
    %763 = tensor.collapse_shape %762 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<256x1x56x56xf32> into tensor<802816xf32>
    %764 = tensor.expand_shape %763 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 56, 56] {prov.region_id = "conv_7", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.conv3"} : tensor<802816xf32> into tensor<1x256x56x56xf32>
    %765 = tensor.empty() : tensor<1x256x56x56xf32>
    %766 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%764, %182 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%765 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} {
    ^bb58(%767: f32, %768: f32, %769: f32):
      %770 = arith.subf %767, %768 : f32
      linalg.yield %770 : f32
    } -> tensor<1x256x56x56xf32>
    %771 = arith.constant {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} 1.000000e-05 : f32
    %772 = tensor.splat %771 {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} : tensor<256xf32>
    %773 = tensor.empty() : tensor<256xf32>
    %774 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%183, %772 : tensor<256xf32>, tensor<256xf32>) outs(%773 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} {
    ^bb59(%775: f32, %776: f32, %777: f32):
      %778 = arith.addf %775, %776 : f32
      linalg.yield %778 : f32
    } -> tensor<256xf32>
    %779 = tensor.empty() : tensor<256xf32>
    %780 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%774 : tensor<256xf32>) outs(%779 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} {
    ^bb60(%781: f32, %782: f32):
      %783 = math.rsqrt %781 : f32
      linalg.yield %783 : f32
    } -> tensor<256xf32>
    %784 = tensor.empty() : tensor<1x256x56x56xf32>
    %785 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%766, %780 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%784 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} {
    ^bb61(%786: f32, %787: f32, %788: f32):
      %789 = arith.mulf %786, %787 : f32
      linalg.yield %789 : f32
    } -> tensor<1x256x56x56xf32>
    %790 = tensor.empty() : tensor<1x256x56x56xf32>
    %791 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%785, %22 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%790 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} {
    ^bb62(%792: f32, %793: f32, %794: f32):
      %795 = arith.mulf %792, %793 : f32
      linalg.yield %795 : f32
    } -> tensor<1x256x56x56xf32>
    %796 = tensor.empty() : tensor<1x256x56x56xf32>
    %797 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%791, %23 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%796 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_7", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.bn3"} {
    ^bb63(%798: f32, %799: f32, %800: f32):
      %801 = arith.addf %798, %799 : f32
      linalg.yield %801 : f32
    } -> tensor<1x256x56x56xf32>
    %802 = tensor.empty() : tensor<1x256x56x56xf32>
    %803 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%797, %626 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%802 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "add_1", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1"} {
    ^bb64(%804: f32, %805: f32, %806: f32):
      %807 = arith.addf %804, %805 : f32
      linalg.yield %807 : f32
    } -> tensor<1x256x56x56xf32>
    %808 = tensor.empty() : tensor<1x256x56x56xf32>
    %809 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%803 : tensor<1x256x56x56xf32>) outs(%808 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "minmax_6", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.1.relu"} {
    ^bb65(%810: f32, %811: f32):
      %812 = arith.constant 0.000000e+00 : f32
      %813 = arith.maximumf %810, %812 : f32
      linalg.yield %813 : f32
    } -> tensor<1x256x56x56xf32>
    %814 = tensor.empty() : tensor<256x1x1x1x56x56xf32>
    %815 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%809 : tensor<1x256x56x56xf32>) outs(%814 : tensor<256x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} {
    ^bb66(%816: f32, %817: f32):
      linalg.yield %816 : f32
    } -> tensor<256x1x1x1x56x56xf32>
    %818 = tensor.collapse_shape %815 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<256x1x1x1x56x56xf32> into tensor<802816xf32>
    %819 = tensor.expand_shape %818 [[0 : i64, 1 : i64]] output_shape [256, 3136] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<802816xf32> into tensor<256x3136xf32>
    %820 = tensor.collapse_shape %24 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<64x256x1x1xf32> into tensor<16384xf32>
    %821 = tensor.expand_shape %820 [[0 : i64, 1 : i64]] output_shape [64, 256] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<16384xf32> into tensor<64x256xf32>
    %822 = arith.constant {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} 0.000000e+00 : f32
    %823 = tensor.splat %822 {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<64x3136xf32>
    %824 = linalg.matmul {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} ins(%821, %819 : tensor<64x256xf32>, tensor<256x3136xf32>) outs(%823 : tensor<64x3136xf32>) -> tensor<64x3136xf32>
    %825 = tensor.collapse_shape %824 [[0 : i64, 1 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<64x3136xf32> into tensor<200704xf32>
    %826 = tensor.expand_shape %825 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 56, 56] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<200704xf32> into tensor<64x1x56x56xf32>
    %827 = tensor.collapse_shape %826 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<64x1x56x56xf32> into tensor<200704xf32>
    %828 = tensor.expand_shape %827 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 56, 56] {prov.region_id = "conv_8", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv1"} : tensor<200704xf32> into tensor<1x64x56x56xf32>
    %829 = tensor.empty() : tensor<1x64x56x56xf32>
    %830 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%828, %185 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%829 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} {
    ^bb67(%831: f32, %832: f32, %833: f32):
      %834 = arith.subf %831, %832 : f32
      linalg.yield %834 : f32
    } -> tensor<1x64x56x56xf32>
    %835 = arith.constant {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} 1.000000e-05 : f32
    %836 = tensor.splat %835 {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} : tensor<64xf32>
    %837 = tensor.empty() : tensor<64xf32>
    %838 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%186, %836 : tensor<64xf32>, tensor<64xf32>) outs(%837 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} {
    ^bb68(%839: f32, %840: f32, %841: f32):
      %842 = arith.addf %839, %840 : f32
      linalg.yield %842 : f32
    } -> tensor<64xf32>
    %843 = tensor.empty() : tensor<64xf32>
    %844 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%838 : tensor<64xf32>) outs(%843 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} {
    ^bb69(%845: f32, %846: f32):
      %847 = math.rsqrt %845 : f32
      linalg.yield %847 : f32
    } -> tensor<64xf32>
    %848 = tensor.empty() : tensor<1x64x56x56xf32>
    %849 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%830, %844 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%848 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} {
    ^bb70(%850: f32, %851: f32, %852: f32):
      %853 = arith.mulf %850, %851 : f32
      linalg.yield %853 : f32
    } -> tensor<1x64x56x56xf32>
    %854 = tensor.empty() : tensor<1x64x56x56xf32>
    %855 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%849, %25 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%854 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} {
    ^bb71(%856: f32, %857: f32, %858: f32):
      %859 = arith.mulf %856, %857 : f32
      linalg.yield %859 : f32
    } -> tensor<1x64x56x56xf32>
    %860 = tensor.empty() : tensor<1x64x56x56xf32>
    %861 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%855, %26 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%860 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_8", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn1"} {
    ^bb72(%862: f32, %863: f32, %864: f32):
      %865 = arith.addf %862, %863 : f32
      linalg.yield %865 : f32
    } -> tensor<1x64x56x56xf32>
    %866 = tensor.empty() : tensor<1x64x56x56xf32>
    %867 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%861 : tensor<1x64x56x56xf32>) outs(%866 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "minmax_7", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.relu"} {
    ^bb73(%868: f32, %869: f32):
      %870 = arith.constant 0.000000e+00 : f32
      %871 = arith.maximumf %868, %870 : f32
      linalg.yield %871 : f32
    } -> tensor<1x64x56x56xf32>
    %872 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} 0.000000e+00 : f32
    %873 = tensor.splat %872 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<1x64x58x58xf32>
    %874 = "tensor.insert_slice"(%867, %873) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 64, 56, 56>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : (tensor<1x64x56x56xf32>, tensor<1x64x58x58xf32>) -> tensor<1x64x58x58xf32>
    %875 = tensor.empty() : tensor<64x3x3x1x56x56xf32>
    %876 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%874 : tensor<1x64x58x58xf32>) outs(%875 : tensor<64x3x3x1x56x56xf32>) attrs =  {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} {
    ^bb74(%877: f32, %878: f32):
      linalg.yield %877 : f32
    } -> tensor<64x3x3x1x56x56xf32>
    %879 = tensor.collapse_shape %876 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<64x3x3x1x56x56xf32> into tensor<1806336xf32>
    %880 = tensor.expand_shape %879 [[0 : i64, 1 : i64]] output_shape [576, 3136] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<1806336xf32> into tensor<576x3136xf32>
    %881 = tensor.collapse_shape %27 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<64x64x3x3xf32> into tensor<36864xf32>
    %882 = tensor.expand_shape %881 [[0 : i64, 1 : i64]] output_shape [64, 576] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<36864xf32> into tensor<64x576xf32>
    %883 = arith.constant {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} 0.000000e+00 : f32
    %884 = tensor.splat %883 {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<64x3136xf32>
    %885 = linalg.matmul {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} ins(%882, %880 : tensor<64x576xf32>, tensor<576x3136xf32>) outs(%884 : tensor<64x3136xf32>) -> tensor<64x3136xf32>
    %886 = tensor.collapse_shape %885 [[0 : i64, 1 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<64x3136xf32> into tensor<200704xf32>
    %887 = tensor.expand_shape %886 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [64, 1, 56, 56] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<200704xf32> into tensor<64x1x56x56xf32>
    %888 = tensor.collapse_shape %887 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<64x1x56x56xf32> into tensor<200704xf32>
    %889 = tensor.expand_shape %888 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 64, 56, 56] {prov.region_id = "conv_9", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv2"} : tensor<200704xf32> into tensor<1x64x56x56xf32>
    %890 = tensor.empty() : tensor<1x64x56x56xf32>
    %891 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%889, %188 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%890 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} {
    ^bb75(%892: f32, %893: f32, %894: f32):
      %895 = arith.subf %892, %893 : f32
      linalg.yield %895 : f32
    } -> tensor<1x64x56x56xf32>
    %896 = arith.constant {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} 1.000000e-05 : f32
    %897 = tensor.splat %896 {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} : tensor<64xf32>
    %898 = tensor.empty() : tensor<64xf32>
    %899 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%189, %897 : tensor<64xf32>, tensor<64xf32>) outs(%898 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} {
    ^bb76(%900: f32, %901: f32, %902: f32):
      %903 = arith.addf %900, %901 : f32
      linalg.yield %903 : f32
    } -> tensor<64xf32>
    %904 = tensor.empty() : tensor<64xf32>
    %905 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%899 : tensor<64xf32>) outs(%904 : tensor<64xf32>) attrs =  {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} {
    ^bb77(%906: f32, %907: f32):
      %908 = math.rsqrt %906 : f32
      linalg.yield %908 : f32
    } -> tensor<64xf32>
    %909 = tensor.empty() : tensor<1x64x56x56xf32>
    %910 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%891, %905 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%909 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} {
    ^bb78(%911: f32, %912: f32, %913: f32):
      %914 = arith.mulf %911, %912 : f32
      linalg.yield %914 : f32
    } -> tensor<1x64x56x56xf32>
    %915 = tensor.empty() : tensor<1x64x56x56xf32>
    %916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%910, %28 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%915 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} {
    ^bb79(%917: f32, %918: f32, %919: f32):
      %920 = arith.mulf %917, %918 : f32
      linalg.yield %920 : f32
    } -> tensor<1x64x56x56xf32>
    %921 = tensor.empty() : tensor<1x64x56x56xf32>
    %922 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%916, %29 : tensor<1x64x56x56xf32>, tensor<64xf32>) outs(%921 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "batch_norm_9", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn2"} {
    ^bb80(%923: f32, %924: f32, %925: f32):
      %926 = arith.addf %923, %924 : f32
      linalg.yield %926 : f32
    } -> tensor<1x64x56x56xf32>
    %927 = tensor.empty() : tensor<1x64x56x56xf32>
    %928 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%922 : tensor<1x64x56x56xf32>) outs(%927 : tensor<1x64x56x56xf32>) attrs =  {prov.region_id = "minmax_8", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.relu"} {
    ^bb81(%929: f32, %930: f32):
      %931 = arith.constant 0.000000e+00 : f32
      %932 = arith.maximumf %929, %931 : f32
      linalg.yield %932 : f32
    } -> tensor<1x64x56x56xf32>
    %933 = tensor.empty() : tensor<64x1x1x1x56x56xf32>
    %934 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%928 : tensor<1x64x56x56xf32>) outs(%933 : tensor<64x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} {
    ^bb82(%935: f32, %936: f32):
      linalg.yield %935 : f32
    } -> tensor<64x1x1x1x56x56xf32>
    %937 = tensor.collapse_shape %934 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<64x1x1x1x56x56xf32> into tensor<200704xf32>
    %938 = tensor.expand_shape %937 [[0 : i64, 1 : i64]] output_shape [64, 3136] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<200704xf32> into tensor<64x3136xf32>
    %939 = tensor.collapse_shape %30 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<256x64x1x1xf32> into tensor<16384xf32>
    %940 = tensor.expand_shape %939 [[0 : i64, 1 : i64]] output_shape [256, 64] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<16384xf32> into tensor<256x64xf32>
    %941 = arith.constant {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} 0.000000e+00 : f32
    %942 = tensor.splat %941 {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<256x3136xf32>
    %943 = linalg.matmul {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} ins(%940, %938 : tensor<256x64xf32>, tensor<64x3136xf32>) outs(%942 : tensor<256x3136xf32>) -> tensor<256x3136xf32>
    %944 = tensor.collapse_shape %943 [[0 : i64, 1 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<256x3136xf32> into tensor<802816xf32>
    %945 = tensor.expand_shape %944 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 56, 56] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<802816xf32> into tensor<256x1x56x56xf32>
    %946 = tensor.collapse_shape %945 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<256x1x56x56xf32> into tensor<802816xf32>
    %947 = tensor.expand_shape %946 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 56, 56] {prov.region_id = "conv_10", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.conv3"} : tensor<802816xf32> into tensor<1x256x56x56xf32>
    %948 = tensor.empty() : tensor<1x256x56x56xf32>
    %949 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%947, %191 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%948 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} {
    ^bb83(%950: f32, %951: f32, %952: f32):
      %953 = arith.subf %950, %951 : f32
      linalg.yield %953 : f32
    } -> tensor<1x256x56x56xf32>
    %954 = arith.constant {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} 1.000000e-05 : f32
    %955 = tensor.splat %954 {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} : tensor<256xf32>
    %956 = tensor.empty() : tensor<256xf32>
    %957 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%192, %955 : tensor<256xf32>, tensor<256xf32>) outs(%956 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} {
    ^bb84(%958: f32, %959: f32, %960: f32):
      %961 = arith.addf %958, %959 : f32
      linalg.yield %961 : f32
    } -> tensor<256xf32>
    %962 = tensor.empty() : tensor<256xf32>
    %963 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%957 : tensor<256xf32>) outs(%962 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} {
    ^bb85(%964: f32, %965: f32):
      %966 = math.rsqrt %964 : f32
      linalg.yield %966 : f32
    } -> tensor<256xf32>
    %967 = tensor.empty() : tensor<1x256x56x56xf32>
    %968 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%949, %963 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%967 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} {
    ^bb86(%969: f32, %970: f32, %971: f32):
      %972 = arith.mulf %969, %970 : f32
      linalg.yield %972 : f32
    } -> tensor<1x256x56x56xf32>
    %973 = tensor.empty() : tensor<1x256x56x56xf32>
    %974 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%968, %31 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%973 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} {
    ^bb87(%975: f32, %976: f32, %977: f32):
      %978 = arith.mulf %975, %976 : f32
      linalg.yield %978 : f32
    } -> tensor<1x256x56x56xf32>
    %979 = tensor.empty() : tensor<1x256x56x56xf32>
    %980 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%974, %32 : tensor<1x256x56x56xf32>, tensor<256xf32>) outs(%979 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "batch_norm_10", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.bn3"} {
    ^bb88(%981: f32, %982: f32, %983: f32):
      %984 = arith.addf %981, %982 : f32
      linalg.yield %984 : f32
    } -> tensor<1x256x56x56xf32>
    %985 = tensor.empty() : tensor<1x256x56x56xf32>
    %986 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%980, %809 : tensor<1x256x56x56xf32>, tensor<1x256x56x56xf32>) outs(%985 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "add_2", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2"} {
    ^bb89(%987: f32, %988: f32, %989: f32):
      %990 = arith.addf %987, %988 : f32
      linalg.yield %990 : f32
    } -> tensor<1x256x56x56xf32>
    %991 = tensor.empty() : tensor<1x256x56x56xf32>
    %992 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%986 : tensor<1x256x56x56xf32>) outs(%991 : tensor<1x256x56x56xf32>) attrs =  {prov.region_id = "minmax_9", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer1.2.relu"} {
    ^bb90(%993: f32, %994: f32):
      %995 = arith.constant 0.000000e+00 : f32
      %996 = arith.maximumf %993, %995 : f32
      linalg.yield %996 : f32
    } -> tensor<1x256x56x56xf32>
    %997 = tensor.empty() : tensor<256x1x1x1x56x56xf32>
    %998 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%992 : tensor<1x256x56x56xf32>) outs(%997 : tensor<256x1x1x1x56x56xf32>) attrs =  {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} {
    ^bb91(%999: f32, %1000: f32):
      linalg.yield %999 : f32
    } -> tensor<256x1x1x1x56x56xf32>
    %1001 = tensor.collapse_shape %998 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<256x1x1x1x56x56xf32> into tensor<802816xf32>
    %1002 = tensor.expand_shape %1001 [[0 : i64, 1 : i64]] output_shape [256, 3136] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<802816xf32> into tensor<256x3136xf32>
    %1003 = tensor.collapse_shape %33 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<128x256x1x1xf32> into tensor<32768xf32>
    %1004 = tensor.expand_shape %1003 [[0 : i64, 1 : i64]] output_shape [128, 256] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<32768xf32> into tensor<128x256xf32>
    %1005 = arith.constant {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} 0.000000e+00 : f32
    %1006 = tensor.splat %1005 {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<128x3136xf32>
    %1007 = linalg.matmul {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} ins(%1004, %1002 : tensor<128x256xf32>, tensor<256x3136xf32>) outs(%1006 : tensor<128x3136xf32>) -> tensor<128x3136xf32>
    %1008 = tensor.collapse_shape %1007 [[0 : i64, 1 : i64]] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<128x3136xf32> into tensor<401408xf32>
    %1009 = tensor.expand_shape %1008 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 56, 56] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<401408xf32> into tensor<128x1x56x56xf32>
    %1010 = tensor.collapse_shape %1009 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<128x1x56x56xf32> into tensor<401408xf32>
    %1011 = tensor.expand_shape %1010 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 56, 56] {prov.region_id = "conv_11", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv1"} : tensor<401408xf32> into tensor<1x128x56x56xf32>
    %1012 = tensor.empty() : tensor<1x128x56x56xf32>
    %1013 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1011, %194 : tensor<1x128x56x56xf32>, tensor<128xf32>) outs(%1012 : tensor<1x128x56x56xf32>) attrs =  {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} {
    ^bb92(%1014: f32, %1015: f32, %1016: f32):
      %1017 = arith.subf %1014, %1015 : f32
      linalg.yield %1017 : f32
    } -> tensor<1x128x56x56xf32>
    %1018 = arith.constant {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} 1.000000e-05 : f32
    %1019 = tensor.splat %1018 {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} : tensor<128xf32>
    %1020 = tensor.empty() : tensor<128xf32>
    %1021 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%195, %1019 : tensor<128xf32>, tensor<128xf32>) outs(%1020 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} {
    ^bb93(%1022: f32, %1023: f32, %1024: f32):
      %1025 = arith.addf %1022, %1023 : f32
      linalg.yield %1025 : f32
    } -> tensor<128xf32>
    %1026 = tensor.empty() : tensor<128xf32>
    %1027 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1021 : tensor<128xf32>) outs(%1026 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} {
    ^bb94(%1028: f32, %1029: f32):
      %1030 = math.rsqrt %1028 : f32
      linalg.yield %1030 : f32
    } -> tensor<128xf32>
    %1031 = tensor.empty() : tensor<1x128x56x56xf32>
    %1032 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1013, %1027 : tensor<1x128x56x56xf32>, tensor<128xf32>) outs(%1031 : tensor<1x128x56x56xf32>) attrs =  {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} {
    ^bb95(%1033: f32, %1034: f32, %1035: f32):
      %1036 = arith.mulf %1033, %1034 : f32
      linalg.yield %1036 : f32
    } -> tensor<1x128x56x56xf32>
    %1037 = tensor.empty() : tensor<1x128x56x56xf32>
    %1038 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1032, %34 : tensor<1x128x56x56xf32>, tensor<128xf32>) outs(%1037 : tensor<1x128x56x56xf32>) attrs =  {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} {
    ^bb96(%1039: f32, %1040: f32, %1041: f32):
      %1042 = arith.mulf %1039, %1040 : f32
      linalg.yield %1042 : f32
    } -> tensor<1x128x56x56xf32>
    %1043 = tensor.empty() : tensor<1x128x56x56xf32>
    %1044 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1038, %35 : tensor<1x128x56x56xf32>, tensor<128xf32>) outs(%1043 : tensor<1x128x56x56xf32>) attrs =  {prov.region_id = "batch_norm_11", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn1"} {
    ^bb97(%1045: f32, %1046: f32, %1047: f32):
      %1048 = arith.addf %1045, %1046 : f32
      linalg.yield %1048 : f32
    } -> tensor<1x128x56x56xf32>
    %1049 = tensor.empty() : tensor<1x128x56x56xf32>
    %1050 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1044 : tensor<1x128x56x56xf32>) outs(%1049 : tensor<1x128x56x56xf32>) attrs =  {prov.region_id = "minmax_10", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.relu"} {
    ^bb98(%1051: f32, %1052: f32):
      %1053 = arith.constant 0.000000e+00 : f32
      %1054 = arith.maximumf %1051, %1053 : f32
      linalg.yield %1054 : f32
    } -> tensor<1x128x56x56xf32>
    %1055 = arith.constant {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} 0.000000e+00 : f32
    %1056 = tensor.splat %1055 {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<1x128x58x58xf32>
    %1057 = "tensor.insert_slice"(%1050, %1056) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 128, 56, 56>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : (tensor<1x128x56x56xf32>, tensor<1x128x58x58xf32>) -> tensor<1x128x58x58xf32>
    %1058 = tensor.empty() : tensor<128x3x3x1x28x28xf32>
    %1059 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1057 : tensor<1x128x58x58xf32>) outs(%1058 : tensor<128x3x3x1x28x28xf32>) attrs =  {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} {
    ^bb99(%1060: f32, %1061: f32):
      linalg.yield %1060 : f32
    } -> tensor<128x3x3x1x28x28xf32>
    %1062 = tensor.collapse_shape %1059 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<128x3x3x1x28x28xf32> into tensor<903168xf32>
    %1063 = tensor.expand_shape %1062 [[0 : i64, 1 : i64]] output_shape [1152, 784] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<903168xf32> into tensor<1152x784xf32>
    %1064 = tensor.collapse_shape %36 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<128x128x3x3xf32> into tensor<147456xf32>
    %1065 = tensor.expand_shape %1064 [[0 : i64, 1 : i64]] output_shape [128, 1152] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<147456xf32> into tensor<128x1152xf32>
    %1066 = arith.constant {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} 0.000000e+00 : f32
    %1067 = tensor.splat %1066 {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<128x784xf32>
    %1068 = linalg.matmul {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} ins(%1065, %1063 : tensor<128x1152xf32>, tensor<1152x784xf32>) outs(%1067 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1069 = tensor.collapse_shape %1068 [[0 : i64, 1 : i64]] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<128x784xf32> into tensor<100352xf32>
    %1070 = tensor.expand_shape %1069 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1071 = tensor.collapse_shape %1070 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1072 = tensor.expand_shape %1071 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_12", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv2"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1073 = tensor.empty() : tensor<1x128x28x28xf32>
    %1074 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1072, %197 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1073 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} {
    ^bb100(%1075: f32, %1076: f32, %1077: f32):
      %1078 = arith.subf %1075, %1076 : f32
      linalg.yield %1078 : f32
    } -> tensor<1x128x28x28xf32>
    %1079 = arith.constant {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} 1.000000e-05 : f32
    %1080 = tensor.splat %1079 {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} : tensor<128xf32>
    %1081 = tensor.empty() : tensor<128xf32>
    %1082 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%198, %1080 : tensor<128xf32>, tensor<128xf32>) outs(%1081 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} {
    ^bb101(%1083: f32, %1084: f32, %1085: f32):
      %1086 = arith.addf %1083, %1084 : f32
      linalg.yield %1086 : f32
    } -> tensor<128xf32>
    %1087 = tensor.empty() : tensor<128xf32>
    %1088 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1082 : tensor<128xf32>) outs(%1087 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} {
    ^bb102(%1089: f32, %1090: f32):
      %1091 = math.rsqrt %1089 : f32
      linalg.yield %1091 : f32
    } -> tensor<128xf32>
    %1092 = tensor.empty() : tensor<1x128x28x28xf32>
    %1093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1074, %1088 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1092 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} {
    ^bb103(%1094: f32, %1095: f32, %1096: f32):
      %1097 = arith.mulf %1094, %1095 : f32
      linalg.yield %1097 : f32
    } -> tensor<1x128x28x28xf32>
    %1098 = tensor.empty() : tensor<1x128x28x28xf32>
    %1099 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1093, %37 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1098 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} {
    ^bb104(%1100: f32, %1101: f32, %1102: f32):
      %1103 = arith.mulf %1100, %1101 : f32
      linalg.yield %1103 : f32
    } -> tensor<1x128x28x28xf32>
    %1104 = tensor.empty() : tensor<1x128x28x28xf32>
    %1105 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1099, %38 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1104 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_12", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn2"} {
    ^bb105(%1106: f32, %1107: f32, %1108: f32):
      %1109 = arith.addf %1106, %1107 : f32
      linalg.yield %1109 : f32
    } -> tensor<1x128x28x28xf32>
    %1110 = tensor.empty() : tensor<1x128x28x28xf32>
    %1111 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1105 : tensor<1x128x28x28xf32>) outs(%1110 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_11", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.relu"} {
    ^bb106(%1112: f32, %1113: f32):
      %1114 = arith.constant 0.000000e+00 : f32
      %1115 = arith.maximumf %1112, %1114 : f32
      linalg.yield %1115 : f32
    } -> tensor<1x128x28x28xf32>
    %1116 = tensor.empty() : tensor<128x1x1x1x28x28xf32>
    %1117 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1111 : tensor<1x128x28x28xf32>) outs(%1116 : tensor<128x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} {
    ^bb107(%1118: f32, %1119: f32):
      linalg.yield %1118 : f32
    } -> tensor<128x1x1x1x28x28xf32>
    %1120 = tensor.collapse_shape %1117 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<128x1x1x1x28x28xf32> into tensor<100352xf32>
    %1121 = tensor.expand_shape %1120 [[0 : i64, 1 : i64]] output_shape [128, 784] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<100352xf32> into tensor<128x784xf32>
    %1122 = tensor.collapse_shape %39 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<512x128x1x1xf32> into tensor<65536xf32>
    %1123 = tensor.expand_shape %1122 [[0 : i64, 1 : i64]] output_shape [512, 128] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<65536xf32> into tensor<512x128xf32>
    %1124 = arith.constant {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} 0.000000e+00 : f32
    %1125 = tensor.splat %1124 {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<512x784xf32>
    %1126 = linalg.matmul {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} ins(%1123, %1121 : tensor<512x128xf32>, tensor<128x784xf32>) outs(%1125 : tensor<512x784xf32>) -> tensor<512x784xf32>
    %1127 = tensor.collapse_shape %1126 [[0 : i64, 1 : i64]] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<512x784xf32> into tensor<401408xf32>
    %1128 = tensor.expand_shape %1127 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 28, 28] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<401408xf32> into tensor<512x1x28x28xf32>
    %1129 = tensor.collapse_shape %1128 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<512x1x28x28xf32> into tensor<401408xf32>
    %1130 = tensor.expand_shape %1129 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 28, 28] {prov.region_id = "conv_13", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.conv3"} : tensor<401408xf32> into tensor<1x512x28x28xf32>
    %1131 = tensor.empty() : tensor<1x512x28x28xf32>
    %1132 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1130, %200 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1131 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} {
    ^bb108(%1133: f32, %1134: f32, %1135: f32):
      %1136 = arith.subf %1133, %1134 : f32
      linalg.yield %1136 : f32
    } -> tensor<1x512x28x28xf32>
    %1137 = arith.constant {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} 1.000000e-05 : f32
    %1138 = tensor.splat %1137 {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} : tensor<512xf32>
    %1139 = tensor.empty() : tensor<512xf32>
    %1140 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%201, %1138 : tensor<512xf32>, tensor<512xf32>) outs(%1139 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} {
    ^bb109(%1141: f32, %1142: f32, %1143: f32):
      %1144 = arith.addf %1141, %1142 : f32
      linalg.yield %1144 : f32
    } -> tensor<512xf32>
    %1145 = tensor.empty() : tensor<512xf32>
    %1146 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1140 : tensor<512xf32>) outs(%1145 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} {
    ^bb110(%1147: f32, %1148: f32):
      %1149 = math.rsqrt %1147 : f32
      linalg.yield %1149 : f32
    } -> tensor<512xf32>
    %1150 = tensor.empty() : tensor<1x512x28x28xf32>
    %1151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1132, %1146 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1150 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} {
    ^bb111(%1152: f32, %1153: f32, %1154: f32):
      %1155 = arith.mulf %1152, %1153 : f32
      linalg.yield %1155 : f32
    } -> tensor<1x512x28x28xf32>
    %1156 = tensor.empty() : tensor<1x512x28x28xf32>
    %1157 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1151, %40 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1156 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} {
    ^bb112(%1158: f32, %1159: f32, %1160: f32):
      %1161 = arith.mulf %1158, %1159 : f32
      linalg.yield %1161 : f32
    } -> tensor<1x512x28x28xf32>
    %1162 = tensor.empty() : tensor<1x512x28x28xf32>
    %1163 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1157, %41 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1162 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_13", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.bn3"} {
    ^bb113(%1164: f32, %1165: f32, %1166: f32):
      %1167 = arith.addf %1164, %1165 : f32
      linalg.yield %1167 : f32
    } -> tensor<1x512x28x28xf32>
    %1168 = tensor.empty() : tensor<256x1x1x1x28x28xf32>
    %1169 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%992 : tensor<1x256x56x56xf32>) outs(%1168 : tensor<256x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} {
    ^bb114(%1170: f32, %1171: f32):
      linalg.yield %1170 : f32
    } -> tensor<256x1x1x1x28x28xf32>
    %1172 = tensor.collapse_shape %1169 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<256x1x1x1x28x28xf32> into tensor<200704xf32>
    %1173 = tensor.expand_shape %1172 [[0 : i64, 1 : i64]] output_shape [256, 784] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<200704xf32> into tensor<256x784xf32>
    %1174 = tensor.collapse_shape %42 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<512x256x1x1xf32> into tensor<131072xf32>
    %1175 = tensor.expand_shape %1174 [[0 : i64, 1 : i64]] output_shape [512, 256] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<131072xf32> into tensor<512x256xf32>
    %1176 = arith.constant {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} 0.000000e+00 : f32
    %1177 = tensor.splat %1176 {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<512x784xf32>
    %1178 = linalg.matmul {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} ins(%1175, %1173 : tensor<512x256xf32>, tensor<256x784xf32>) outs(%1177 : tensor<512x784xf32>) -> tensor<512x784xf32>
    %1179 = tensor.collapse_shape %1178 [[0 : i64, 1 : i64]] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<512x784xf32> into tensor<401408xf32>
    %1180 = tensor.expand_shape %1179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 28, 28] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<401408xf32> into tensor<512x1x28x28xf32>
    %1181 = tensor.collapse_shape %1180 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<512x1x28x28xf32> into tensor<401408xf32>
    %1182 = tensor.expand_shape %1181 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 28, 28] {prov.region_id = "conv_14", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.0"} : tensor<401408xf32> into tensor<1x512x28x28xf32>
    %1183 = tensor.empty() : tensor<1x512x28x28xf32>
    %1184 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1182, %203 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1183 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} {
    ^bb115(%1185: f32, %1186: f32, %1187: f32):
      %1188 = arith.subf %1185, %1186 : f32
      linalg.yield %1188 : f32
    } -> tensor<1x512x28x28xf32>
    %1189 = arith.constant {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} 1.000000e-05 : f32
    %1190 = tensor.splat %1189 {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} : tensor<512xf32>
    %1191 = tensor.empty() : tensor<512xf32>
    %1192 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%204, %1190 : tensor<512xf32>, tensor<512xf32>) outs(%1191 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} {
    ^bb116(%1193: f32, %1194: f32, %1195: f32):
      %1196 = arith.addf %1193, %1194 : f32
      linalg.yield %1196 : f32
    } -> tensor<512xf32>
    %1197 = tensor.empty() : tensor<512xf32>
    %1198 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1192 : tensor<512xf32>) outs(%1197 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} {
    ^bb117(%1199: f32, %1200: f32):
      %1201 = math.rsqrt %1199 : f32
      linalg.yield %1201 : f32
    } -> tensor<512xf32>
    %1202 = tensor.empty() : tensor<1x512x28x28xf32>
    %1203 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1184, %1198 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1202 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} {
    ^bb118(%1204: f32, %1205: f32, %1206: f32):
      %1207 = arith.mulf %1204, %1205 : f32
      linalg.yield %1207 : f32
    } -> tensor<1x512x28x28xf32>
    %1208 = tensor.empty() : tensor<1x512x28x28xf32>
    %1209 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1203, %43 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1208 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} {
    ^bb119(%1210: f32, %1211: f32, %1212: f32):
      %1213 = arith.mulf %1210, %1211 : f32
      linalg.yield %1213 : f32
    } -> tensor<1x512x28x28xf32>
    %1214 = tensor.empty() : tensor<1x512x28x28xf32>
    %1215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1209, %44 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1214 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_14", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.downsample.1"} {
    ^bb120(%1216: f32, %1217: f32, %1218: f32):
      %1219 = arith.addf %1216, %1217 : f32
      linalg.yield %1219 : f32
    } -> tensor<1x512x28x28xf32>
    %1220 = tensor.empty() : tensor<1x512x28x28xf32>
    %1221 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1163, %1215 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%1220 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "add_3", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0"} {
    ^bb121(%1222: f32, %1223: f32, %1224: f32):
      %1225 = arith.addf %1222, %1223 : f32
      linalg.yield %1225 : f32
    } -> tensor<1x512x28x28xf32>
    %1226 = tensor.empty() : tensor<1x512x28x28xf32>
    %1227 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1221 : tensor<1x512x28x28xf32>) outs(%1226 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "minmax_12", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.0.relu"} {
    ^bb122(%1228: f32, %1229: f32):
      %1230 = arith.constant 0.000000e+00 : f32
      %1231 = arith.maximumf %1228, %1230 : f32
      linalg.yield %1231 : f32
    } -> tensor<1x512x28x28xf32>
    %1232 = tensor.empty() : tensor<512x1x1x1x28x28xf32>
    %1233 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1227 : tensor<1x512x28x28xf32>) outs(%1232 : tensor<512x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} {
    ^bb123(%1234: f32, %1235: f32):
      linalg.yield %1234 : f32
    } -> tensor<512x1x1x1x28x28xf32>
    %1236 = tensor.collapse_shape %1233 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<512x1x1x1x28x28xf32> into tensor<401408xf32>
    %1237 = tensor.expand_shape %1236 [[0 : i64, 1 : i64]] output_shape [512, 784] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<401408xf32> into tensor<512x784xf32>
    %1238 = tensor.collapse_shape %45 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<128x512x1x1xf32> into tensor<65536xf32>
    %1239 = tensor.expand_shape %1238 [[0 : i64, 1 : i64]] output_shape [128, 512] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<65536xf32> into tensor<128x512xf32>
    %1240 = arith.constant {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} 0.000000e+00 : f32
    %1241 = tensor.splat %1240 {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<128x784xf32>
    %1242 = linalg.matmul {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} ins(%1239, %1237 : tensor<128x512xf32>, tensor<512x784xf32>) outs(%1241 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1243 = tensor.collapse_shape %1242 [[0 : i64, 1 : i64]] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<128x784xf32> into tensor<100352xf32>
    %1244 = tensor.expand_shape %1243 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1245 = tensor.collapse_shape %1244 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1246 = tensor.expand_shape %1245 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_15", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv1"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1247 = tensor.empty() : tensor<1x128x28x28xf32>
    %1248 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1246, %206 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1247 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} {
    ^bb124(%1249: f32, %1250: f32, %1251: f32):
      %1252 = arith.subf %1249, %1250 : f32
      linalg.yield %1252 : f32
    } -> tensor<1x128x28x28xf32>
    %1253 = arith.constant {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} 1.000000e-05 : f32
    %1254 = tensor.splat %1253 {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} : tensor<128xf32>
    %1255 = tensor.empty() : tensor<128xf32>
    %1256 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%207, %1254 : tensor<128xf32>, tensor<128xf32>) outs(%1255 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} {
    ^bb125(%1257: f32, %1258: f32, %1259: f32):
      %1260 = arith.addf %1257, %1258 : f32
      linalg.yield %1260 : f32
    } -> tensor<128xf32>
    %1261 = tensor.empty() : tensor<128xf32>
    %1262 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1256 : tensor<128xf32>) outs(%1261 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} {
    ^bb126(%1263: f32, %1264: f32):
      %1265 = math.rsqrt %1263 : f32
      linalg.yield %1265 : f32
    } -> tensor<128xf32>
    %1266 = tensor.empty() : tensor<1x128x28x28xf32>
    %1267 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1248, %1262 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1266 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} {
    ^bb127(%1268: f32, %1269: f32, %1270: f32):
      %1271 = arith.mulf %1268, %1269 : f32
      linalg.yield %1271 : f32
    } -> tensor<1x128x28x28xf32>
    %1272 = tensor.empty() : tensor<1x128x28x28xf32>
    %1273 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1267, %46 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1272 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} {
    ^bb128(%1274: f32, %1275: f32, %1276: f32):
      %1277 = arith.mulf %1274, %1275 : f32
      linalg.yield %1277 : f32
    } -> tensor<1x128x28x28xf32>
    %1278 = tensor.empty() : tensor<1x128x28x28xf32>
    %1279 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1273, %47 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1278 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_15", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn1"} {
    ^bb129(%1280: f32, %1281: f32, %1282: f32):
      %1283 = arith.addf %1280, %1281 : f32
      linalg.yield %1283 : f32
    } -> tensor<1x128x28x28xf32>
    %1284 = tensor.empty() : tensor<1x128x28x28xf32>
    %1285 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1279 : tensor<1x128x28x28xf32>) outs(%1284 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_13", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.relu"} {
    ^bb130(%1286: f32, %1287: f32):
      %1288 = arith.constant 0.000000e+00 : f32
      %1289 = arith.maximumf %1286, %1288 : f32
      linalg.yield %1289 : f32
    } -> tensor<1x128x28x28xf32>
    %1290 = arith.constant {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} 0.000000e+00 : f32
    %1291 = tensor.splat %1290 {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<1x128x30x30xf32>
    %1292 = "tensor.insert_slice"(%1285, %1291) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 128, 28, 28>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : (tensor<1x128x28x28xf32>, tensor<1x128x30x30xf32>) -> tensor<1x128x30x30xf32>
    %1293 = tensor.empty() : tensor<128x3x3x1x28x28xf32>
    %1294 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1292 : tensor<1x128x30x30xf32>) outs(%1293 : tensor<128x3x3x1x28x28xf32>) attrs =  {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} {
    ^bb131(%1295: f32, %1296: f32):
      linalg.yield %1295 : f32
    } -> tensor<128x3x3x1x28x28xf32>
    %1297 = tensor.collapse_shape %1294 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<128x3x3x1x28x28xf32> into tensor<903168xf32>
    %1298 = tensor.expand_shape %1297 [[0 : i64, 1 : i64]] output_shape [1152, 784] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<903168xf32> into tensor<1152x784xf32>
    %1299 = tensor.collapse_shape %48 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<128x128x3x3xf32> into tensor<147456xf32>
    %1300 = tensor.expand_shape %1299 [[0 : i64, 1 : i64]] output_shape [128, 1152] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<147456xf32> into tensor<128x1152xf32>
    %1301 = arith.constant {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} 0.000000e+00 : f32
    %1302 = tensor.splat %1301 {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<128x784xf32>
    %1303 = linalg.matmul {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} ins(%1300, %1298 : tensor<128x1152xf32>, tensor<1152x784xf32>) outs(%1302 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1304 = tensor.collapse_shape %1303 [[0 : i64, 1 : i64]] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<128x784xf32> into tensor<100352xf32>
    %1305 = tensor.expand_shape %1304 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1306 = tensor.collapse_shape %1305 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1307 = tensor.expand_shape %1306 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_16", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv2"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1308 = tensor.empty() : tensor<1x128x28x28xf32>
    %1309 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1307, %209 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1308 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} {
    ^bb132(%1310: f32, %1311: f32, %1312: f32):
      %1313 = arith.subf %1310, %1311 : f32
      linalg.yield %1313 : f32
    } -> tensor<1x128x28x28xf32>
    %1314 = arith.constant {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} 1.000000e-05 : f32
    %1315 = tensor.splat %1314 {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} : tensor<128xf32>
    %1316 = tensor.empty() : tensor<128xf32>
    %1317 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%210, %1315 : tensor<128xf32>, tensor<128xf32>) outs(%1316 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} {
    ^bb133(%1318: f32, %1319: f32, %1320: f32):
      %1321 = arith.addf %1318, %1319 : f32
      linalg.yield %1321 : f32
    } -> tensor<128xf32>
    %1322 = tensor.empty() : tensor<128xf32>
    %1323 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1317 : tensor<128xf32>) outs(%1322 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} {
    ^bb134(%1324: f32, %1325: f32):
      %1326 = math.rsqrt %1324 : f32
      linalg.yield %1326 : f32
    } -> tensor<128xf32>
    %1327 = tensor.empty() : tensor<1x128x28x28xf32>
    %1328 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1309, %1323 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1327 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} {
    ^bb135(%1329: f32, %1330: f32, %1331: f32):
      %1332 = arith.mulf %1329, %1330 : f32
      linalg.yield %1332 : f32
    } -> tensor<1x128x28x28xf32>
    %1333 = tensor.empty() : tensor<1x128x28x28xf32>
    %1334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1328, %49 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1333 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} {
    ^bb136(%1335: f32, %1336: f32, %1337: f32):
      %1338 = arith.mulf %1335, %1336 : f32
      linalg.yield %1338 : f32
    } -> tensor<1x128x28x28xf32>
    %1339 = tensor.empty() : tensor<1x128x28x28xf32>
    %1340 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1334, %50 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1339 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_16", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn2"} {
    ^bb137(%1341: f32, %1342: f32, %1343: f32):
      %1344 = arith.addf %1341, %1342 : f32
      linalg.yield %1344 : f32
    } -> tensor<1x128x28x28xf32>
    %1345 = tensor.empty() : tensor<1x128x28x28xf32>
    %1346 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1340 : tensor<1x128x28x28xf32>) outs(%1345 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_14", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.relu"} {
    ^bb138(%1347: f32, %1348: f32):
      %1349 = arith.constant 0.000000e+00 : f32
      %1350 = arith.maximumf %1347, %1349 : f32
      linalg.yield %1350 : f32
    } -> tensor<1x128x28x28xf32>
    %1351 = tensor.empty() : tensor<128x1x1x1x28x28xf32>
    %1352 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1346 : tensor<1x128x28x28xf32>) outs(%1351 : tensor<128x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} {
    ^bb139(%1353: f32, %1354: f32):
      linalg.yield %1353 : f32
    } -> tensor<128x1x1x1x28x28xf32>
    %1355 = tensor.collapse_shape %1352 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<128x1x1x1x28x28xf32> into tensor<100352xf32>
    %1356 = tensor.expand_shape %1355 [[0 : i64, 1 : i64]] output_shape [128, 784] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<100352xf32> into tensor<128x784xf32>
    %1357 = tensor.collapse_shape %51 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<512x128x1x1xf32> into tensor<65536xf32>
    %1358 = tensor.expand_shape %1357 [[0 : i64, 1 : i64]] output_shape [512, 128] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<65536xf32> into tensor<512x128xf32>
    %1359 = arith.constant {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} 0.000000e+00 : f32
    %1360 = tensor.splat %1359 {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<512x784xf32>
    %1361 = linalg.matmul {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} ins(%1358, %1356 : tensor<512x128xf32>, tensor<128x784xf32>) outs(%1360 : tensor<512x784xf32>) -> tensor<512x784xf32>
    %1362 = tensor.collapse_shape %1361 [[0 : i64, 1 : i64]] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<512x784xf32> into tensor<401408xf32>
    %1363 = tensor.expand_shape %1362 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 28, 28] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<401408xf32> into tensor<512x1x28x28xf32>
    %1364 = tensor.collapse_shape %1363 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<512x1x28x28xf32> into tensor<401408xf32>
    %1365 = tensor.expand_shape %1364 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 28, 28] {prov.region_id = "conv_17", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.conv3"} : tensor<401408xf32> into tensor<1x512x28x28xf32>
    %1366 = tensor.empty() : tensor<1x512x28x28xf32>
    %1367 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1365, %212 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1366 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} {
    ^bb140(%1368: f32, %1369: f32, %1370: f32):
      %1371 = arith.subf %1368, %1369 : f32
      linalg.yield %1371 : f32
    } -> tensor<1x512x28x28xf32>
    %1372 = arith.constant {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} 1.000000e-05 : f32
    %1373 = tensor.splat %1372 {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} : tensor<512xf32>
    %1374 = tensor.empty() : tensor<512xf32>
    %1375 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%213, %1373 : tensor<512xf32>, tensor<512xf32>) outs(%1374 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} {
    ^bb141(%1376: f32, %1377: f32, %1378: f32):
      %1379 = arith.addf %1376, %1377 : f32
      linalg.yield %1379 : f32
    } -> tensor<512xf32>
    %1380 = tensor.empty() : tensor<512xf32>
    %1381 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1375 : tensor<512xf32>) outs(%1380 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} {
    ^bb142(%1382: f32, %1383: f32):
      %1384 = math.rsqrt %1382 : f32
      linalg.yield %1384 : f32
    } -> tensor<512xf32>
    %1385 = tensor.empty() : tensor<1x512x28x28xf32>
    %1386 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1367, %1381 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1385 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} {
    ^bb143(%1387: f32, %1388: f32, %1389: f32):
      %1390 = arith.mulf %1387, %1388 : f32
      linalg.yield %1390 : f32
    } -> tensor<1x512x28x28xf32>
    %1391 = tensor.empty() : tensor<1x512x28x28xf32>
    %1392 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1386, %52 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1391 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} {
    ^bb144(%1393: f32, %1394: f32, %1395: f32):
      %1396 = arith.mulf %1393, %1394 : f32
      linalg.yield %1396 : f32
    } -> tensor<1x512x28x28xf32>
    %1397 = tensor.empty() : tensor<1x512x28x28xf32>
    %1398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1392, %53 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1397 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_17", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.bn3"} {
    ^bb145(%1399: f32, %1400: f32, %1401: f32):
      %1402 = arith.addf %1399, %1400 : f32
      linalg.yield %1402 : f32
    } -> tensor<1x512x28x28xf32>
    %1403 = tensor.empty() : tensor<1x512x28x28xf32>
    %1404 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1398, %1227 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%1403 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "add_4", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1"} {
    ^bb146(%1405: f32, %1406: f32, %1407: f32):
      %1408 = arith.addf %1405, %1406 : f32
      linalg.yield %1408 : f32
    } -> tensor<1x512x28x28xf32>
    %1409 = tensor.empty() : tensor<1x512x28x28xf32>
    %1410 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1404 : tensor<1x512x28x28xf32>) outs(%1409 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "minmax_15", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.1.relu"} {
    ^bb147(%1411: f32, %1412: f32):
      %1413 = arith.constant 0.000000e+00 : f32
      %1414 = arith.maximumf %1411, %1413 : f32
      linalg.yield %1414 : f32
    } -> tensor<1x512x28x28xf32>
    %1415 = tensor.empty() : tensor<512x1x1x1x28x28xf32>
    %1416 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1410 : tensor<1x512x28x28xf32>) outs(%1415 : tensor<512x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} {
    ^bb148(%1417: f32, %1418: f32):
      linalg.yield %1417 : f32
    } -> tensor<512x1x1x1x28x28xf32>
    %1419 = tensor.collapse_shape %1416 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<512x1x1x1x28x28xf32> into tensor<401408xf32>
    %1420 = tensor.expand_shape %1419 [[0 : i64, 1 : i64]] output_shape [512, 784] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<401408xf32> into tensor<512x784xf32>
    %1421 = tensor.collapse_shape %54 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<128x512x1x1xf32> into tensor<65536xf32>
    %1422 = tensor.expand_shape %1421 [[0 : i64, 1 : i64]] output_shape [128, 512] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<65536xf32> into tensor<128x512xf32>
    %1423 = arith.constant {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} 0.000000e+00 : f32
    %1424 = tensor.splat %1423 {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<128x784xf32>
    %1425 = linalg.matmul {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} ins(%1422, %1420 : tensor<128x512xf32>, tensor<512x784xf32>) outs(%1424 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1426 = tensor.collapse_shape %1425 [[0 : i64, 1 : i64]] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<128x784xf32> into tensor<100352xf32>
    %1427 = tensor.expand_shape %1426 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1428 = tensor.collapse_shape %1427 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1429 = tensor.expand_shape %1428 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_18", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv1"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1430 = tensor.empty() : tensor<1x128x28x28xf32>
    %1431 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1429, %215 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1430 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} {
    ^bb149(%1432: f32, %1433: f32, %1434: f32):
      %1435 = arith.subf %1432, %1433 : f32
      linalg.yield %1435 : f32
    } -> tensor<1x128x28x28xf32>
    %1436 = arith.constant {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} 1.000000e-05 : f32
    %1437 = tensor.splat %1436 {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} : tensor<128xf32>
    %1438 = tensor.empty() : tensor<128xf32>
    %1439 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%216, %1437 : tensor<128xf32>, tensor<128xf32>) outs(%1438 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} {
    ^bb150(%1440: f32, %1441: f32, %1442: f32):
      %1443 = arith.addf %1440, %1441 : f32
      linalg.yield %1443 : f32
    } -> tensor<128xf32>
    %1444 = tensor.empty() : tensor<128xf32>
    %1445 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1439 : tensor<128xf32>) outs(%1444 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} {
    ^bb151(%1446: f32, %1447: f32):
      %1448 = math.rsqrt %1446 : f32
      linalg.yield %1448 : f32
    } -> tensor<128xf32>
    %1449 = tensor.empty() : tensor<1x128x28x28xf32>
    %1450 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1431, %1445 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1449 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} {
    ^bb152(%1451: f32, %1452: f32, %1453: f32):
      %1454 = arith.mulf %1451, %1452 : f32
      linalg.yield %1454 : f32
    } -> tensor<1x128x28x28xf32>
    %1455 = tensor.empty() : tensor<1x128x28x28xf32>
    %1456 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1450, %55 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1455 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} {
    ^bb153(%1457: f32, %1458: f32, %1459: f32):
      %1460 = arith.mulf %1457, %1458 : f32
      linalg.yield %1460 : f32
    } -> tensor<1x128x28x28xf32>
    %1461 = tensor.empty() : tensor<1x128x28x28xf32>
    %1462 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1456, %56 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1461 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_18", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn1"} {
    ^bb154(%1463: f32, %1464: f32, %1465: f32):
      %1466 = arith.addf %1463, %1464 : f32
      linalg.yield %1466 : f32
    } -> tensor<1x128x28x28xf32>
    %1467 = tensor.empty() : tensor<1x128x28x28xf32>
    %1468 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1462 : tensor<1x128x28x28xf32>) outs(%1467 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_16", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.relu"} {
    ^bb155(%1469: f32, %1470: f32):
      %1471 = arith.constant 0.000000e+00 : f32
      %1472 = arith.maximumf %1469, %1471 : f32
      linalg.yield %1472 : f32
    } -> tensor<1x128x28x28xf32>
    %1473 = arith.constant {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} 0.000000e+00 : f32
    %1474 = tensor.splat %1473 {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<1x128x30x30xf32>
    %1475 = "tensor.insert_slice"(%1468, %1474) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 128, 28, 28>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : (tensor<1x128x28x28xf32>, tensor<1x128x30x30xf32>) -> tensor<1x128x30x30xf32>
    %1476 = tensor.empty() : tensor<128x3x3x1x28x28xf32>
    %1477 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1475 : tensor<1x128x30x30xf32>) outs(%1476 : tensor<128x3x3x1x28x28xf32>) attrs =  {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} {
    ^bb156(%1478: f32, %1479: f32):
      linalg.yield %1478 : f32
    } -> tensor<128x3x3x1x28x28xf32>
    %1480 = tensor.collapse_shape %1477 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<128x3x3x1x28x28xf32> into tensor<903168xf32>
    %1481 = tensor.expand_shape %1480 [[0 : i64, 1 : i64]] output_shape [1152, 784] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<903168xf32> into tensor<1152x784xf32>
    %1482 = tensor.collapse_shape %57 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<128x128x3x3xf32> into tensor<147456xf32>
    %1483 = tensor.expand_shape %1482 [[0 : i64, 1 : i64]] output_shape [128, 1152] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<147456xf32> into tensor<128x1152xf32>
    %1484 = arith.constant {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} 0.000000e+00 : f32
    %1485 = tensor.splat %1484 {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<128x784xf32>
    %1486 = linalg.matmul {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} ins(%1483, %1481 : tensor<128x1152xf32>, tensor<1152x784xf32>) outs(%1485 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1487 = tensor.collapse_shape %1486 [[0 : i64, 1 : i64]] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<128x784xf32> into tensor<100352xf32>
    %1488 = tensor.expand_shape %1487 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1489 = tensor.collapse_shape %1488 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1490 = tensor.expand_shape %1489 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_19", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv2"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1491 = tensor.empty() : tensor<1x128x28x28xf32>
    %1492 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1490, %218 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1491 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} {
    ^bb157(%1493: f32, %1494: f32, %1495: f32):
      %1496 = arith.subf %1493, %1494 : f32
      linalg.yield %1496 : f32
    } -> tensor<1x128x28x28xf32>
    %1497 = arith.constant {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} 1.000000e-05 : f32
    %1498 = tensor.splat %1497 {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} : tensor<128xf32>
    %1499 = tensor.empty() : tensor<128xf32>
    %1500 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%219, %1498 : tensor<128xf32>, tensor<128xf32>) outs(%1499 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} {
    ^bb158(%1501: f32, %1502: f32, %1503: f32):
      %1504 = arith.addf %1501, %1502 : f32
      linalg.yield %1504 : f32
    } -> tensor<128xf32>
    %1505 = tensor.empty() : tensor<128xf32>
    %1506 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1500 : tensor<128xf32>) outs(%1505 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} {
    ^bb159(%1507: f32, %1508: f32):
      %1509 = math.rsqrt %1507 : f32
      linalg.yield %1509 : f32
    } -> tensor<128xf32>
    %1510 = tensor.empty() : tensor<1x128x28x28xf32>
    %1511 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1492, %1506 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1510 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} {
    ^bb160(%1512: f32, %1513: f32, %1514: f32):
      %1515 = arith.mulf %1512, %1513 : f32
      linalg.yield %1515 : f32
    } -> tensor<1x128x28x28xf32>
    %1516 = tensor.empty() : tensor<1x128x28x28xf32>
    %1517 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1511, %58 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1516 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} {
    ^bb161(%1518: f32, %1519: f32, %1520: f32):
      %1521 = arith.mulf %1518, %1519 : f32
      linalg.yield %1521 : f32
    } -> tensor<1x128x28x28xf32>
    %1522 = tensor.empty() : tensor<1x128x28x28xf32>
    %1523 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1517, %59 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1522 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_19", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn2"} {
    ^bb162(%1524: f32, %1525: f32, %1526: f32):
      %1527 = arith.addf %1524, %1525 : f32
      linalg.yield %1527 : f32
    } -> tensor<1x128x28x28xf32>
    %1528 = tensor.empty() : tensor<1x128x28x28xf32>
    %1529 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1523 : tensor<1x128x28x28xf32>) outs(%1528 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_17", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.relu"} {
    ^bb163(%1530: f32, %1531: f32):
      %1532 = arith.constant 0.000000e+00 : f32
      %1533 = arith.maximumf %1530, %1532 : f32
      linalg.yield %1533 : f32
    } -> tensor<1x128x28x28xf32>
    %1534 = tensor.empty() : tensor<128x1x1x1x28x28xf32>
    %1535 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1529 : tensor<1x128x28x28xf32>) outs(%1534 : tensor<128x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} {
    ^bb164(%1536: f32, %1537: f32):
      linalg.yield %1536 : f32
    } -> tensor<128x1x1x1x28x28xf32>
    %1538 = tensor.collapse_shape %1535 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<128x1x1x1x28x28xf32> into tensor<100352xf32>
    %1539 = tensor.expand_shape %1538 [[0 : i64, 1 : i64]] output_shape [128, 784] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<100352xf32> into tensor<128x784xf32>
    %1540 = tensor.collapse_shape %60 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<512x128x1x1xf32> into tensor<65536xf32>
    %1541 = tensor.expand_shape %1540 [[0 : i64, 1 : i64]] output_shape [512, 128] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<65536xf32> into tensor<512x128xf32>
    %1542 = arith.constant {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} 0.000000e+00 : f32
    %1543 = tensor.splat %1542 {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<512x784xf32>
    %1544 = linalg.matmul {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} ins(%1541, %1539 : tensor<512x128xf32>, tensor<128x784xf32>) outs(%1543 : tensor<512x784xf32>) -> tensor<512x784xf32>
    %1545 = tensor.collapse_shape %1544 [[0 : i64, 1 : i64]] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<512x784xf32> into tensor<401408xf32>
    %1546 = tensor.expand_shape %1545 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 28, 28] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<401408xf32> into tensor<512x1x28x28xf32>
    %1547 = tensor.collapse_shape %1546 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<512x1x28x28xf32> into tensor<401408xf32>
    %1548 = tensor.expand_shape %1547 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 28, 28] {prov.region_id = "conv_20", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.conv3"} : tensor<401408xf32> into tensor<1x512x28x28xf32>
    %1549 = tensor.empty() : tensor<1x512x28x28xf32>
    %1550 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1548, %221 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1549 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} {
    ^bb165(%1551: f32, %1552: f32, %1553: f32):
      %1554 = arith.subf %1551, %1552 : f32
      linalg.yield %1554 : f32
    } -> tensor<1x512x28x28xf32>
    %1555 = arith.constant {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} 1.000000e-05 : f32
    %1556 = tensor.splat %1555 {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} : tensor<512xf32>
    %1557 = tensor.empty() : tensor<512xf32>
    %1558 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%222, %1556 : tensor<512xf32>, tensor<512xf32>) outs(%1557 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} {
    ^bb166(%1559: f32, %1560: f32, %1561: f32):
      %1562 = arith.addf %1559, %1560 : f32
      linalg.yield %1562 : f32
    } -> tensor<512xf32>
    %1563 = tensor.empty() : tensor<512xf32>
    %1564 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1558 : tensor<512xf32>) outs(%1563 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} {
    ^bb167(%1565: f32, %1566: f32):
      %1567 = math.rsqrt %1565 : f32
      linalg.yield %1567 : f32
    } -> tensor<512xf32>
    %1568 = tensor.empty() : tensor<1x512x28x28xf32>
    %1569 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1550, %1564 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1568 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} {
    ^bb168(%1570: f32, %1571: f32, %1572: f32):
      %1573 = arith.mulf %1570, %1571 : f32
      linalg.yield %1573 : f32
    } -> tensor<1x512x28x28xf32>
    %1574 = tensor.empty() : tensor<1x512x28x28xf32>
    %1575 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1569, %61 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1574 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} {
    ^bb169(%1576: f32, %1577: f32, %1578: f32):
      %1579 = arith.mulf %1576, %1577 : f32
      linalg.yield %1579 : f32
    } -> tensor<1x512x28x28xf32>
    %1580 = tensor.empty() : tensor<1x512x28x28xf32>
    %1581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1575, %62 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1580 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_20", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.bn3"} {
    ^bb170(%1582: f32, %1583: f32, %1584: f32):
      %1585 = arith.addf %1582, %1583 : f32
      linalg.yield %1585 : f32
    } -> tensor<1x512x28x28xf32>
    %1586 = tensor.empty() : tensor<1x512x28x28xf32>
    %1587 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1581, %1410 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%1586 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "add_5", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2"} {
    ^bb171(%1588: f32, %1589: f32, %1590: f32):
      %1591 = arith.addf %1588, %1589 : f32
      linalg.yield %1591 : f32
    } -> tensor<1x512x28x28xf32>
    %1592 = tensor.empty() : tensor<1x512x28x28xf32>
    %1593 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1587 : tensor<1x512x28x28xf32>) outs(%1592 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "minmax_18", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.2.relu"} {
    ^bb172(%1594: f32, %1595: f32):
      %1596 = arith.constant 0.000000e+00 : f32
      %1597 = arith.maximumf %1594, %1596 : f32
      linalg.yield %1597 : f32
    } -> tensor<1x512x28x28xf32>
    %1598 = tensor.empty() : tensor<512x1x1x1x28x28xf32>
    %1599 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1593 : tensor<1x512x28x28xf32>) outs(%1598 : tensor<512x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} {
    ^bb173(%1600: f32, %1601: f32):
      linalg.yield %1600 : f32
    } -> tensor<512x1x1x1x28x28xf32>
    %1602 = tensor.collapse_shape %1599 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<512x1x1x1x28x28xf32> into tensor<401408xf32>
    %1603 = tensor.expand_shape %1602 [[0 : i64, 1 : i64]] output_shape [512, 784] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<401408xf32> into tensor<512x784xf32>
    %1604 = tensor.collapse_shape %63 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<128x512x1x1xf32> into tensor<65536xf32>
    %1605 = tensor.expand_shape %1604 [[0 : i64, 1 : i64]] output_shape [128, 512] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<65536xf32> into tensor<128x512xf32>
    %1606 = arith.constant {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} 0.000000e+00 : f32
    %1607 = tensor.splat %1606 {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<128x784xf32>
    %1608 = linalg.matmul {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} ins(%1605, %1603 : tensor<128x512xf32>, tensor<512x784xf32>) outs(%1607 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1609 = tensor.collapse_shape %1608 [[0 : i64, 1 : i64]] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<128x784xf32> into tensor<100352xf32>
    %1610 = tensor.expand_shape %1609 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1611 = tensor.collapse_shape %1610 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1612 = tensor.expand_shape %1611 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_21", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv1"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1613 = tensor.empty() : tensor<1x128x28x28xf32>
    %1614 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1612, %224 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1613 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} {
    ^bb174(%1615: f32, %1616: f32, %1617: f32):
      %1618 = arith.subf %1615, %1616 : f32
      linalg.yield %1618 : f32
    } -> tensor<1x128x28x28xf32>
    %1619 = arith.constant {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} 1.000000e-05 : f32
    %1620 = tensor.splat %1619 {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} : tensor<128xf32>
    %1621 = tensor.empty() : tensor<128xf32>
    %1622 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%225, %1620 : tensor<128xf32>, tensor<128xf32>) outs(%1621 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} {
    ^bb175(%1623: f32, %1624: f32, %1625: f32):
      %1626 = arith.addf %1623, %1624 : f32
      linalg.yield %1626 : f32
    } -> tensor<128xf32>
    %1627 = tensor.empty() : tensor<128xf32>
    %1628 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1622 : tensor<128xf32>) outs(%1627 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} {
    ^bb176(%1629: f32, %1630: f32):
      %1631 = math.rsqrt %1629 : f32
      linalg.yield %1631 : f32
    } -> tensor<128xf32>
    %1632 = tensor.empty() : tensor<1x128x28x28xf32>
    %1633 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1614, %1628 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1632 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} {
    ^bb177(%1634: f32, %1635: f32, %1636: f32):
      %1637 = arith.mulf %1634, %1635 : f32
      linalg.yield %1637 : f32
    } -> tensor<1x128x28x28xf32>
    %1638 = tensor.empty() : tensor<1x128x28x28xf32>
    %1639 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1633, %64 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1638 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} {
    ^bb178(%1640: f32, %1641: f32, %1642: f32):
      %1643 = arith.mulf %1640, %1641 : f32
      linalg.yield %1643 : f32
    } -> tensor<1x128x28x28xf32>
    %1644 = tensor.empty() : tensor<1x128x28x28xf32>
    %1645 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1639, %65 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1644 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_21", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn1"} {
    ^bb179(%1646: f32, %1647: f32, %1648: f32):
      %1649 = arith.addf %1646, %1647 : f32
      linalg.yield %1649 : f32
    } -> tensor<1x128x28x28xf32>
    %1650 = tensor.empty() : tensor<1x128x28x28xf32>
    %1651 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1645 : tensor<1x128x28x28xf32>) outs(%1650 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_19", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.relu"} {
    ^bb180(%1652: f32, %1653: f32):
      %1654 = arith.constant 0.000000e+00 : f32
      %1655 = arith.maximumf %1652, %1654 : f32
      linalg.yield %1655 : f32
    } -> tensor<1x128x28x28xf32>
    %1656 = arith.constant {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} 0.000000e+00 : f32
    %1657 = tensor.splat %1656 {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<1x128x30x30xf32>
    %1658 = "tensor.insert_slice"(%1651, %1657) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 128, 28, 28>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : (tensor<1x128x28x28xf32>, tensor<1x128x30x30xf32>) -> tensor<1x128x30x30xf32>
    %1659 = tensor.empty() : tensor<128x3x3x1x28x28xf32>
    %1660 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1658 : tensor<1x128x30x30xf32>) outs(%1659 : tensor<128x3x3x1x28x28xf32>) attrs =  {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} {
    ^bb181(%1661: f32, %1662: f32):
      linalg.yield %1661 : f32
    } -> tensor<128x3x3x1x28x28xf32>
    %1663 = tensor.collapse_shape %1660 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<128x3x3x1x28x28xf32> into tensor<903168xf32>
    %1664 = tensor.expand_shape %1663 [[0 : i64, 1 : i64]] output_shape [1152, 784] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<903168xf32> into tensor<1152x784xf32>
    %1665 = tensor.collapse_shape %66 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<128x128x3x3xf32> into tensor<147456xf32>
    %1666 = tensor.expand_shape %1665 [[0 : i64, 1 : i64]] output_shape [128, 1152] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<147456xf32> into tensor<128x1152xf32>
    %1667 = arith.constant {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} 0.000000e+00 : f32
    %1668 = tensor.splat %1667 {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<128x784xf32>
    %1669 = linalg.matmul {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} ins(%1666, %1664 : tensor<128x1152xf32>, tensor<1152x784xf32>) outs(%1668 : tensor<128x784xf32>) -> tensor<128x784xf32>
    %1670 = tensor.collapse_shape %1669 [[0 : i64, 1 : i64]] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<128x784xf32> into tensor<100352xf32>
    %1671 = tensor.expand_shape %1670 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [128, 1, 28, 28] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<100352xf32> into tensor<128x1x28x28xf32>
    %1672 = tensor.collapse_shape %1671 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<128x1x28x28xf32> into tensor<100352xf32>
    %1673 = tensor.expand_shape %1672 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 128, 28, 28] {prov.region_id = "conv_22", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv2"} : tensor<100352xf32> into tensor<1x128x28x28xf32>
    %1674 = tensor.empty() : tensor<1x128x28x28xf32>
    %1675 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1673, %227 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1674 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} {
    ^bb182(%1676: f32, %1677: f32, %1678: f32):
      %1679 = arith.subf %1676, %1677 : f32
      linalg.yield %1679 : f32
    } -> tensor<1x128x28x28xf32>
    %1680 = arith.constant {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} 1.000000e-05 : f32
    %1681 = tensor.splat %1680 {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} : tensor<128xf32>
    %1682 = tensor.empty() : tensor<128xf32>
    %1683 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%228, %1681 : tensor<128xf32>, tensor<128xf32>) outs(%1682 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} {
    ^bb183(%1684: f32, %1685: f32, %1686: f32):
      %1687 = arith.addf %1684, %1685 : f32
      linalg.yield %1687 : f32
    } -> tensor<128xf32>
    %1688 = tensor.empty() : tensor<128xf32>
    %1689 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1683 : tensor<128xf32>) outs(%1688 : tensor<128xf32>) attrs =  {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} {
    ^bb184(%1690: f32, %1691: f32):
      %1692 = math.rsqrt %1690 : f32
      linalg.yield %1692 : f32
    } -> tensor<128xf32>
    %1693 = tensor.empty() : tensor<1x128x28x28xf32>
    %1694 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1675, %1689 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1693 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} {
    ^bb185(%1695: f32, %1696: f32, %1697: f32):
      %1698 = arith.mulf %1695, %1696 : f32
      linalg.yield %1698 : f32
    } -> tensor<1x128x28x28xf32>
    %1699 = tensor.empty() : tensor<1x128x28x28xf32>
    %1700 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1694, %67 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1699 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} {
    ^bb186(%1701: f32, %1702: f32, %1703: f32):
      %1704 = arith.mulf %1701, %1702 : f32
      linalg.yield %1704 : f32
    } -> tensor<1x128x28x28xf32>
    %1705 = tensor.empty() : tensor<1x128x28x28xf32>
    %1706 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1700, %68 : tensor<1x128x28x28xf32>, tensor<128xf32>) outs(%1705 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "batch_norm_22", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn2"} {
    ^bb187(%1707: f32, %1708: f32, %1709: f32):
      %1710 = arith.addf %1707, %1708 : f32
      linalg.yield %1710 : f32
    } -> tensor<1x128x28x28xf32>
    %1711 = tensor.empty() : tensor<1x128x28x28xf32>
    %1712 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1706 : tensor<1x128x28x28xf32>) outs(%1711 : tensor<1x128x28x28xf32>) attrs =  {prov.region_id = "minmax_20", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.relu"} {
    ^bb188(%1713: f32, %1714: f32):
      %1715 = arith.constant 0.000000e+00 : f32
      %1716 = arith.maximumf %1713, %1715 : f32
      linalg.yield %1716 : f32
    } -> tensor<1x128x28x28xf32>
    %1717 = tensor.empty() : tensor<128x1x1x1x28x28xf32>
    %1718 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1712 : tensor<1x128x28x28xf32>) outs(%1717 : tensor<128x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} {
    ^bb189(%1719: f32, %1720: f32):
      linalg.yield %1719 : f32
    } -> tensor<128x1x1x1x28x28xf32>
    %1721 = tensor.collapse_shape %1718 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<128x1x1x1x28x28xf32> into tensor<100352xf32>
    %1722 = tensor.expand_shape %1721 [[0 : i64, 1 : i64]] output_shape [128, 784] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<100352xf32> into tensor<128x784xf32>
    %1723 = tensor.collapse_shape %69 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<512x128x1x1xf32> into tensor<65536xf32>
    %1724 = tensor.expand_shape %1723 [[0 : i64, 1 : i64]] output_shape [512, 128] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<65536xf32> into tensor<512x128xf32>
    %1725 = arith.constant {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} 0.000000e+00 : f32
    %1726 = tensor.splat %1725 {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<512x784xf32>
    %1727 = linalg.matmul {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} ins(%1724, %1722 : tensor<512x128xf32>, tensor<128x784xf32>) outs(%1726 : tensor<512x784xf32>) -> tensor<512x784xf32>
    %1728 = tensor.collapse_shape %1727 [[0 : i64, 1 : i64]] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<512x784xf32> into tensor<401408xf32>
    %1729 = tensor.expand_shape %1728 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 28, 28] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<401408xf32> into tensor<512x1x28x28xf32>
    %1730 = tensor.collapse_shape %1729 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<512x1x28x28xf32> into tensor<401408xf32>
    %1731 = tensor.expand_shape %1730 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 28, 28] {prov.region_id = "conv_23", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.conv3"} : tensor<401408xf32> into tensor<1x512x28x28xf32>
    %1732 = tensor.empty() : tensor<1x512x28x28xf32>
    %1733 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1731, %230 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1732 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} {
    ^bb190(%1734: f32, %1735: f32, %1736: f32):
      %1737 = arith.subf %1734, %1735 : f32
      linalg.yield %1737 : f32
    } -> tensor<1x512x28x28xf32>
    %1738 = arith.constant {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} 1.000000e-05 : f32
    %1739 = tensor.splat %1738 {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} : tensor<512xf32>
    %1740 = tensor.empty() : tensor<512xf32>
    %1741 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%231, %1739 : tensor<512xf32>, tensor<512xf32>) outs(%1740 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} {
    ^bb191(%1742: f32, %1743: f32, %1744: f32):
      %1745 = arith.addf %1742, %1743 : f32
      linalg.yield %1745 : f32
    } -> tensor<512xf32>
    %1746 = tensor.empty() : tensor<512xf32>
    %1747 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1741 : tensor<512xf32>) outs(%1746 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} {
    ^bb192(%1748: f32, %1749: f32):
      %1750 = math.rsqrt %1748 : f32
      linalg.yield %1750 : f32
    } -> tensor<512xf32>
    %1751 = tensor.empty() : tensor<1x512x28x28xf32>
    %1752 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1733, %1747 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1751 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} {
    ^bb193(%1753: f32, %1754: f32, %1755: f32):
      %1756 = arith.mulf %1753, %1754 : f32
      linalg.yield %1756 : f32
    } -> tensor<1x512x28x28xf32>
    %1757 = tensor.empty() : tensor<1x512x28x28xf32>
    %1758 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1752, %70 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1757 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} {
    ^bb194(%1759: f32, %1760: f32, %1761: f32):
      %1762 = arith.mulf %1759, %1760 : f32
      linalg.yield %1762 : f32
    } -> tensor<1x512x28x28xf32>
    %1763 = tensor.empty() : tensor<1x512x28x28xf32>
    %1764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1758, %71 : tensor<1x512x28x28xf32>, tensor<512xf32>) outs(%1763 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "batch_norm_23", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.bn3"} {
    ^bb195(%1765: f32, %1766: f32, %1767: f32):
      %1768 = arith.addf %1765, %1766 : f32
      linalg.yield %1768 : f32
    } -> tensor<1x512x28x28xf32>
    %1769 = tensor.empty() : tensor<1x512x28x28xf32>
    %1770 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1764, %1593 : tensor<1x512x28x28xf32>, tensor<1x512x28x28xf32>) outs(%1769 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "add_6", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3"} {
    ^bb196(%1771: f32, %1772: f32, %1773: f32):
      %1774 = arith.addf %1771, %1772 : f32
      linalg.yield %1774 : f32
    } -> tensor<1x512x28x28xf32>
    %1775 = tensor.empty() : tensor<1x512x28x28xf32>
    %1776 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1770 : tensor<1x512x28x28xf32>) outs(%1775 : tensor<1x512x28x28xf32>) attrs =  {prov.region_id = "minmax_21", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer2.3.relu"} {
    ^bb197(%1777: f32, %1778: f32):
      %1779 = arith.constant 0.000000e+00 : f32
      %1780 = arith.maximumf %1777, %1779 : f32
      linalg.yield %1780 : f32
    } -> tensor<1x512x28x28xf32>
    %1781 = tensor.empty() : tensor<512x1x1x1x28x28xf32>
    %1782 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1776 : tensor<1x512x28x28xf32>) outs(%1781 : tensor<512x1x1x1x28x28xf32>) attrs =  {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} {
    ^bb198(%1783: f32, %1784: f32):
      linalg.yield %1783 : f32
    } -> tensor<512x1x1x1x28x28xf32>
    %1785 = tensor.collapse_shape %1782 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<512x1x1x1x28x28xf32> into tensor<401408xf32>
    %1786 = tensor.expand_shape %1785 [[0 : i64, 1 : i64]] output_shape [512, 784] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<401408xf32> into tensor<512x784xf32>
    %1787 = tensor.collapse_shape %72 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<256x512x1x1xf32> into tensor<131072xf32>
    %1788 = tensor.expand_shape %1787 [[0 : i64, 1 : i64]] output_shape [256, 512] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<131072xf32> into tensor<256x512xf32>
    %1789 = arith.constant {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} 0.000000e+00 : f32
    %1790 = tensor.splat %1789 {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<256x784xf32>
    %1791 = linalg.matmul {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} ins(%1788, %1786 : tensor<256x512xf32>, tensor<512x784xf32>) outs(%1790 : tensor<256x784xf32>) -> tensor<256x784xf32>
    %1792 = tensor.collapse_shape %1791 [[0 : i64, 1 : i64]] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<256x784xf32> into tensor<200704xf32>
    %1793 = tensor.expand_shape %1792 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 28, 28] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<200704xf32> into tensor<256x1x28x28xf32>
    %1794 = tensor.collapse_shape %1793 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<256x1x28x28xf32> into tensor<200704xf32>
    %1795 = tensor.expand_shape %1794 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 28, 28] {prov.region_id = "conv_24", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv1"} : tensor<200704xf32> into tensor<1x256x28x28xf32>
    %1796 = tensor.empty() : tensor<1x256x28x28xf32>
    %1797 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1795, %233 : tensor<1x256x28x28xf32>, tensor<256xf32>) outs(%1796 : tensor<1x256x28x28xf32>) attrs =  {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} {
    ^bb199(%1798: f32, %1799: f32, %1800: f32):
      %1801 = arith.subf %1798, %1799 : f32
      linalg.yield %1801 : f32
    } -> tensor<1x256x28x28xf32>
    %1802 = arith.constant {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} 1.000000e-05 : f32
    %1803 = tensor.splat %1802 {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} : tensor<256xf32>
    %1804 = tensor.empty() : tensor<256xf32>
    %1805 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%234, %1803 : tensor<256xf32>, tensor<256xf32>) outs(%1804 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} {
    ^bb200(%1806: f32, %1807: f32, %1808: f32):
      %1809 = arith.addf %1806, %1807 : f32
      linalg.yield %1809 : f32
    } -> tensor<256xf32>
    %1810 = tensor.empty() : tensor<256xf32>
    %1811 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1805 : tensor<256xf32>) outs(%1810 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} {
    ^bb201(%1812: f32, %1813: f32):
      %1814 = math.rsqrt %1812 : f32
      linalg.yield %1814 : f32
    } -> tensor<256xf32>
    %1815 = tensor.empty() : tensor<1x256x28x28xf32>
    %1816 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1797, %1811 : tensor<1x256x28x28xf32>, tensor<256xf32>) outs(%1815 : tensor<1x256x28x28xf32>) attrs =  {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} {
    ^bb202(%1817: f32, %1818: f32, %1819: f32):
      %1820 = arith.mulf %1817, %1818 : f32
      linalg.yield %1820 : f32
    } -> tensor<1x256x28x28xf32>
    %1821 = tensor.empty() : tensor<1x256x28x28xf32>
    %1822 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1816, %73 : tensor<1x256x28x28xf32>, tensor<256xf32>) outs(%1821 : tensor<1x256x28x28xf32>) attrs =  {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} {
    ^bb203(%1823: f32, %1824: f32, %1825: f32):
      %1826 = arith.mulf %1823, %1824 : f32
      linalg.yield %1826 : f32
    } -> tensor<1x256x28x28xf32>
    %1827 = tensor.empty() : tensor<1x256x28x28xf32>
    %1828 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1822, %74 : tensor<1x256x28x28xf32>, tensor<256xf32>) outs(%1827 : tensor<1x256x28x28xf32>) attrs =  {prov.region_id = "batch_norm_24", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn1"} {
    ^bb204(%1829: f32, %1830: f32, %1831: f32):
      %1832 = arith.addf %1829, %1830 : f32
      linalg.yield %1832 : f32
    } -> tensor<1x256x28x28xf32>
    %1833 = tensor.empty() : tensor<1x256x28x28xf32>
    %1834 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1828 : tensor<1x256x28x28xf32>) outs(%1833 : tensor<1x256x28x28xf32>) attrs =  {prov.region_id = "minmax_22", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.relu"} {
    ^bb205(%1835: f32, %1836: f32):
      %1837 = arith.constant 0.000000e+00 : f32
      %1838 = arith.maximumf %1835, %1837 : f32
      linalg.yield %1838 : f32
    } -> tensor<1x256x28x28xf32>
    %1839 = arith.constant {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} 0.000000e+00 : f32
    %1840 = tensor.splat %1839 {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<1x256x30x30xf32>
    %1841 = "tensor.insert_slice"(%1834, %1840) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 28, 28>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : (tensor<1x256x28x28xf32>, tensor<1x256x30x30xf32>) -> tensor<1x256x30x30xf32>
    %1842 = tensor.empty() : tensor<256x3x3x1x14x14xf32>
    %1843 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1841 : tensor<1x256x30x30xf32>) outs(%1842 : tensor<256x3x3x1x14x14xf32>) attrs =  {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} {
    ^bb206(%1844: f32, %1845: f32):
      linalg.yield %1844 : f32
    } -> tensor<256x3x3x1x14x14xf32>
    %1846 = tensor.collapse_shape %1843 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<256x3x3x1x14x14xf32> into tensor<451584xf32>
    %1847 = tensor.expand_shape %1846 [[0 : i64, 1 : i64]] output_shape [2304, 196] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<451584xf32> into tensor<2304x196xf32>
    %1848 = tensor.collapse_shape %75 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<256x256x3x3xf32> into tensor<589824xf32>
    %1849 = tensor.expand_shape %1848 [[0 : i64, 1 : i64]] output_shape [256, 2304] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<589824xf32> into tensor<256x2304xf32>
    %1850 = arith.constant {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} 0.000000e+00 : f32
    %1851 = tensor.splat %1850 {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<256x196xf32>
    %1852 = linalg.matmul {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} ins(%1849, %1847 : tensor<256x2304xf32>, tensor<2304x196xf32>) outs(%1851 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %1853 = tensor.collapse_shape %1852 [[0 : i64, 1 : i64]] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<256x196xf32> into tensor<50176xf32>
    %1854 = tensor.expand_shape %1853 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %1855 = tensor.collapse_shape %1854 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %1856 = tensor.expand_shape %1855 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_25", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv2"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %1857 = tensor.empty() : tensor<1x256x14x14xf32>
    %1858 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1856, %236 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%1857 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} {
    ^bb207(%1859: f32, %1860: f32, %1861: f32):
      %1862 = arith.subf %1859, %1860 : f32
      linalg.yield %1862 : f32
    } -> tensor<1x256x14x14xf32>
    %1863 = arith.constant {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} 1.000000e-05 : f32
    %1864 = tensor.splat %1863 {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} : tensor<256xf32>
    %1865 = tensor.empty() : tensor<256xf32>
    %1866 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%237, %1864 : tensor<256xf32>, tensor<256xf32>) outs(%1865 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} {
    ^bb208(%1867: f32, %1868: f32, %1869: f32):
      %1870 = arith.addf %1867, %1868 : f32
      linalg.yield %1870 : f32
    } -> tensor<256xf32>
    %1871 = tensor.empty() : tensor<256xf32>
    %1872 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1866 : tensor<256xf32>) outs(%1871 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} {
    ^bb209(%1873: f32, %1874: f32):
      %1875 = math.rsqrt %1873 : f32
      linalg.yield %1875 : f32
    } -> tensor<256xf32>
    %1876 = tensor.empty() : tensor<1x256x14x14xf32>
    %1877 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1858, %1872 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%1876 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} {
    ^bb210(%1878: f32, %1879: f32, %1880: f32):
      %1881 = arith.mulf %1878, %1879 : f32
      linalg.yield %1881 : f32
    } -> tensor<1x256x14x14xf32>
    %1882 = tensor.empty() : tensor<1x256x14x14xf32>
    %1883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1877, %76 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%1882 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} {
    ^bb211(%1884: f32, %1885: f32, %1886: f32):
      %1887 = arith.mulf %1884, %1885 : f32
      linalg.yield %1887 : f32
    } -> tensor<1x256x14x14xf32>
    %1888 = tensor.empty() : tensor<1x256x14x14xf32>
    %1889 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1883, %77 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%1888 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_25", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn2"} {
    ^bb212(%1890: f32, %1891: f32, %1892: f32):
      %1893 = arith.addf %1890, %1891 : f32
      linalg.yield %1893 : f32
    } -> tensor<1x256x14x14xf32>
    %1894 = tensor.empty() : tensor<1x256x14x14xf32>
    %1895 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1889 : tensor<1x256x14x14xf32>) outs(%1894 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_23", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.relu"} {
    ^bb213(%1896: f32, %1897: f32):
      %1898 = arith.constant 0.000000e+00 : f32
      %1899 = arith.maximumf %1896, %1898 : f32
      linalg.yield %1899 : f32
    } -> tensor<1x256x14x14xf32>
    %1900 = tensor.empty() : tensor<256x1x1x1x14x14xf32>
    %1901 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1895 : tensor<1x256x14x14xf32>) outs(%1900 : tensor<256x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} {
    ^bb214(%1902: f32, %1903: f32):
      linalg.yield %1902 : f32
    } -> tensor<256x1x1x1x14x14xf32>
    %1904 = tensor.collapse_shape %1901 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<256x1x1x1x14x14xf32> into tensor<50176xf32>
    %1905 = tensor.expand_shape %1904 [[0 : i64, 1 : i64]] output_shape [256, 196] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<50176xf32> into tensor<256x196xf32>
    %1906 = tensor.collapse_shape %78 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<1024x256x1x1xf32> into tensor<262144xf32>
    %1907 = tensor.expand_shape %1906 [[0 : i64, 1 : i64]] output_shape [1024, 256] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<262144xf32> into tensor<1024x256xf32>
    %1908 = arith.constant {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} 0.000000e+00 : f32
    %1909 = tensor.splat %1908 {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<1024x196xf32>
    %1910 = linalg.matmul {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} ins(%1907, %1905 : tensor<1024x256xf32>, tensor<256x196xf32>) outs(%1909 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %1911 = tensor.collapse_shape %1910 [[0 : i64, 1 : i64]] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<1024x196xf32> into tensor<200704xf32>
    %1912 = tensor.expand_shape %1911 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %1913 = tensor.collapse_shape %1912 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %1914 = tensor.expand_shape %1913 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_26", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.conv3"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %1915 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1916 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1914, %239 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1915 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} {
    ^bb215(%1917: f32, %1918: f32, %1919: f32):
      %1920 = arith.subf %1917, %1918 : f32
      linalg.yield %1920 : f32
    } -> tensor<1x1024x14x14xf32>
    %1921 = arith.constant {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} 1.000000e-05 : f32
    %1922 = tensor.splat %1921 {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} : tensor<1024xf32>
    %1923 = tensor.empty() : tensor<1024xf32>
    %1924 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%240, %1922 : tensor<1024xf32>, tensor<1024xf32>) outs(%1923 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} {
    ^bb216(%1925: f32, %1926: f32, %1927: f32):
      %1928 = arith.addf %1925, %1926 : f32
      linalg.yield %1928 : f32
    } -> tensor<1024xf32>
    %1929 = tensor.empty() : tensor<1024xf32>
    %1930 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1924 : tensor<1024xf32>) outs(%1929 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} {
    ^bb217(%1931: f32, %1932: f32):
      %1933 = math.rsqrt %1931 : f32
      linalg.yield %1933 : f32
    } -> tensor<1024xf32>
    %1934 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1935 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1916, %1930 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1934 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} {
    ^bb218(%1936: f32, %1937: f32, %1938: f32):
      %1939 = arith.mulf %1936, %1937 : f32
      linalg.yield %1939 : f32
    } -> tensor<1x1024x14x14xf32>
    %1940 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1941 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1935, %79 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1940 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} {
    ^bb219(%1942: f32, %1943: f32, %1944: f32):
      %1945 = arith.mulf %1942, %1943 : f32
      linalg.yield %1945 : f32
    } -> tensor<1x1024x14x14xf32>
    %1946 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1947 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1941, %80 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1946 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_26", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.bn3"} {
    ^bb220(%1948: f32, %1949: f32, %1950: f32):
      %1951 = arith.addf %1948, %1949 : f32
      linalg.yield %1951 : f32
    } -> tensor<1x1024x14x14xf32>
    %1952 = tensor.empty() : tensor<512x1x1x1x14x14xf32>
    %1953 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1776 : tensor<1x512x28x28xf32>) outs(%1952 : tensor<512x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} {
    ^bb221(%1954: f32, %1955: f32):
      linalg.yield %1954 : f32
    } -> tensor<512x1x1x1x14x14xf32>
    %1956 = tensor.collapse_shape %1953 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<512x1x1x1x14x14xf32> into tensor<100352xf32>
    %1957 = tensor.expand_shape %1956 [[0 : i64, 1 : i64]] output_shape [512, 196] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<100352xf32> into tensor<512x196xf32>
    %1958 = tensor.collapse_shape %81 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<1024x512x1x1xf32> into tensor<524288xf32>
    %1959 = tensor.expand_shape %1958 [[0 : i64, 1 : i64]] output_shape [1024, 512] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<524288xf32> into tensor<1024x512xf32>
    %1960 = arith.constant {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} 0.000000e+00 : f32
    %1961 = tensor.splat %1960 {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<1024x196xf32>
    %1962 = linalg.matmul {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} ins(%1959, %1957 : tensor<1024x512xf32>, tensor<512x196xf32>) outs(%1961 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %1963 = tensor.collapse_shape %1962 [[0 : i64, 1 : i64]] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<1024x196xf32> into tensor<200704xf32>
    %1964 = tensor.expand_shape %1963 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %1965 = tensor.collapse_shape %1964 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %1966 = tensor.expand_shape %1965 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_27", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.0"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %1967 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1968 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1966, %242 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1967 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} {
    ^bb222(%1969: f32, %1970: f32, %1971: f32):
      %1972 = arith.subf %1969, %1970 : f32
      linalg.yield %1972 : f32
    } -> tensor<1x1024x14x14xf32>
    %1973 = arith.constant {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} 1.000000e-05 : f32
    %1974 = tensor.splat %1973 {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} : tensor<1024xf32>
    %1975 = tensor.empty() : tensor<1024xf32>
    %1976 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%243, %1974 : tensor<1024xf32>, tensor<1024xf32>) outs(%1975 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} {
    ^bb223(%1977: f32, %1978: f32, %1979: f32):
      %1980 = arith.addf %1977, %1978 : f32
      linalg.yield %1980 : f32
    } -> tensor<1024xf32>
    %1981 = tensor.empty() : tensor<1024xf32>
    %1982 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1976 : tensor<1024xf32>) outs(%1981 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} {
    ^bb224(%1983: f32, %1984: f32):
      %1985 = math.rsqrt %1983 : f32
      linalg.yield %1985 : f32
    } -> tensor<1024xf32>
    %1986 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1987 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1968, %1982 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1986 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} {
    ^bb225(%1988: f32, %1989: f32, %1990: f32):
      %1991 = arith.mulf %1988, %1989 : f32
      linalg.yield %1991 : f32
    } -> tensor<1x1024x14x14xf32>
    %1992 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1987, %82 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1992 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} {
    ^bb226(%1994: f32, %1995: f32, %1996: f32):
      %1997 = arith.mulf %1994, %1995 : f32
      linalg.yield %1997 : f32
    } -> tensor<1x1024x14x14xf32>
    %1998 = tensor.empty() : tensor<1x1024x14x14xf32>
    %1999 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1993, %83 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%1998 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_27", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.downsample.1"} {
    ^bb227(%2000: f32, %2001: f32, %2002: f32):
      %2003 = arith.addf %2000, %2001 : f32
      linalg.yield %2003 : f32
    } -> tensor<1x1024x14x14xf32>
    %2004 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2005 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1947, %1999 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%2004 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "add_7", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0"} {
    ^bb228(%2006: f32, %2007: f32, %2008: f32):
      %2009 = arith.addf %2006, %2007 : f32
      linalg.yield %2009 : f32
    } -> tensor<1x1024x14x14xf32>
    %2010 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2011 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2005 : tensor<1x1024x14x14xf32>) outs(%2010 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "minmax_24", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.0.relu"} {
    ^bb229(%2012: f32, %2013: f32):
      %2014 = arith.constant 0.000000e+00 : f32
      %2015 = arith.maximumf %2012, %2014 : f32
      linalg.yield %2015 : f32
    } -> tensor<1x1024x14x14xf32>
    %2016 = tensor.empty() : tensor<1024x1x1x1x14x14xf32>
    %2017 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2011 : tensor<1x1024x14x14xf32>) outs(%2016 : tensor<1024x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} {
    ^bb230(%2018: f32, %2019: f32):
      linalg.yield %2018 : f32
    } -> tensor<1024x1x1x1x14x14xf32>
    %2020 = tensor.collapse_shape %2017 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<1024x1x1x1x14x14xf32> into tensor<200704xf32>
    %2021 = tensor.expand_shape %2020 [[0 : i64, 1 : i64]] output_shape [1024, 196] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<200704xf32> into tensor<1024x196xf32>
    %2022 = tensor.collapse_shape %84 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<256x1024x1x1xf32> into tensor<262144xf32>
    %2023 = tensor.expand_shape %2022 [[0 : i64, 1 : i64]] output_shape [256, 1024] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<262144xf32> into tensor<256x1024xf32>
    %2024 = arith.constant {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} 0.000000e+00 : f32
    %2025 = tensor.splat %2024 {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<256x196xf32>
    %2026 = linalg.matmul {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} ins(%2023, %2021 : tensor<256x1024xf32>, tensor<1024x196xf32>) outs(%2025 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2027 = tensor.collapse_shape %2026 [[0 : i64, 1 : i64]] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<256x196xf32> into tensor<50176xf32>
    %2028 = tensor.expand_shape %2027 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2029 = tensor.collapse_shape %2028 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2030 = tensor.expand_shape %2029 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_28", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv1"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2031 = tensor.empty() : tensor<1x256x14x14xf32>
    %2032 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2030, %245 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2031 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} {
    ^bb231(%2033: f32, %2034: f32, %2035: f32):
      %2036 = arith.subf %2033, %2034 : f32
      linalg.yield %2036 : f32
    } -> tensor<1x256x14x14xf32>
    %2037 = arith.constant {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} 1.000000e-05 : f32
    %2038 = tensor.splat %2037 {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} : tensor<256xf32>
    %2039 = tensor.empty() : tensor<256xf32>
    %2040 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%246, %2038 : tensor<256xf32>, tensor<256xf32>) outs(%2039 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} {
    ^bb232(%2041: f32, %2042: f32, %2043: f32):
      %2044 = arith.addf %2041, %2042 : f32
      linalg.yield %2044 : f32
    } -> tensor<256xf32>
    %2045 = tensor.empty() : tensor<256xf32>
    %2046 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2040 : tensor<256xf32>) outs(%2045 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} {
    ^bb233(%2047: f32, %2048: f32):
      %2049 = math.rsqrt %2047 : f32
      linalg.yield %2049 : f32
    } -> tensor<256xf32>
    %2050 = tensor.empty() : tensor<1x256x14x14xf32>
    %2051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2032, %2046 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2050 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} {
    ^bb234(%2052: f32, %2053: f32, %2054: f32):
      %2055 = arith.mulf %2052, %2053 : f32
      linalg.yield %2055 : f32
    } -> tensor<1x256x14x14xf32>
    %2056 = tensor.empty() : tensor<1x256x14x14xf32>
    %2057 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2051, %85 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2056 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} {
    ^bb235(%2058: f32, %2059: f32, %2060: f32):
      %2061 = arith.mulf %2058, %2059 : f32
      linalg.yield %2061 : f32
    } -> tensor<1x256x14x14xf32>
    %2062 = tensor.empty() : tensor<1x256x14x14xf32>
    %2063 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2057, %86 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2062 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_28", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn1"} {
    ^bb236(%2064: f32, %2065: f32, %2066: f32):
      %2067 = arith.addf %2064, %2065 : f32
      linalg.yield %2067 : f32
    } -> tensor<1x256x14x14xf32>
    %2068 = tensor.empty() : tensor<1x256x14x14xf32>
    %2069 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2063 : tensor<1x256x14x14xf32>) outs(%2068 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_25", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.relu"} {
    ^bb237(%2070: f32, %2071: f32):
      %2072 = arith.constant 0.000000e+00 : f32
      %2073 = arith.maximumf %2070, %2072 : f32
      linalg.yield %2073 : f32
    } -> tensor<1x256x14x14xf32>
    %2074 = arith.constant {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} 0.000000e+00 : f32
    %2075 = tensor.splat %2074 {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<1x256x16x16xf32>
    %2076 = "tensor.insert_slice"(%2069, %2075) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 14, 14>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : (tensor<1x256x14x14xf32>, tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %2077 = tensor.empty() : tensor<256x3x3x1x14x14xf32>
    %2078 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2076 : tensor<1x256x16x16xf32>) outs(%2077 : tensor<256x3x3x1x14x14xf32>) attrs =  {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} {
    ^bb238(%2079: f32, %2080: f32):
      linalg.yield %2079 : f32
    } -> tensor<256x3x3x1x14x14xf32>
    %2081 = tensor.collapse_shape %2078 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<256x3x3x1x14x14xf32> into tensor<451584xf32>
    %2082 = tensor.expand_shape %2081 [[0 : i64, 1 : i64]] output_shape [2304, 196] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<451584xf32> into tensor<2304x196xf32>
    %2083 = tensor.collapse_shape %87 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<256x256x3x3xf32> into tensor<589824xf32>
    %2084 = tensor.expand_shape %2083 [[0 : i64, 1 : i64]] output_shape [256, 2304] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<589824xf32> into tensor<256x2304xf32>
    %2085 = arith.constant {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} 0.000000e+00 : f32
    %2086 = tensor.splat %2085 {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<256x196xf32>
    %2087 = linalg.matmul {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} ins(%2084, %2082 : tensor<256x2304xf32>, tensor<2304x196xf32>) outs(%2086 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2088 = tensor.collapse_shape %2087 [[0 : i64, 1 : i64]] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<256x196xf32> into tensor<50176xf32>
    %2089 = tensor.expand_shape %2088 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2090 = tensor.collapse_shape %2089 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2091 = tensor.expand_shape %2090 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_29", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv2"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2092 = tensor.empty() : tensor<1x256x14x14xf32>
    %2093 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2091, %248 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2092 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} {
    ^bb239(%2094: f32, %2095: f32, %2096: f32):
      %2097 = arith.subf %2094, %2095 : f32
      linalg.yield %2097 : f32
    } -> tensor<1x256x14x14xf32>
    %2098 = arith.constant {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} 1.000000e-05 : f32
    %2099 = tensor.splat %2098 {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} : tensor<256xf32>
    %2100 = tensor.empty() : tensor<256xf32>
    %2101 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%249, %2099 : tensor<256xf32>, tensor<256xf32>) outs(%2100 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} {
    ^bb240(%2102: f32, %2103: f32, %2104: f32):
      %2105 = arith.addf %2102, %2103 : f32
      linalg.yield %2105 : f32
    } -> tensor<256xf32>
    %2106 = tensor.empty() : tensor<256xf32>
    %2107 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2101 : tensor<256xf32>) outs(%2106 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} {
    ^bb241(%2108: f32, %2109: f32):
      %2110 = math.rsqrt %2108 : f32
      linalg.yield %2110 : f32
    } -> tensor<256xf32>
    %2111 = tensor.empty() : tensor<1x256x14x14xf32>
    %2112 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2093, %2107 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2111 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} {
    ^bb242(%2113: f32, %2114: f32, %2115: f32):
      %2116 = arith.mulf %2113, %2114 : f32
      linalg.yield %2116 : f32
    } -> tensor<1x256x14x14xf32>
    %2117 = tensor.empty() : tensor<1x256x14x14xf32>
    %2118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2112, %88 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2117 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} {
    ^bb243(%2119: f32, %2120: f32, %2121: f32):
      %2122 = arith.mulf %2119, %2120 : f32
      linalg.yield %2122 : f32
    } -> tensor<1x256x14x14xf32>
    %2123 = tensor.empty() : tensor<1x256x14x14xf32>
    %2124 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2118, %89 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2123 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_29", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn2"} {
    ^bb244(%2125: f32, %2126: f32, %2127: f32):
      %2128 = arith.addf %2125, %2126 : f32
      linalg.yield %2128 : f32
    } -> tensor<1x256x14x14xf32>
    %2129 = tensor.empty() : tensor<1x256x14x14xf32>
    %2130 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2124 : tensor<1x256x14x14xf32>) outs(%2129 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_26", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.relu"} {
    ^bb245(%2131: f32, %2132: f32):
      %2133 = arith.constant 0.000000e+00 : f32
      %2134 = arith.maximumf %2131, %2133 : f32
      linalg.yield %2134 : f32
    } -> tensor<1x256x14x14xf32>
    %2135 = tensor.empty() : tensor<256x1x1x1x14x14xf32>
    %2136 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2130 : tensor<1x256x14x14xf32>) outs(%2135 : tensor<256x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} {
    ^bb246(%2137: f32, %2138: f32):
      linalg.yield %2137 : f32
    } -> tensor<256x1x1x1x14x14xf32>
    %2139 = tensor.collapse_shape %2136 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<256x1x1x1x14x14xf32> into tensor<50176xf32>
    %2140 = tensor.expand_shape %2139 [[0 : i64, 1 : i64]] output_shape [256, 196] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<50176xf32> into tensor<256x196xf32>
    %2141 = tensor.collapse_shape %90 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<1024x256x1x1xf32> into tensor<262144xf32>
    %2142 = tensor.expand_shape %2141 [[0 : i64, 1 : i64]] output_shape [1024, 256] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<262144xf32> into tensor<1024x256xf32>
    %2143 = arith.constant {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} 0.000000e+00 : f32
    %2144 = tensor.splat %2143 {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<1024x196xf32>
    %2145 = linalg.matmul {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} ins(%2142, %2140 : tensor<1024x256xf32>, tensor<256x196xf32>) outs(%2144 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %2146 = tensor.collapse_shape %2145 [[0 : i64, 1 : i64]] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<1024x196xf32> into tensor<200704xf32>
    %2147 = tensor.expand_shape %2146 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %2148 = tensor.collapse_shape %2147 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %2149 = tensor.expand_shape %2148 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_30", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.conv3"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %2150 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2151 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2149, %251 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2150 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} {
    ^bb247(%2152: f32, %2153: f32, %2154: f32):
      %2155 = arith.subf %2152, %2153 : f32
      linalg.yield %2155 : f32
    } -> tensor<1x1024x14x14xf32>
    %2156 = arith.constant {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} 1.000000e-05 : f32
    %2157 = tensor.splat %2156 {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} : tensor<1024xf32>
    %2158 = tensor.empty() : tensor<1024xf32>
    %2159 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%252, %2157 : tensor<1024xf32>, tensor<1024xf32>) outs(%2158 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} {
    ^bb248(%2160: f32, %2161: f32, %2162: f32):
      %2163 = arith.addf %2160, %2161 : f32
      linalg.yield %2163 : f32
    } -> tensor<1024xf32>
    %2164 = tensor.empty() : tensor<1024xf32>
    %2165 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2159 : tensor<1024xf32>) outs(%2164 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} {
    ^bb249(%2166: f32, %2167: f32):
      %2168 = math.rsqrt %2166 : f32
      linalg.yield %2168 : f32
    } -> tensor<1024xf32>
    %2169 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2170 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2151, %2165 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2169 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} {
    ^bb250(%2171: f32, %2172: f32, %2173: f32):
      %2174 = arith.mulf %2171, %2172 : f32
      linalg.yield %2174 : f32
    } -> tensor<1x1024x14x14xf32>
    %2175 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2176 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2170, %91 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2175 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} {
    ^bb251(%2177: f32, %2178: f32, %2179: f32):
      %2180 = arith.mulf %2177, %2178 : f32
      linalg.yield %2180 : f32
    } -> tensor<1x1024x14x14xf32>
    %2181 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2176, %92 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2181 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_30", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.bn3"} {
    ^bb252(%2183: f32, %2184: f32, %2185: f32):
      %2186 = arith.addf %2183, %2184 : f32
      linalg.yield %2186 : f32
    } -> tensor<1x1024x14x14xf32>
    %2187 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2188 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2182, %2011 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%2187 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "add_8", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1"} {
    ^bb253(%2189: f32, %2190: f32, %2191: f32):
      %2192 = arith.addf %2189, %2190 : f32
      linalg.yield %2192 : f32
    } -> tensor<1x1024x14x14xf32>
    %2193 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2194 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2188 : tensor<1x1024x14x14xf32>) outs(%2193 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "minmax_27", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.1.relu"} {
    ^bb254(%2195: f32, %2196: f32):
      %2197 = arith.constant 0.000000e+00 : f32
      %2198 = arith.maximumf %2195, %2197 : f32
      linalg.yield %2198 : f32
    } -> tensor<1x1024x14x14xf32>
    %2199 = tensor.empty() : tensor<1024x1x1x1x14x14xf32>
    %2200 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2194 : tensor<1x1024x14x14xf32>) outs(%2199 : tensor<1024x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} {
    ^bb255(%2201: f32, %2202: f32):
      linalg.yield %2201 : f32
    } -> tensor<1024x1x1x1x14x14xf32>
    %2203 = tensor.collapse_shape %2200 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<1024x1x1x1x14x14xf32> into tensor<200704xf32>
    %2204 = tensor.expand_shape %2203 [[0 : i64, 1 : i64]] output_shape [1024, 196] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<200704xf32> into tensor<1024x196xf32>
    %2205 = tensor.collapse_shape %93 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<256x1024x1x1xf32> into tensor<262144xf32>
    %2206 = tensor.expand_shape %2205 [[0 : i64, 1 : i64]] output_shape [256, 1024] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<262144xf32> into tensor<256x1024xf32>
    %2207 = arith.constant {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} 0.000000e+00 : f32
    %2208 = tensor.splat %2207 {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<256x196xf32>
    %2209 = linalg.matmul {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} ins(%2206, %2204 : tensor<256x1024xf32>, tensor<1024x196xf32>) outs(%2208 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2210 = tensor.collapse_shape %2209 [[0 : i64, 1 : i64]] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<256x196xf32> into tensor<50176xf32>
    %2211 = tensor.expand_shape %2210 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2212 = tensor.collapse_shape %2211 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2213 = tensor.expand_shape %2212 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_31", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv1"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2214 = tensor.empty() : tensor<1x256x14x14xf32>
    %2215 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2213, %254 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2214 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} {
    ^bb256(%2216: f32, %2217: f32, %2218: f32):
      %2219 = arith.subf %2216, %2217 : f32
      linalg.yield %2219 : f32
    } -> tensor<1x256x14x14xf32>
    %2220 = arith.constant {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} 1.000000e-05 : f32
    %2221 = tensor.splat %2220 {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} : tensor<256xf32>
    %2222 = tensor.empty() : tensor<256xf32>
    %2223 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%255, %2221 : tensor<256xf32>, tensor<256xf32>) outs(%2222 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} {
    ^bb257(%2224: f32, %2225: f32, %2226: f32):
      %2227 = arith.addf %2224, %2225 : f32
      linalg.yield %2227 : f32
    } -> tensor<256xf32>
    %2228 = tensor.empty() : tensor<256xf32>
    %2229 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2223 : tensor<256xf32>) outs(%2228 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} {
    ^bb258(%2230: f32, %2231: f32):
      %2232 = math.rsqrt %2230 : f32
      linalg.yield %2232 : f32
    } -> tensor<256xf32>
    %2233 = tensor.empty() : tensor<1x256x14x14xf32>
    %2234 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2215, %2229 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2233 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} {
    ^bb259(%2235: f32, %2236: f32, %2237: f32):
      %2238 = arith.mulf %2235, %2236 : f32
      linalg.yield %2238 : f32
    } -> tensor<1x256x14x14xf32>
    %2239 = tensor.empty() : tensor<1x256x14x14xf32>
    %2240 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2234, %94 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2239 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} {
    ^bb260(%2241: f32, %2242: f32, %2243: f32):
      %2244 = arith.mulf %2241, %2242 : f32
      linalg.yield %2244 : f32
    } -> tensor<1x256x14x14xf32>
    %2245 = tensor.empty() : tensor<1x256x14x14xf32>
    %2246 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2240, %95 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2245 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_31", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn1"} {
    ^bb261(%2247: f32, %2248: f32, %2249: f32):
      %2250 = arith.addf %2247, %2248 : f32
      linalg.yield %2250 : f32
    } -> tensor<1x256x14x14xf32>
    %2251 = tensor.empty() : tensor<1x256x14x14xf32>
    %2252 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2246 : tensor<1x256x14x14xf32>) outs(%2251 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_28", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.relu"} {
    ^bb262(%2253: f32, %2254: f32):
      %2255 = arith.constant 0.000000e+00 : f32
      %2256 = arith.maximumf %2253, %2255 : f32
      linalg.yield %2256 : f32
    } -> tensor<1x256x14x14xf32>
    %2257 = arith.constant {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} 0.000000e+00 : f32
    %2258 = tensor.splat %2257 {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<1x256x16x16xf32>
    %2259 = "tensor.insert_slice"(%2252, %2258) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 14, 14>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : (tensor<1x256x14x14xf32>, tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %2260 = tensor.empty() : tensor<256x3x3x1x14x14xf32>
    %2261 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2259 : tensor<1x256x16x16xf32>) outs(%2260 : tensor<256x3x3x1x14x14xf32>) attrs =  {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} {
    ^bb263(%2262: f32, %2263: f32):
      linalg.yield %2262 : f32
    } -> tensor<256x3x3x1x14x14xf32>
    %2264 = tensor.collapse_shape %2261 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<256x3x3x1x14x14xf32> into tensor<451584xf32>
    %2265 = tensor.expand_shape %2264 [[0 : i64, 1 : i64]] output_shape [2304, 196] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<451584xf32> into tensor<2304x196xf32>
    %2266 = tensor.collapse_shape %96 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<256x256x3x3xf32> into tensor<589824xf32>
    %2267 = tensor.expand_shape %2266 [[0 : i64, 1 : i64]] output_shape [256, 2304] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<589824xf32> into tensor<256x2304xf32>
    %2268 = arith.constant {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} 0.000000e+00 : f32
    %2269 = tensor.splat %2268 {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<256x196xf32>
    %2270 = linalg.matmul {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} ins(%2267, %2265 : tensor<256x2304xf32>, tensor<2304x196xf32>) outs(%2269 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2271 = tensor.collapse_shape %2270 [[0 : i64, 1 : i64]] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<256x196xf32> into tensor<50176xf32>
    %2272 = tensor.expand_shape %2271 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2273 = tensor.collapse_shape %2272 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2274 = tensor.expand_shape %2273 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_32", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv2"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2275 = tensor.empty() : tensor<1x256x14x14xf32>
    %2276 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2274, %257 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2275 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} {
    ^bb264(%2277: f32, %2278: f32, %2279: f32):
      %2280 = arith.subf %2277, %2278 : f32
      linalg.yield %2280 : f32
    } -> tensor<1x256x14x14xf32>
    %2281 = arith.constant {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} 1.000000e-05 : f32
    %2282 = tensor.splat %2281 {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} : tensor<256xf32>
    %2283 = tensor.empty() : tensor<256xf32>
    %2284 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%258, %2282 : tensor<256xf32>, tensor<256xf32>) outs(%2283 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} {
    ^bb265(%2285: f32, %2286: f32, %2287: f32):
      %2288 = arith.addf %2285, %2286 : f32
      linalg.yield %2288 : f32
    } -> tensor<256xf32>
    %2289 = tensor.empty() : tensor<256xf32>
    %2290 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2284 : tensor<256xf32>) outs(%2289 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} {
    ^bb266(%2291: f32, %2292: f32):
      %2293 = math.rsqrt %2291 : f32
      linalg.yield %2293 : f32
    } -> tensor<256xf32>
    %2294 = tensor.empty() : tensor<1x256x14x14xf32>
    %2295 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2276, %2290 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2294 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} {
    ^bb267(%2296: f32, %2297: f32, %2298: f32):
      %2299 = arith.mulf %2296, %2297 : f32
      linalg.yield %2299 : f32
    } -> tensor<1x256x14x14xf32>
    %2300 = tensor.empty() : tensor<1x256x14x14xf32>
    %2301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2295, %97 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2300 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} {
    ^bb268(%2302: f32, %2303: f32, %2304: f32):
      %2305 = arith.mulf %2302, %2303 : f32
      linalg.yield %2305 : f32
    } -> tensor<1x256x14x14xf32>
    %2306 = tensor.empty() : tensor<1x256x14x14xf32>
    %2307 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2301, %98 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2306 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_32", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn2"} {
    ^bb269(%2308: f32, %2309: f32, %2310: f32):
      %2311 = arith.addf %2308, %2309 : f32
      linalg.yield %2311 : f32
    } -> tensor<1x256x14x14xf32>
    %2312 = tensor.empty() : tensor<1x256x14x14xf32>
    %2313 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2307 : tensor<1x256x14x14xf32>) outs(%2312 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_29", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.relu"} {
    ^bb270(%2314: f32, %2315: f32):
      %2316 = arith.constant 0.000000e+00 : f32
      %2317 = arith.maximumf %2314, %2316 : f32
      linalg.yield %2317 : f32
    } -> tensor<1x256x14x14xf32>
    %2318 = tensor.empty() : tensor<256x1x1x1x14x14xf32>
    %2319 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2313 : tensor<1x256x14x14xf32>) outs(%2318 : tensor<256x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} {
    ^bb271(%2320: f32, %2321: f32):
      linalg.yield %2320 : f32
    } -> tensor<256x1x1x1x14x14xf32>
    %2322 = tensor.collapse_shape %2319 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<256x1x1x1x14x14xf32> into tensor<50176xf32>
    %2323 = tensor.expand_shape %2322 [[0 : i64, 1 : i64]] output_shape [256, 196] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<50176xf32> into tensor<256x196xf32>
    %2324 = tensor.collapse_shape %99 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<1024x256x1x1xf32> into tensor<262144xf32>
    %2325 = tensor.expand_shape %2324 [[0 : i64, 1 : i64]] output_shape [1024, 256] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<262144xf32> into tensor<1024x256xf32>
    %2326 = arith.constant {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} 0.000000e+00 : f32
    %2327 = tensor.splat %2326 {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<1024x196xf32>
    %2328 = linalg.matmul {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} ins(%2325, %2323 : tensor<1024x256xf32>, tensor<256x196xf32>) outs(%2327 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %2329 = tensor.collapse_shape %2328 [[0 : i64, 1 : i64]] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<1024x196xf32> into tensor<200704xf32>
    %2330 = tensor.expand_shape %2329 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %2331 = tensor.collapse_shape %2330 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %2332 = tensor.expand_shape %2331 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_33", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.conv3"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %2333 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2334 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2332, %260 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2333 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} {
    ^bb272(%2335: f32, %2336: f32, %2337: f32):
      %2338 = arith.subf %2335, %2336 : f32
      linalg.yield %2338 : f32
    } -> tensor<1x1024x14x14xf32>
    %2339 = arith.constant {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} 1.000000e-05 : f32
    %2340 = tensor.splat %2339 {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} : tensor<1024xf32>
    %2341 = tensor.empty() : tensor<1024xf32>
    %2342 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%261, %2340 : tensor<1024xf32>, tensor<1024xf32>) outs(%2341 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} {
    ^bb273(%2343: f32, %2344: f32, %2345: f32):
      %2346 = arith.addf %2343, %2344 : f32
      linalg.yield %2346 : f32
    } -> tensor<1024xf32>
    %2347 = tensor.empty() : tensor<1024xf32>
    %2348 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2342 : tensor<1024xf32>) outs(%2347 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} {
    ^bb274(%2349: f32, %2350: f32):
      %2351 = math.rsqrt %2349 : f32
      linalg.yield %2351 : f32
    } -> tensor<1024xf32>
    %2352 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2353 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2334, %2348 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2352 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} {
    ^bb275(%2354: f32, %2355: f32, %2356: f32):
      %2357 = arith.mulf %2354, %2355 : f32
      linalg.yield %2357 : f32
    } -> tensor<1x1024x14x14xf32>
    %2358 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2359 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2353, %100 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2358 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} {
    ^bb276(%2360: f32, %2361: f32, %2362: f32):
      %2363 = arith.mulf %2360, %2361 : f32
      linalg.yield %2363 : f32
    } -> tensor<1x1024x14x14xf32>
    %2364 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2359, %101 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2364 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_33", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.bn3"} {
    ^bb277(%2366: f32, %2367: f32, %2368: f32):
      %2369 = arith.addf %2366, %2367 : f32
      linalg.yield %2369 : f32
    } -> tensor<1x1024x14x14xf32>
    %2370 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2371 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2365, %2194 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%2370 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "add_9", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2"} {
    ^bb278(%2372: f32, %2373: f32, %2374: f32):
      %2375 = arith.addf %2372, %2373 : f32
      linalg.yield %2375 : f32
    } -> tensor<1x1024x14x14xf32>
    %2376 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2377 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2371 : tensor<1x1024x14x14xf32>) outs(%2376 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "minmax_30", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.2.relu"} {
    ^bb279(%2378: f32, %2379: f32):
      %2380 = arith.constant 0.000000e+00 : f32
      %2381 = arith.maximumf %2378, %2380 : f32
      linalg.yield %2381 : f32
    } -> tensor<1x1024x14x14xf32>
    %2382 = tensor.empty() : tensor<1024x1x1x1x14x14xf32>
    %2383 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2377 : tensor<1x1024x14x14xf32>) outs(%2382 : tensor<1024x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} {
    ^bb280(%2384: f32, %2385: f32):
      linalg.yield %2384 : f32
    } -> tensor<1024x1x1x1x14x14xf32>
    %2386 = tensor.collapse_shape %2383 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<1024x1x1x1x14x14xf32> into tensor<200704xf32>
    %2387 = tensor.expand_shape %2386 [[0 : i64, 1 : i64]] output_shape [1024, 196] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<200704xf32> into tensor<1024x196xf32>
    %2388 = tensor.collapse_shape %102 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<256x1024x1x1xf32> into tensor<262144xf32>
    %2389 = tensor.expand_shape %2388 [[0 : i64, 1 : i64]] output_shape [256, 1024] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<262144xf32> into tensor<256x1024xf32>
    %2390 = arith.constant {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} 0.000000e+00 : f32
    %2391 = tensor.splat %2390 {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<256x196xf32>
    %2392 = linalg.matmul {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} ins(%2389, %2387 : tensor<256x1024xf32>, tensor<1024x196xf32>) outs(%2391 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2393 = tensor.collapse_shape %2392 [[0 : i64, 1 : i64]] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<256x196xf32> into tensor<50176xf32>
    %2394 = tensor.expand_shape %2393 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2395 = tensor.collapse_shape %2394 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2396 = tensor.expand_shape %2395 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_34", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv1"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2397 = tensor.empty() : tensor<1x256x14x14xf32>
    %2398 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2396, %263 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2397 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} {
    ^bb281(%2399: f32, %2400: f32, %2401: f32):
      %2402 = arith.subf %2399, %2400 : f32
      linalg.yield %2402 : f32
    } -> tensor<1x256x14x14xf32>
    %2403 = arith.constant {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} 1.000000e-05 : f32
    %2404 = tensor.splat %2403 {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} : tensor<256xf32>
    %2405 = tensor.empty() : tensor<256xf32>
    %2406 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%264, %2404 : tensor<256xf32>, tensor<256xf32>) outs(%2405 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} {
    ^bb282(%2407: f32, %2408: f32, %2409: f32):
      %2410 = arith.addf %2407, %2408 : f32
      linalg.yield %2410 : f32
    } -> tensor<256xf32>
    %2411 = tensor.empty() : tensor<256xf32>
    %2412 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2406 : tensor<256xf32>) outs(%2411 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} {
    ^bb283(%2413: f32, %2414: f32):
      %2415 = math.rsqrt %2413 : f32
      linalg.yield %2415 : f32
    } -> tensor<256xf32>
    %2416 = tensor.empty() : tensor<1x256x14x14xf32>
    %2417 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2398, %2412 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2416 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} {
    ^bb284(%2418: f32, %2419: f32, %2420: f32):
      %2421 = arith.mulf %2418, %2419 : f32
      linalg.yield %2421 : f32
    } -> tensor<1x256x14x14xf32>
    %2422 = tensor.empty() : tensor<1x256x14x14xf32>
    %2423 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2417, %103 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2422 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} {
    ^bb285(%2424: f32, %2425: f32, %2426: f32):
      %2427 = arith.mulf %2424, %2425 : f32
      linalg.yield %2427 : f32
    } -> tensor<1x256x14x14xf32>
    %2428 = tensor.empty() : tensor<1x256x14x14xf32>
    %2429 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2423, %104 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2428 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_34", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn1"} {
    ^bb286(%2430: f32, %2431: f32, %2432: f32):
      %2433 = arith.addf %2430, %2431 : f32
      linalg.yield %2433 : f32
    } -> tensor<1x256x14x14xf32>
    %2434 = tensor.empty() : tensor<1x256x14x14xf32>
    %2435 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2429 : tensor<1x256x14x14xf32>) outs(%2434 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_31", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.relu"} {
    ^bb287(%2436: f32, %2437: f32):
      %2438 = arith.constant 0.000000e+00 : f32
      %2439 = arith.maximumf %2436, %2438 : f32
      linalg.yield %2439 : f32
    } -> tensor<1x256x14x14xf32>
    %2440 = arith.constant {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} 0.000000e+00 : f32
    %2441 = tensor.splat %2440 {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<1x256x16x16xf32>
    %2442 = "tensor.insert_slice"(%2435, %2441) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 14, 14>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : (tensor<1x256x14x14xf32>, tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %2443 = tensor.empty() : tensor<256x3x3x1x14x14xf32>
    %2444 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2442 : tensor<1x256x16x16xf32>) outs(%2443 : tensor<256x3x3x1x14x14xf32>) attrs =  {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} {
    ^bb288(%2445: f32, %2446: f32):
      linalg.yield %2445 : f32
    } -> tensor<256x3x3x1x14x14xf32>
    %2447 = tensor.collapse_shape %2444 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<256x3x3x1x14x14xf32> into tensor<451584xf32>
    %2448 = tensor.expand_shape %2447 [[0 : i64, 1 : i64]] output_shape [2304, 196] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<451584xf32> into tensor<2304x196xf32>
    %2449 = tensor.collapse_shape %105 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<256x256x3x3xf32> into tensor<589824xf32>
    %2450 = tensor.expand_shape %2449 [[0 : i64, 1 : i64]] output_shape [256, 2304] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<589824xf32> into tensor<256x2304xf32>
    %2451 = arith.constant {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} 0.000000e+00 : f32
    %2452 = tensor.splat %2451 {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<256x196xf32>
    %2453 = linalg.matmul {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} ins(%2450, %2448 : tensor<256x2304xf32>, tensor<2304x196xf32>) outs(%2452 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2454 = tensor.collapse_shape %2453 [[0 : i64, 1 : i64]] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<256x196xf32> into tensor<50176xf32>
    %2455 = tensor.expand_shape %2454 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2456 = tensor.collapse_shape %2455 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2457 = tensor.expand_shape %2456 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_35", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv2"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2458 = tensor.empty() : tensor<1x256x14x14xf32>
    %2459 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2457, %266 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2458 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} {
    ^bb289(%2460: f32, %2461: f32, %2462: f32):
      %2463 = arith.subf %2460, %2461 : f32
      linalg.yield %2463 : f32
    } -> tensor<1x256x14x14xf32>
    %2464 = arith.constant {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} 1.000000e-05 : f32
    %2465 = tensor.splat %2464 {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} : tensor<256xf32>
    %2466 = tensor.empty() : tensor<256xf32>
    %2467 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%267, %2465 : tensor<256xf32>, tensor<256xf32>) outs(%2466 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} {
    ^bb290(%2468: f32, %2469: f32, %2470: f32):
      %2471 = arith.addf %2468, %2469 : f32
      linalg.yield %2471 : f32
    } -> tensor<256xf32>
    %2472 = tensor.empty() : tensor<256xf32>
    %2473 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2467 : tensor<256xf32>) outs(%2472 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} {
    ^bb291(%2474: f32, %2475: f32):
      %2476 = math.rsqrt %2474 : f32
      linalg.yield %2476 : f32
    } -> tensor<256xf32>
    %2477 = tensor.empty() : tensor<1x256x14x14xf32>
    %2478 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2459, %2473 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2477 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} {
    ^bb292(%2479: f32, %2480: f32, %2481: f32):
      %2482 = arith.mulf %2479, %2480 : f32
      linalg.yield %2482 : f32
    } -> tensor<1x256x14x14xf32>
    %2483 = tensor.empty() : tensor<1x256x14x14xf32>
    %2484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2478, %106 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2483 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} {
    ^bb293(%2485: f32, %2486: f32, %2487: f32):
      %2488 = arith.mulf %2485, %2486 : f32
      linalg.yield %2488 : f32
    } -> tensor<1x256x14x14xf32>
    %2489 = tensor.empty() : tensor<1x256x14x14xf32>
    %2490 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2484, %107 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2489 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_35", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn2"} {
    ^bb294(%2491: f32, %2492: f32, %2493: f32):
      %2494 = arith.addf %2491, %2492 : f32
      linalg.yield %2494 : f32
    } -> tensor<1x256x14x14xf32>
    %2495 = tensor.empty() : tensor<1x256x14x14xf32>
    %2496 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2490 : tensor<1x256x14x14xf32>) outs(%2495 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_32", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.relu"} {
    ^bb295(%2497: f32, %2498: f32):
      %2499 = arith.constant 0.000000e+00 : f32
      %2500 = arith.maximumf %2497, %2499 : f32
      linalg.yield %2500 : f32
    } -> tensor<1x256x14x14xf32>
    %2501 = tensor.empty() : tensor<256x1x1x1x14x14xf32>
    %2502 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2496 : tensor<1x256x14x14xf32>) outs(%2501 : tensor<256x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} {
    ^bb296(%2503: f32, %2504: f32):
      linalg.yield %2503 : f32
    } -> tensor<256x1x1x1x14x14xf32>
    %2505 = tensor.collapse_shape %2502 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<256x1x1x1x14x14xf32> into tensor<50176xf32>
    %2506 = tensor.expand_shape %2505 [[0 : i64, 1 : i64]] output_shape [256, 196] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<50176xf32> into tensor<256x196xf32>
    %2507 = tensor.collapse_shape %108 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<1024x256x1x1xf32> into tensor<262144xf32>
    %2508 = tensor.expand_shape %2507 [[0 : i64, 1 : i64]] output_shape [1024, 256] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<262144xf32> into tensor<1024x256xf32>
    %2509 = arith.constant {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} 0.000000e+00 : f32
    %2510 = tensor.splat %2509 {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<1024x196xf32>
    %2511 = linalg.matmul {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} ins(%2508, %2506 : tensor<1024x256xf32>, tensor<256x196xf32>) outs(%2510 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %2512 = tensor.collapse_shape %2511 [[0 : i64, 1 : i64]] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<1024x196xf32> into tensor<200704xf32>
    %2513 = tensor.expand_shape %2512 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %2514 = tensor.collapse_shape %2513 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %2515 = tensor.expand_shape %2514 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_36", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.conv3"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %2516 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2517 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2515, %269 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2516 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} {
    ^bb297(%2518: f32, %2519: f32, %2520: f32):
      %2521 = arith.subf %2518, %2519 : f32
      linalg.yield %2521 : f32
    } -> tensor<1x1024x14x14xf32>
    %2522 = arith.constant {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} 1.000000e-05 : f32
    %2523 = tensor.splat %2522 {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} : tensor<1024xf32>
    %2524 = tensor.empty() : tensor<1024xf32>
    %2525 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%270, %2523 : tensor<1024xf32>, tensor<1024xf32>) outs(%2524 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} {
    ^bb298(%2526: f32, %2527: f32, %2528: f32):
      %2529 = arith.addf %2526, %2527 : f32
      linalg.yield %2529 : f32
    } -> tensor<1024xf32>
    %2530 = tensor.empty() : tensor<1024xf32>
    %2531 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2525 : tensor<1024xf32>) outs(%2530 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} {
    ^bb299(%2532: f32, %2533: f32):
      %2534 = math.rsqrt %2532 : f32
      linalg.yield %2534 : f32
    } -> tensor<1024xf32>
    %2535 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2536 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2517, %2531 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2535 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} {
    ^bb300(%2537: f32, %2538: f32, %2539: f32):
      %2540 = arith.mulf %2537, %2538 : f32
      linalg.yield %2540 : f32
    } -> tensor<1x1024x14x14xf32>
    %2541 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2542 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2536, %109 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2541 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} {
    ^bb301(%2543: f32, %2544: f32, %2545: f32):
      %2546 = arith.mulf %2543, %2544 : f32
      linalg.yield %2546 : f32
    } -> tensor<1x1024x14x14xf32>
    %2547 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2548 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2542, %110 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2547 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_36", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.bn3"} {
    ^bb302(%2549: f32, %2550: f32, %2551: f32):
      %2552 = arith.addf %2549, %2550 : f32
      linalg.yield %2552 : f32
    } -> tensor<1x1024x14x14xf32>
    %2553 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2554 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2548, %2377 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%2553 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "add_10", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3"} {
    ^bb303(%2555: f32, %2556: f32, %2557: f32):
      %2558 = arith.addf %2555, %2556 : f32
      linalg.yield %2558 : f32
    } -> tensor<1x1024x14x14xf32>
    %2559 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2560 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2554 : tensor<1x1024x14x14xf32>) outs(%2559 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "minmax_33", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.3.relu"} {
    ^bb304(%2561: f32, %2562: f32):
      %2563 = arith.constant 0.000000e+00 : f32
      %2564 = arith.maximumf %2561, %2563 : f32
      linalg.yield %2564 : f32
    } -> tensor<1x1024x14x14xf32>
    %2565 = tensor.empty() : tensor<1024x1x1x1x14x14xf32>
    %2566 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2560 : tensor<1x1024x14x14xf32>) outs(%2565 : tensor<1024x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} {
    ^bb305(%2567: f32, %2568: f32):
      linalg.yield %2567 : f32
    } -> tensor<1024x1x1x1x14x14xf32>
    %2569 = tensor.collapse_shape %2566 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<1024x1x1x1x14x14xf32> into tensor<200704xf32>
    %2570 = tensor.expand_shape %2569 [[0 : i64, 1 : i64]] output_shape [1024, 196] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<200704xf32> into tensor<1024x196xf32>
    %2571 = tensor.collapse_shape %111 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<256x1024x1x1xf32> into tensor<262144xf32>
    %2572 = tensor.expand_shape %2571 [[0 : i64, 1 : i64]] output_shape [256, 1024] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<262144xf32> into tensor<256x1024xf32>
    %2573 = arith.constant {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} 0.000000e+00 : f32
    %2574 = tensor.splat %2573 {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<256x196xf32>
    %2575 = linalg.matmul {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} ins(%2572, %2570 : tensor<256x1024xf32>, tensor<1024x196xf32>) outs(%2574 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2576 = tensor.collapse_shape %2575 [[0 : i64, 1 : i64]] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<256x196xf32> into tensor<50176xf32>
    %2577 = tensor.expand_shape %2576 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2578 = tensor.collapse_shape %2577 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2579 = tensor.expand_shape %2578 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_37", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv1"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2580 = tensor.empty() : tensor<1x256x14x14xf32>
    %2581 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2579, %272 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2580 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} {
    ^bb306(%2582: f32, %2583: f32, %2584: f32):
      %2585 = arith.subf %2582, %2583 : f32
      linalg.yield %2585 : f32
    } -> tensor<1x256x14x14xf32>
    %2586 = arith.constant {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} 1.000000e-05 : f32
    %2587 = tensor.splat %2586 {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} : tensor<256xf32>
    %2588 = tensor.empty() : tensor<256xf32>
    %2589 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%273, %2587 : tensor<256xf32>, tensor<256xf32>) outs(%2588 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} {
    ^bb307(%2590: f32, %2591: f32, %2592: f32):
      %2593 = arith.addf %2590, %2591 : f32
      linalg.yield %2593 : f32
    } -> tensor<256xf32>
    %2594 = tensor.empty() : tensor<256xf32>
    %2595 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2589 : tensor<256xf32>) outs(%2594 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} {
    ^bb308(%2596: f32, %2597: f32):
      %2598 = math.rsqrt %2596 : f32
      linalg.yield %2598 : f32
    } -> tensor<256xf32>
    %2599 = tensor.empty() : tensor<1x256x14x14xf32>
    %2600 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2581, %2595 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2599 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} {
    ^bb309(%2601: f32, %2602: f32, %2603: f32):
      %2604 = arith.mulf %2601, %2602 : f32
      linalg.yield %2604 : f32
    } -> tensor<1x256x14x14xf32>
    %2605 = tensor.empty() : tensor<1x256x14x14xf32>
    %2606 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2600, %112 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2605 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} {
    ^bb310(%2607: f32, %2608: f32, %2609: f32):
      %2610 = arith.mulf %2607, %2608 : f32
      linalg.yield %2610 : f32
    } -> tensor<1x256x14x14xf32>
    %2611 = tensor.empty() : tensor<1x256x14x14xf32>
    %2612 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2606, %113 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2611 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_37", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn1"} {
    ^bb311(%2613: f32, %2614: f32, %2615: f32):
      %2616 = arith.addf %2613, %2614 : f32
      linalg.yield %2616 : f32
    } -> tensor<1x256x14x14xf32>
    %2617 = tensor.empty() : tensor<1x256x14x14xf32>
    %2618 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2612 : tensor<1x256x14x14xf32>) outs(%2617 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_34", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.relu"} {
    ^bb312(%2619: f32, %2620: f32):
      %2621 = arith.constant 0.000000e+00 : f32
      %2622 = arith.maximumf %2619, %2621 : f32
      linalg.yield %2622 : f32
    } -> tensor<1x256x14x14xf32>
    %2623 = arith.constant {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} 0.000000e+00 : f32
    %2624 = tensor.splat %2623 {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<1x256x16x16xf32>
    %2625 = "tensor.insert_slice"(%2618, %2624) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 14, 14>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : (tensor<1x256x14x14xf32>, tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %2626 = tensor.empty() : tensor<256x3x3x1x14x14xf32>
    %2627 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2625 : tensor<1x256x16x16xf32>) outs(%2626 : tensor<256x3x3x1x14x14xf32>) attrs =  {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} {
    ^bb313(%2628: f32, %2629: f32):
      linalg.yield %2628 : f32
    } -> tensor<256x3x3x1x14x14xf32>
    %2630 = tensor.collapse_shape %2627 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<256x3x3x1x14x14xf32> into tensor<451584xf32>
    %2631 = tensor.expand_shape %2630 [[0 : i64, 1 : i64]] output_shape [2304, 196] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<451584xf32> into tensor<2304x196xf32>
    %2632 = tensor.collapse_shape %114 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<256x256x3x3xf32> into tensor<589824xf32>
    %2633 = tensor.expand_shape %2632 [[0 : i64, 1 : i64]] output_shape [256, 2304] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<589824xf32> into tensor<256x2304xf32>
    %2634 = arith.constant {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} 0.000000e+00 : f32
    %2635 = tensor.splat %2634 {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<256x196xf32>
    %2636 = linalg.matmul {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} ins(%2633, %2631 : tensor<256x2304xf32>, tensor<2304x196xf32>) outs(%2635 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2637 = tensor.collapse_shape %2636 [[0 : i64, 1 : i64]] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<256x196xf32> into tensor<50176xf32>
    %2638 = tensor.expand_shape %2637 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2639 = tensor.collapse_shape %2638 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2640 = tensor.expand_shape %2639 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_38", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv2"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2641 = tensor.empty() : tensor<1x256x14x14xf32>
    %2642 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2640, %275 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2641 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} {
    ^bb314(%2643: f32, %2644: f32, %2645: f32):
      %2646 = arith.subf %2643, %2644 : f32
      linalg.yield %2646 : f32
    } -> tensor<1x256x14x14xf32>
    %2647 = arith.constant {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} 1.000000e-05 : f32
    %2648 = tensor.splat %2647 {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} : tensor<256xf32>
    %2649 = tensor.empty() : tensor<256xf32>
    %2650 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%276, %2648 : tensor<256xf32>, tensor<256xf32>) outs(%2649 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} {
    ^bb315(%2651: f32, %2652: f32, %2653: f32):
      %2654 = arith.addf %2651, %2652 : f32
      linalg.yield %2654 : f32
    } -> tensor<256xf32>
    %2655 = tensor.empty() : tensor<256xf32>
    %2656 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2650 : tensor<256xf32>) outs(%2655 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} {
    ^bb316(%2657: f32, %2658: f32):
      %2659 = math.rsqrt %2657 : f32
      linalg.yield %2659 : f32
    } -> tensor<256xf32>
    %2660 = tensor.empty() : tensor<1x256x14x14xf32>
    %2661 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2642, %2656 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2660 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} {
    ^bb317(%2662: f32, %2663: f32, %2664: f32):
      %2665 = arith.mulf %2662, %2663 : f32
      linalg.yield %2665 : f32
    } -> tensor<1x256x14x14xf32>
    %2666 = tensor.empty() : tensor<1x256x14x14xf32>
    %2667 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2661, %115 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2666 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} {
    ^bb318(%2668: f32, %2669: f32, %2670: f32):
      %2671 = arith.mulf %2668, %2669 : f32
      linalg.yield %2671 : f32
    } -> tensor<1x256x14x14xf32>
    %2672 = tensor.empty() : tensor<1x256x14x14xf32>
    %2673 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2667, %116 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2672 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_38", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn2"} {
    ^bb319(%2674: f32, %2675: f32, %2676: f32):
      %2677 = arith.addf %2674, %2675 : f32
      linalg.yield %2677 : f32
    } -> tensor<1x256x14x14xf32>
    %2678 = tensor.empty() : tensor<1x256x14x14xf32>
    %2679 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2673 : tensor<1x256x14x14xf32>) outs(%2678 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_35", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.relu"} {
    ^bb320(%2680: f32, %2681: f32):
      %2682 = arith.constant 0.000000e+00 : f32
      %2683 = arith.maximumf %2680, %2682 : f32
      linalg.yield %2683 : f32
    } -> tensor<1x256x14x14xf32>
    %2684 = tensor.empty() : tensor<256x1x1x1x14x14xf32>
    %2685 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2679 : tensor<1x256x14x14xf32>) outs(%2684 : tensor<256x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} {
    ^bb321(%2686: f32, %2687: f32):
      linalg.yield %2686 : f32
    } -> tensor<256x1x1x1x14x14xf32>
    %2688 = tensor.collapse_shape %2685 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<256x1x1x1x14x14xf32> into tensor<50176xf32>
    %2689 = tensor.expand_shape %2688 [[0 : i64, 1 : i64]] output_shape [256, 196] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<50176xf32> into tensor<256x196xf32>
    %2690 = tensor.collapse_shape %117 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<1024x256x1x1xf32> into tensor<262144xf32>
    %2691 = tensor.expand_shape %2690 [[0 : i64, 1 : i64]] output_shape [1024, 256] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<262144xf32> into tensor<1024x256xf32>
    %2692 = arith.constant {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} 0.000000e+00 : f32
    %2693 = tensor.splat %2692 {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<1024x196xf32>
    %2694 = linalg.matmul {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} ins(%2691, %2689 : tensor<1024x256xf32>, tensor<256x196xf32>) outs(%2693 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %2695 = tensor.collapse_shape %2694 [[0 : i64, 1 : i64]] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<1024x196xf32> into tensor<200704xf32>
    %2696 = tensor.expand_shape %2695 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %2697 = tensor.collapse_shape %2696 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %2698 = tensor.expand_shape %2697 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_39", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.conv3"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %2699 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2700 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2698, %278 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2699 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} {
    ^bb322(%2701: f32, %2702: f32, %2703: f32):
      %2704 = arith.subf %2701, %2702 : f32
      linalg.yield %2704 : f32
    } -> tensor<1x1024x14x14xf32>
    %2705 = arith.constant {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} 1.000000e-05 : f32
    %2706 = tensor.splat %2705 {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} : tensor<1024xf32>
    %2707 = tensor.empty() : tensor<1024xf32>
    %2708 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%279, %2706 : tensor<1024xf32>, tensor<1024xf32>) outs(%2707 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} {
    ^bb323(%2709: f32, %2710: f32, %2711: f32):
      %2712 = arith.addf %2709, %2710 : f32
      linalg.yield %2712 : f32
    } -> tensor<1024xf32>
    %2713 = tensor.empty() : tensor<1024xf32>
    %2714 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2708 : tensor<1024xf32>) outs(%2713 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} {
    ^bb324(%2715: f32, %2716: f32):
      %2717 = math.rsqrt %2715 : f32
      linalg.yield %2717 : f32
    } -> tensor<1024xf32>
    %2718 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2719 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2700, %2714 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2718 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} {
    ^bb325(%2720: f32, %2721: f32, %2722: f32):
      %2723 = arith.mulf %2720, %2721 : f32
      linalg.yield %2723 : f32
    } -> tensor<1x1024x14x14xf32>
    %2724 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2725 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2719, %118 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2724 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} {
    ^bb326(%2726: f32, %2727: f32, %2728: f32):
      %2729 = arith.mulf %2726, %2727 : f32
      linalg.yield %2729 : f32
    } -> tensor<1x1024x14x14xf32>
    %2730 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2731 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2725, %119 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2730 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_39", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.bn3"} {
    ^bb327(%2732: f32, %2733: f32, %2734: f32):
      %2735 = arith.addf %2732, %2733 : f32
      linalg.yield %2735 : f32
    } -> tensor<1x1024x14x14xf32>
    %2736 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2737 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2731, %2560 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%2736 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "add_11", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4"} {
    ^bb328(%2738: f32, %2739: f32, %2740: f32):
      %2741 = arith.addf %2738, %2739 : f32
      linalg.yield %2741 : f32
    } -> tensor<1x1024x14x14xf32>
    %2742 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2743 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2737 : tensor<1x1024x14x14xf32>) outs(%2742 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "minmax_36", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.4.relu"} {
    ^bb329(%2744: f32, %2745: f32):
      %2746 = arith.constant 0.000000e+00 : f32
      %2747 = arith.maximumf %2744, %2746 : f32
      linalg.yield %2747 : f32
    } -> tensor<1x1024x14x14xf32>
    %2748 = tensor.empty() : tensor<1024x1x1x1x14x14xf32>
    %2749 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2743 : tensor<1x1024x14x14xf32>) outs(%2748 : tensor<1024x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} {
    ^bb330(%2750: f32, %2751: f32):
      linalg.yield %2750 : f32
    } -> tensor<1024x1x1x1x14x14xf32>
    %2752 = tensor.collapse_shape %2749 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<1024x1x1x1x14x14xf32> into tensor<200704xf32>
    %2753 = tensor.expand_shape %2752 [[0 : i64, 1 : i64]] output_shape [1024, 196] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<200704xf32> into tensor<1024x196xf32>
    %2754 = tensor.collapse_shape %120 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<256x1024x1x1xf32> into tensor<262144xf32>
    %2755 = tensor.expand_shape %2754 [[0 : i64, 1 : i64]] output_shape [256, 1024] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<262144xf32> into tensor<256x1024xf32>
    %2756 = arith.constant {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} 0.000000e+00 : f32
    %2757 = tensor.splat %2756 {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<256x196xf32>
    %2758 = linalg.matmul {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} ins(%2755, %2753 : tensor<256x1024xf32>, tensor<1024x196xf32>) outs(%2757 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2759 = tensor.collapse_shape %2758 [[0 : i64, 1 : i64]] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<256x196xf32> into tensor<50176xf32>
    %2760 = tensor.expand_shape %2759 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2761 = tensor.collapse_shape %2760 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2762 = tensor.expand_shape %2761 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_40", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv1"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2763 = tensor.empty() : tensor<1x256x14x14xf32>
    %2764 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2762, %281 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2763 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} {
    ^bb331(%2765: f32, %2766: f32, %2767: f32):
      %2768 = arith.subf %2765, %2766 : f32
      linalg.yield %2768 : f32
    } -> tensor<1x256x14x14xf32>
    %2769 = arith.constant {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} 1.000000e-05 : f32
    %2770 = tensor.splat %2769 {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} : tensor<256xf32>
    %2771 = tensor.empty() : tensor<256xf32>
    %2772 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%282, %2770 : tensor<256xf32>, tensor<256xf32>) outs(%2771 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} {
    ^bb332(%2773: f32, %2774: f32, %2775: f32):
      %2776 = arith.addf %2773, %2774 : f32
      linalg.yield %2776 : f32
    } -> tensor<256xf32>
    %2777 = tensor.empty() : tensor<256xf32>
    %2778 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2772 : tensor<256xf32>) outs(%2777 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} {
    ^bb333(%2779: f32, %2780: f32):
      %2781 = math.rsqrt %2779 : f32
      linalg.yield %2781 : f32
    } -> tensor<256xf32>
    %2782 = tensor.empty() : tensor<1x256x14x14xf32>
    %2783 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2764, %2778 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2782 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} {
    ^bb334(%2784: f32, %2785: f32, %2786: f32):
      %2787 = arith.mulf %2784, %2785 : f32
      linalg.yield %2787 : f32
    } -> tensor<1x256x14x14xf32>
    %2788 = tensor.empty() : tensor<1x256x14x14xf32>
    %2789 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2783, %121 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2788 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} {
    ^bb335(%2790: f32, %2791: f32, %2792: f32):
      %2793 = arith.mulf %2790, %2791 : f32
      linalg.yield %2793 : f32
    } -> tensor<1x256x14x14xf32>
    %2794 = tensor.empty() : tensor<1x256x14x14xf32>
    %2795 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2789, %122 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2794 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_40", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn1"} {
    ^bb336(%2796: f32, %2797: f32, %2798: f32):
      %2799 = arith.addf %2796, %2797 : f32
      linalg.yield %2799 : f32
    } -> tensor<1x256x14x14xf32>
    %2800 = tensor.empty() : tensor<1x256x14x14xf32>
    %2801 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2795 : tensor<1x256x14x14xf32>) outs(%2800 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_37", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.relu"} {
    ^bb337(%2802: f32, %2803: f32):
      %2804 = arith.constant 0.000000e+00 : f32
      %2805 = arith.maximumf %2802, %2804 : f32
      linalg.yield %2805 : f32
    } -> tensor<1x256x14x14xf32>
    %2806 = arith.constant {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} 0.000000e+00 : f32
    %2807 = tensor.splat %2806 {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<1x256x16x16xf32>
    %2808 = "tensor.insert_slice"(%2801, %2807) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 256, 14, 14>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : (tensor<1x256x14x14xf32>, tensor<1x256x16x16xf32>) -> tensor<1x256x16x16xf32>
    %2809 = tensor.empty() : tensor<256x3x3x1x14x14xf32>
    %2810 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2808 : tensor<1x256x16x16xf32>) outs(%2809 : tensor<256x3x3x1x14x14xf32>) attrs =  {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} {
    ^bb338(%2811: f32, %2812: f32):
      linalg.yield %2811 : f32
    } -> tensor<256x3x3x1x14x14xf32>
    %2813 = tensor.collapse_shape %2810 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<256x3x3x1x14x14xf32> into tensor<451584xf32>
    %2814 = tensor.expand_shape %2813 [[0 : i64, 1 : i64]] output_shape [2304, 196] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<451584xf32> into tensor<2304x196xf32>
    %2815 = tensor.collapse_shape %123 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<256x256x3x3xf32> into tensor<589824xf32>
    %2816 = tensor.expand_shape %2815 [[0 : i64, 1 : i64]] output_shape [256, 2304] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<589824xf32> into tensor<256x2304xf32>
    %2817 = arith.constant {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} 0.000000e+00 : f32
    %2818 = tensor.splat %2817 {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<256x196xf32>
    %2819 = linalg.matmul {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} ins(%2816, %2814 : tensor<256x2304xf32>, tensor<2304x196xf32>) outs(%2818 : tensor<256x196xf32>) -> tensor<256x196xf32>
    %2820 = tensor.collapse_shape %2819 [[0 : i64, 1 : i64]] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<256x196xf32> into tensor<50176xf32>
    %2821 = tensor.expand_shape %2820 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [256, 1, 14, 14] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<50176xf32> into tensor<256x1x14x14xf32>
    %2822 = tensor.collapse_shape %2821 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<256x1x14x14xf32> into tensor<50176xf32>
    %2823 = tensor.expand_shape %2822 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 256, 14, 14] {prov.region_id = "conv_41", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv2"} : tensor<50176xf32> into tensor<1x256x14x14xf32>
    %2824 = tensor.empty() : tensor<1x256x14x14xf32>
    %2825 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2823, %284 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2824 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} {
    ^bb339(%2826: f32, %2827: f32, %2828: f32):
      %2829 = arith.subf %2826, %2827 : f32
      linalg.yield %2829 : f32
    } -> tensor<1x256x14x14xf32>
    %2830 = arith.constant {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} 1.000000e-05 : f32
    %2831 = tensor.splat %2830 {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} : tensor<256xf32>
    %2832 = tensor.empty() : tensor<256xf32>
    %2833 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%285, %2831 : tensor<256xf32>, tensor<256xf32>) outs(%2832 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} {
    ^bb340(%2834: f32, %2835: f32, %2836: f32):
      %2837 = arith.addf %2834, %2835 : f32
      linalg.yield %2837 : f32
    } -> tensor<256xf32>
    %2838 = tensor.empty() : tensor<256xf32>
    %2839 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2833 : tensor<256xf32>) outs(%2838 : tensor<256xf32>) attrs =  {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} {
    ^bb341(%2840: f32, %2841: f32):
      %2842 = math.rsqrt %2840 : f32
      linalg.yield %2842 : f32
    } -> tensor<256xf32>
    %2843 = tensor.empty() : tensor<1x256x14x14xf32>
    %2844 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2825, %2839 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2843 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} {
    ^bb342(%2845: f32, %2846: f32, %2847: f32):
      %2848 = arith.mulf %2845, %2846 : f32
      linalg.yield %2848 : f32
    } -> tensor<1x256x14x14xf32>
    %2849 = tensor.empty() : tensor<1x256x14x14xf32>
    %2850 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2844, %124 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2849 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} {
    ^bb343(%2851: f32, %2852: f32, %2853: f32):
      %2854 = arith.mulf %2851, %2852 : f32
      linalg.yield %2854 : f32
    } -> tensor<1x256x14x14xf32>
    %2855 = tensor.empty() : tensor<1x256x14x14xf32>
    %2856 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2850, %125 : tensor<1x256x14x14xf32>, tensor<256xf32>) outs(%2855 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "batch_norm_41", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn2"} {
    ^bb344(%2857: f32, %2858: f32, %2859: f32):
      %2860 = arith.addf %2857, %2858 : f32
      linalg.yield %2860 : f32
    } -> tensor<1x256x14x14xf32>
    %2861 = tensor.empty() : tensor<1x256x14x14xf32>
    %2862 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2856 : tensor<1x256x14x14xf32>) outs(%2861 : tensor<1x256x14x14xf32>) attrs =  {prov.region_id = "minmax_38", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.relu"} {
    ^bb345(%2863: f32, %2864: f32):
      %2865 = arith.constant 0.000000e+00 : f32
      %2866 = arith.maximumf %2863, %2865 : f32
      linalg.yield %2866 : f32
    } -> tensor<1x256x14x14xf32>
    %2867 = tensor.empty() : tensor<256x1x1x1x14x14xf32>
    %2868 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2862 : tensor<1x256x14x14xf32>) outs(%2867 : tensor<256x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} {
    ^bb346(%2869: f32, %2870: f32):
      linalg.yield %2869 : f32
    } -> tensor<256x1x1x1x14x14xf32>
    %2871 = tensor.collapse_shape %2868 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<256x1x1x1x14x14xf32> into tensor<50176xf32>
    %2872 = tensor.expand_shape %2871 [[0 : i64, 1 : i64]] output_shape [256, 196] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<50176xf32> into tensor<256x196xf32>
    %2873 = tensor.collapse_shape %126 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<1024x256x1x1xf32> into tensor<262144xf32>
    %2874 = tensor.expand_shape %2873 [[0 : i64, 1 : i64]] output_shape [1024, 256] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<262144xf32> into tensor<1024x256xf32>
    %2875 = arith.constant {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} 0.000000e+00 : f32
    %2876 = tensor.splat %2875 {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<1024x196xf32>
    %2877 = linalg.matmul {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} ins(%2874, %2872 : tensor<1024x256xf32>, tensor<256x196xf32>) outs(%2876 : tensor<1024x196xf32>) -> tensor<1024x196xf32>
    %2878 = tensor.collapse_shape %2877 [[0 : i64, 1 : i64]] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<1024x196xf32> into tensor<200704xf32>
    %2879 = tensor.expand_shape %2878 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1024, 1, 14, 14] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<200704xf32> into tensor<1024x1x14x14xf32>
    %2880 = tensor.collapse_shape %2879 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<1024x1x14x14xf32> into tensor<200704xf32>
    %2881 = tensor.expand_shape %2880 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 1024, 14, 14] {prov.region_id = "conv_42", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.conv3"} : tensor<200704xf32> into tensor<1x1024x14x14xf32>
    %2882 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2883 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2881, %287 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2882 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} {
    ^bb347(%2884: f32, %2885: f32, %2886: f32):
      %2887 = arith.subf %2884, %2885 : f32
      linalg.yield %2887 : f32
    } -> tensor<1x1024x14x14xf32>
    %2888 = arith.constant {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} 1.000000e-05 : f32
    %2889 = tensor.splat %2888 {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} : tensor<1024xf32>
    %2890 = tensor.empty() : tensor<1024xf32>
    %2891 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%288, %2889 : tensor<1024xf32>, tensor<1024xf32>) outs(%2890 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} {
    ^bb348(%2892: f32, %2893: f32, %2894: f32):
      %2895 = arith.addf %2892, %2893 : f32
      linalg.yield %2895 : f32
    } -> tensor<1024xf32>
    %2896 = tensor.empty() : tensor<1024xf32>
    %2897 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2891 : tensor<1024xf32>) outs(%2896 : tensor<1024xf32>) attrs =  {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} {
    ^bb349(%2898: f32, %2899: f32):
      %2900 = math.rsqrt %2898 : f32
      linalg.yield %2900 : f32
    } -> tensor<1024xf32>
    %2901 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2902 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2883, %2897 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2901 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} {
    ^bb350(%2903: f32, %2904: f32, %2905: f32):
      %2906 = arith.mulf %2903, %2904 : f32
      linalg.yield %2906 : f32
    } -> tensor<1x1024x14x14xf32>
    %2907 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2908 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2902, %127 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2907 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} {
    ^bb351(%2909: f32, %2910: f32, %2911: f32):
      %2912 = arith.mulf %2909, %2910 : f32
      linalg.yield %2912 : f32
    } -> tensor<1x1024x14x14xf32>
    %2913 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2914 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2908, %128 : tensor<1x1024x14x14xf32>, tensor<1024xf32>) outs(%2913 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "batch_norm_42", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.bn3"} {
    ^bb352(%2915: f32, %2916: f32, %2917: f32):
      %2918 = arith.addf %2915, %2916 : f32
      linalg.yield %2918 : f32
    } -> tensor<1x1024x14x14xf32>
    %2919 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2920 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2914, %2743 : tensor<1x1024x14x14xf32>, tensor<1x1024x14x14xf32>) outs(%2919 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "add_12", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5"} {
    ^bb353(%2921: f32, %2922: f32, %2923: f32):
      %2924 = arith.addf %2921, %2922 : f32
      linalg.yield %2924 : f32
    } -> tensor<1x1024x14x14xf32>
    %2925 = tensor.empty() : tensor<1x1024x14x14xf32>
    %2926 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2920 : tensor<1x1024x14x14xf32>) outs(%2925 : tensor<1x1024x14x14xf32>) attrs =  {prov.region_id = "minmax_39", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer3.5.relu"} {
    ^bb354(%2927: f32, %2928: f32):
      %2929 = arith.constant 0.000000e+00 : f32
      %2930 = arith.maximumf %2927, %2929 : f32
      linalg.yield %2930 : f32
    } -> tensor<1x1024x14x14xf32>
    %2931 = tensor.empty() : tensor<1024x1x1x1x14x14xf32>
    %2932 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2926 : tensor<1x1024x14x14xf32>) outs(%2931 : tensor<1024x1x1x1x14x14xf32>) attrs =  {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} {
    ^bb355(%2933: f32, %2934: f32):
      linalg.yield %2933 : f32
    } -> tensor<1024x1x1x1x14x14xf32>
    %2935 = tensor.collapse_shape %2932 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<1024x1x1x1x14x14xf32> into tensor<200704xf32>
    %2936 = tensor.expand_shape %2935 [[0 : i64, 1 : i64]] output_shape [1024, 196] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<200704xf32> into tensor<1024x196xf32>
    %2937 = tensor.collapse_shape %129 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<512x1024x1x1xf32> into tensor<524288xf32>
    %2938 = tensor.expand_shape %2937 [[0 : i64, 1 : i64]] output_shape [512, 1024] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<524288xf32> into tensor<512x1024xf32>
    %2939 = arith.constant {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} 0.000000e+00 : f32
    %2940 = tensor.splat %2939 {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<512x196xf32>
    %2941 = linalg.matmul {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} ins(%2938, %2936 : tensor<512x1024xf32>, tensor<1024x196xf32>) outs(%2940 : tensor<512x196xf32>) -> tensor<512x196xf32>
    %2942 = tensor.collapse_shape %2941 [[0 : i64, 1 : i64]] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<512x196xf32> into tensor<100352xf32>
    %2943 = tensor.expand_shape %2942 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 14, 14] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<100352xf32> into tensor<512x1x14x14xf32>
    %2944 = tensor.collapse_shape %2943 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<512x1x14x14xf32> into tensor<100352xf32>
    %2945 = tensor.expand_shape %2944 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 14, 14] {prov.region_id = "conv_43", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv1"} : tensor<100352xf32> into tensor<1x512x14x14xf32>
    %2946 = tensor.empty() : tensor<1x512x14x14xf32>
    %2947 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2945, %290 : tensor<1x512x14x14xf32>, tensor<512xf32>) outs(%2946 : tensor<1x512x14x14xf32>) attrs =  {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} {
    ^bb356(%2948: f32, %2949: f32, %2950: f32):
      %2951 = arith.subf %2948, %2949 : f32
      linalg.yield %2951 : f32
    } -> tensor<1x512x14x14xf32>
    %2952 = arith.constant {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} 1.000000e-05 : f32
    %2953 = tensor.splat %2952 {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} : tensor<512xf32>
    %2954 = tensor.empty() : tensor<512xf32>
    %2955 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%291, %2953 : tensor<512xf32>, tensor<512xf32>) outs(%2954 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} {
    ^bb357(%2956: f32, %2957: f32, %2958: f32):
      %2959 = arith.addf %2956, %2957 : f32
      linalg.yield %2959 : f32
    } -> tensor<512xf32>
    %2960 = tensor.empty() : tensor<512xf32>
    %2961 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%2955 : tensor<512xf32>) outs(%2960 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} {
    ^bb358(%2962: f32, %2963: f32):
      %2964 = math.rsqrt %2962 : f32
      linalg.yield %2964 : f32
    } -> tensor<512xf32>
    %2965 = tensor.empty() : tensor<1x512x14x14xf32>
    %2966 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2947, %2961 : tensor<1x512x14x14xf32>, tensor<512xf32>) outs(%2965 : tensor<1x512x14x14xf32>) attrs =  {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} {
    ^bb359(%2967: f32, %2968: f32, %2969: f32):
      %2970 = arith.mulf %2967, %2968 : f32
      linalg.yield %2970 : f32
    } -> tensor<1x512x14x14xf32>
    %2971 = tensor.empty() : tensor<1x512x14x14xf32>
    %2972 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2966, %130 : tensor<1x512x14x14xf32>, tensor<512xf32>) outs(%2971 : tensor<1x512x14x14xf32>) attrs =  {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} {
    ^bb360(%2973: f32, %2974: f32, %2975: f32):
      %2976 = arith.mulf %2973, %2974 : f32
      linalg.yield %2976 : f32
    } -> tensor<1x512x14x14xf32>
    %2977 = tensor.empty() : tensor<1x512x14x14xf32>
    %2978 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2972, %131 : tensor<1x512x14x14xf32>, tensor<512xf32>) outs(%2977 : tensor<1x512x14x14xf32>) attrs =  {prov.region_id = "batch_norm_43", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn1"} {
    ^bb361(%2979: f32, %2980: f32, %2981: f32):
      %2982 = arith.addf %2979, %2980 : f32
      linalg.yield %2982 : f32
    } -> tensor<1x512x14x14xf32>
    %2983 = tensor.empty() : tensor<1x512x14x14xf32>
    %2984 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%2978 : tensor<1x512x14x14xf32>) outs(%2983 : tensor<1x512x14x14xf32>) attrs =  {prov.region_id = "minmax_40", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.relu"} {
    ^bb362(%2985: f32, %2986: f32):
      %2987 = arith.constant 0.000000e+00 : f32
      %2988 = arith.maximumf %2985, %2987 : f32
      linalg.yield %2988 : f32
    } -> tensor<1x512x14x14xf32>
    %2989 = arith.constant {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} 0.000000e+00 : f32
    %2990 = tensor.splat %2989 {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<1x512x16x16xf32>
    %2991 = "tensor.insert_slice"(%2984, %2990) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 14, 14>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : (tensor<1x512x14x14xf32>, tensor<1x512x16x16xf32>) -> tensor<1x512x16x16xf32>
    %2992 = tensor.empty() : tensor<512x3x3x1x7x7xf32>
    %2993 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2991 : tensor<1x512x16x16xf32>) outs(%2992 : tensor<512x3x3x1x7x7xf32>) attrs =  {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} {
    ^bb363(%2994: f32, %2995: f32):
      linalg.yield %2994 : f32
    } -> tensor<512x3x3x1x7x7xf32>
    %2996 = tensor.collapse_shape %2993 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<512x3x3x1x7x7xf32> into tensor<225792xf32>
    %2997 = tensor.expand_shape %2996 [[0 : i64, 1 : i64]] output_shape [4608, 49] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<225792xf32> into tensor<4608x49xf32>
    %2998 = tensor.collapse_shape %132 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<512x512x3x3xf32> into tensor<2359296xf32>
    %2999 = tensor.expand_shape %2998 [[0 : i64, 1 : i64]] output_shape [512, 4608] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<2359296xf32> into tensor<512x4608xf32>
    %3000 = arith.constant {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} 0.000000e+00 : f32
    %3001 = tensor.splat %3000 {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<512x49xf32>
    %3002 = linalg.matmul {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} ins(%2999, %2997 : tensor<512x4608xf32>, tensor<4608x49xf32>) outs(%3001 : tensor<512x49xf32>) -> tensor<512x49xf32>
    %3003 = tensor.collapse_shape %3002 [[0 : i64, 1 : i64]] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<512x49xf32> into tensor<25088xf32>
    %3004 = tensor.expand_shape %3003 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 7, 7] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<25088xf32> into tensor<512x1x7x7xf32>
    %3005 = tensor.collapse_shape %3004 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<512x1x7x7xf32> into tensor<25088xf32>
    %3006 = tensor.expand_shape %3005 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 7, 7] {prov.region_id = "conv_44", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv2"} : tensor<25088xf32> into tensor<1x512x7x7xf32>
    %3007 = tensor.empty() : tensor<1x512x7x7xf32>
    %3008 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3006, %293 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3007 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} {
    ^bb364(%3009: f32, %3010: f32, %3011: f32):
      %3012 = arith.subf %3009, %3010 : f32
      linalg.yield %3012 : f32
    } -> tensor<1x512x7x7xf32>
    %3013 = arith.constant {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} 1.000000e-05 : f32
    %3014 = tensor.splat %3013 {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} : tensor<512xf32>
    %3015 = tensor.empty() : tensor<512xf32>
    %3016 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%294, %3014 : tensor<512xf32>, tensor<512xf32>) outs(%3015 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} {
    ^bb365(%3017: f32, %3018: f32, %3019: f32):
      %3020 = arith.addf %3017, %3018 : f32
      linalg.yield %3020 : f32
    } -> tensor<512xf32>
    %3021 = tensor.empty() : tensor<512xf32>
    %3022 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3016 : tensor<512xf32>) outs(%3021 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} {
    ^bb366(%3023: f32, %3024: f32):
      %3025 = math.rsqrt %3023 : f32
      linalg.yield %3025 : f32
    } -> tensor<512xf32>
    %3026 = tensor.empty() : tensor<1x512x7x7xf32>
    %3027 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3008, %3022 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3026 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} {
    ^bb367(%3028: f32, %3029: f32, %3030: f32):
      %3031 = arith.mulf %3028, %3029 : f32
      linalg.yield %3031 : f32
    } -> tensor<1x512x7x7xf32>
    %3032 = tensor.empty() : tensor<1x512x7x7xf32>
    %3033 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3027, %133 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3032 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} {
    ^bb368(%3034: f32, %3035: f32, %3036: f32):
      %3037 = arith.mulf %3034, %3035 : f32
      linalg.yield %3037 : f32
    } -> tensor<1x512x7x7xf32>
    %3038 = tensor.empty() : tensor<1x512x7x7xf32>
    %3039 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3033, %134 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3038 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_44", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn2"} {
    ^bb369(%3040: f32, %3041: f32, %3042: f32):
      %3043 = arith.addf %3040, %3041 : f32
      linalg.yield %3043 : f32
    } -> tensor<1x512x7x7xf32>
    %3044 = tensor.empty() : tensor<1x512x7x7xf32>
    %3045 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3039 : tensor<1x512x7x7xf32>) outs(%3044 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "minmax_41", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.relu"} {
    ^bb370(%3046: f32, %3047: f32):
      %3048 = arith.constant 0.000000e+00 : f32
      %3049 = arith.maximumf %3046, %3048 : f32
      linalg.yield %3049 : f32
    } -> tensor<1x512x7x7xf32>
    %3050 = tensor.empty() : tensor<512x1x1x1x7x7xf32>
    %3051 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3045 : tensor<1x512x7x7xf32>) outs(%3050 : tensor<512x1x1x1x7x7xf32>) attrs =  {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} {
    ^bb371(%3052: f32, %3053: f32):
      linalg.yield %3052 : f32
    } -> tensor<512x1x1x1x7x7xf32>
    %3054 = tensor.collapse_shape %3051 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<512x1x1x1x7x7xf32> into tensor<25088xf32>
    %3055 = tensor.expand_shape %3054 [[0 : i64, 1 : i64]] output_shape [512, 49] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<25088xf32> into tensor<512x49xf32>
    %3056 = tensor.collapse_shape %135 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<2048x512x1x1xf32> into tensor<1048576xf32>
    %3057 = tensor.expand_shape %3056 [[0 : i64, 1 : i64]] output_shape [2048, 512] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<1048576xf32> into tensor<2048x512xf32>
    %3058 = arith.constant {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} 0.000000e+00 : f32
    %3059 = tensor.splat %3058 {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<2048x49xf32>
    %3060 = linalg.matmul {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} ins(%3057, %3055 : tensor<2048x512xf32>, tensor<512x49xf32>) outs(%3059 : tensor<2048x49xf32>) -> tensor<2048x49xf32>
    %3061 = tensor.collapse_shape %3060 [[0 : i64, 1 : i64]] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<2048x49xf32> into tensor<100352xf32>
    %3062 = tensor.expand_shape %3061 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [2048, 1, 7, 7] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<100352xf32> into tensor<2048x1x7x7xf32>
    %3063 = tensor.collapse_shape %3062 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<2048x1x7x7xf32> into tensor<100352xf32>
    %3064 = tensor.expand_shape %3063 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2048, 7, 7] {prov.region_id = "conv_45", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.conv3"} : tensor<100352xf32> into tensor<1x2048x7x7xf32>
    %3065 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3066 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3064, %296 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3065 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} {
    ^bb372(%3067: f32, %3068: f32, %3069: f32):
      %3070 = arith.subf %3067, %3068 : f32
      linalg.yield %3070 : f32
    } -> tensor<1x2048x7x7xf32>
    %3071 = arith.constant {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} 1.000000e-05 : f32
    %3072 = tensor.splat %3071 {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} : tensor<2048xf32>
    %3073 = tensor.empty() : tensor<2048xf32>
    %3074 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%297, %3072 : tensor<2048xf32>, tensor<2048xf32>) outs(%3073 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} {
    ^bb373(%3075: f32, %3076: f32, %3077: f32):
      %3078 = arith.addf %3075, %3076 : f32
      linalg.yield %3078 : f32
    } -> tensor<2048xf32>
    %3079 = tensor.empty() : tensor<2048xf32>
    %3080 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3074 : tensor<2048xf32>) outs(%3079 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} {
    ^bb374(%3081: f32, %3082: f32):
      %3083 = math.rsqrt %3081 : f32
      linalg.yield %3083 : f32
    } -> tensor<2048xf32>
    %3084 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3085 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3066, %3080 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3084 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} {
    ^bb375(%3086: f32, %3087: f32, %3088: f32):
      %3089 = arith.mulf %3086, %3087 : f32
      linalg.yield %3089 : f32
    } -> tensor<1x2048x7x7xf32>
    %3090 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3091 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3085, %136 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3090 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} {
    ^bb376(%3092: f32, %3093: f32, %3094: f32):
      %3095 = arith.mulf %3092, %3093 : f32
      linalg.yield %3095 : f32
    } -> tensor<1x2048x7x7xf32>
    %3096 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3097 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3091, %137 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3096 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_45", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.bn3"} {
    ^bb377(%3098: f32, %3099: f32, %3100: f32):
      %3101 = arith.addf %3098, %3099 : f32
      linalg.yield %3101 : f32
    } -> tensor<1x2048x7x7xf32>
    %3102 = tensor.empty() : tensor<1024x1x1x1x7x7xf32>
    %3103 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, ((d4 * 2) + d1), ((d5 * 2) + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%2926 : tensor<1x1024x14x14xf32>) outs(%3102 : tensor<1024x1x1x1x7x7xf32>) attrs =  {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} {
    ^bb378(%3104: f32, %3105: f32):
      linalg.yield %3104 : f32
    } -> tensor<1024x1x1x1x7x7xf32>
    %3106 = tensor.collapse_shape %3103 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<1024x1x1x1x7x7xf32> into tensor<50176xf32>
    %3107 = tensor.expand_shape %3106 [[0 : i64, 1 : i64]] output_shape [1024, 49] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<50176xf32> into tensor<1024x49xf32>
    %3108 = tensor.collapse_shape %138 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<2048x1024x1x1xf32> into tensor<2097152xf32>
    %3109 = tensor.expand_shape %3108 [[0 : i64, 1 : i64]] output_shape [2048, 1024] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<2097152xf32> into tensor<2048x1024xf32>
    %3110 = arith.constant {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} 0.000000e+00 : f32
    %3111 = tensor.splat %3110 {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<2048x49xf32>
    %3112 = linalg.matmul {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} ins(%3109, %3107 : tensor<2048x1024xf32>, tensor<1024x49xf32>) outs(%3111 : tensor<2048x49xf32>) -> tensor<2048x49xf32>
    %3113 = tensor.collapse_shape %3112 [[0 : i64, 1 : i64]] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<2048x49xf32> into tensor<100352xf32>
    %3114 = tensor.expand_shape %3113 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [2048, 1, 7, 7] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<100352xf32> into tensor<2048x1x7x7xf32>
    %3115 = tensor.collapse_shape %3114 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<2048x1x7x7xf32> into tensor<100352xf32>
    %3116 = tensor.expand_shape %3115 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2048, 7, 7] {prov.region_id = "conv_46", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.0"} : tensor<100352xf32> into tensor<1x2048x7x7xf32>
    %3117 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3118 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3116, %299 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3117 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} {
    ^bb379(%3119: f32, %3120: f32, %3121: f32):
      %3122 = arith.subf %3119, %3120 : f32
      linalg.yield %3122 : f32
    } -> tensor<1x2048x7x7xf32>
    %3123 = arith.constant {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} 1.000000e-05 : f32
    %3124 = tensor.splat %3123 {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} : tensor<2048xf32>
    %3125 = tensor.empty() : tensor<2048xf32>
    %3126 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%300, %3124 : tensor<2048xf32>, tensor<2048xf32>) outs(%3125 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} {
    ^bb380(%3127: f32, %3128: f32, %3129: f32):
      %3130 = arith.addf %3127, %3128 : f32
      linalg.yield %3130 : f32
    } -> tensor<2048xf32>
    %3131 = tensor.empty() : tensor<2048xf32>
    %3132 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3126 : tensor<2048xf32>) outs(%3131 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} {
    ^bb381(%3133: f32, %3134: f32):
      %3135 = math.rsqrt %3133 : f32
      linalg.yield %3135 : f32
    } -> tensor<2048xf32>
    %3136 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3137 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3118, %3132 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3136 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} {
    ^bb382(%3138: f32, %3139: f32, %3140: f32):
      %3141 = arith.mulf %3138, %3139 : f32
      linalg.yield %3141 : f32
    } -> tensor<1x2048x7x7xf32>
    %3142 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3143 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3137, %139 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3142 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} {
    ^bb383(%3144: f32, %3145: f32, %3146: f32):
      %3147 = arith.mulf %3144, %3145 : f32
      linalg.yield %3147 : f32
    } -> tensor<1x2048x7x7xf32>
    %3148 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3149 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3143, %140 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3148 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_46", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.downsample.1"} {
    ^bb384(%3150: f32, %3151: f32, %3152: f32):
      %3153 = arith.addf %3150, %3151 : f32
      linalg.yield %3153 : f32
    } -> tensor<1x2048x7x7xf32>
    %3154 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3155 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3097, %3149 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%3154 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "add_13", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0"} {
    ^bb385(%3156: f32, %3157: f32, %3158: f32):
      %3159 = arith.addf %3156, %3157 : f32
      linalg.yield %3159 : f32
    } -> tensor<1x2048x7x7xf32>
    %3160 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3161 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3155 : tensor<1x2048x7x7xf32>) outs(%3160 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "minmax_42", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.0.relu"} {
    ^bb386(%3162: f32, %3163: f32):
      %3164 = arith.constant 0.000000e+00 : f32
      %3165 = arith.maximumf %3162, %3164 : f32
      linalg.yield %3165 : f32
    } -> tensor<1x2048x7x7xf32>
    %3166 = tensor.empty() : tensor<2048x1x1x1x7x7xf32>
    %3167 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3161 : tensor<1x2048x7x7xf32>) outs(%3166 : tensor<2048x1x1x1x7x7xf32>) attrs =  {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} {
    ^bb387(%3168: f32, %3169: f32):
      linalg.yield %3168 : f32
    } -> tensor<2048x1x1x1x7x7xf32>
    %3170 = tensor.collapse_shape %3167 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<2048x1x1x1x7x7xf32> into tensor<100352xf32>
    %3171 = tensor.expand_shape %3170 [[0 : i64, 1 : i64]] output_shape [2048, 49] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<100352xf32> into tensor<2048x49xf32>
    %3172 = tensor.collapse_shape %141 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<512x2048x1x1xf32> into tensor<1048576xf32>
    %3173 = tensor.expand_shape %3172 [[0 : i64, 1 : i64]] output_shape [512, 2048] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<1048576xf32> into tensor<512x2048xf32>
    %3174 = arith.constant {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} 0.000000e+00 : f32
    %3175 = tensor.splat %3174 {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<512x49xf32>
    %3176 = linalg.matmul {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} ins(%3173, %3171 : tensor<512x2048xf32>, tensor<2048x49xf32>) outs(%3175 : tensor<512x49xf32>) -> tensor<512x49xf32>
    %3177 = tensor.collapse_shape %3176 [[0 : i64, 1 : i64]] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<512x49xf32> into tensor<25088xf32>
    %3178 = tensor.expand_shape %3177 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 7, 7] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<25088xf32> into tensor<512x1x7x7xf32>
    %3179 = tensor.collapse_shape %3178 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<512x1x7x7xf32> into tensor<25088xf32>
    %3180 = tensor.expand_shape %3179 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 7, 7] {prov.region_id = "conv_47", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv1"} : tensor<25088xf32> into tensor<1x512x7x7xf32>
    %3181 = tensor.empty() : tensor<1x512x7x7xf32>
    %3182 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3180, %302 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3181 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} {
    ^bb388(%3183: f32, %3184: f32, %3185: f32):
      %3186 = arith.subf %3183, %3184 : f32
      linalg.yield %3186 : f32
    } -> tensor<1x512x7x7xf32>
    %3187 = arith.constant {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} 1.000000e-05 : f32
    %3188 = tensor.splat %3187 {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} : tensor<512xf32>
    %3189 = tensor.empty() : tensor<512xf32>
    %3190 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%303, %3188 : tensor<512xf32>, tensor<512xf32>) outs(%3189 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} {
    ^bb389(%3191: f32, %3192: f32, %3193: f32):
      %3194 = arith.addf %3191, %3192 : f32
      linalg.yield %3194 : f32
    } -> tensor<512xf32>
    %3195 = tensor.empty() : tensor<512xf32>
    %3196 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3190 : tensor<512xf32>) outs(%3195 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} {
    ^bb390(%3197: f32, %3198: f32):
      %3199 = math.rsqrt %3197 : f32
      linalg.yield %3199 : f32
    } -> tensor<512xf32>
    %3200 = tensor.empty() : tensor<1x512x7x7xf32>
    %3201 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3182, %3196 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3200 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} {
    ^bb391(%3202: f32, %3203: f32, %3204: f32):
      %3205 = arith.mulf %3202, %3203 : f32
      linalg.yield %3205 : f32
    } -> tensor<1x512x7x7xf32>
    %3206 = tensor.empty() : tensor<1x512x7x7xf32>
    %3207 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3201, %142 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3206 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} {
    ^bb392(%3208: f32, %3209: f32, %3210: f32):
      %3211 = arith.mulf %3208, %3209 : f32
      linalg.yield %3211 : f32
    } -> tensor<1x512x7x7xf32>
    %3212 = tensor.empty() : tensor<1x512x7x7xf32>
    %3213 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3207, %143 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3212 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_47", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn1"} {
    ^bb393(%3214: f32, %3215: f32, %3216: f32):
      %3217 = arith.addf %3214, %3215 : f32
      linalg.yield %3217 : f32
    } -> tensor<1x512x7x7xf32>
    %3218 = tensor.empty() : tensor<1x512x7x7xf32>
    %3219 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3213 : tensor<1x512x7x7xf32>) outs(%3218 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "minmax_43", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.relu"} {
    ^bb394(%3220: f32, %3221: f32):
      %3222 = arith.constant 0.000000e+00 : f32
      %3223 = arith.maximumf %3220, %3222 : f32
      linalg.yield %3223 : f32
    } -> tensor<1x512x7x7xf32>
    %3224 = arith.constant {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} 0.000000e+00 : f32
    %3225 = tensor.splat %3224 {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<1x512x9x9xf32>
    %3226 = "tensor.insert_slice"(%3219, %3225) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 7, 7>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : (tensor<1x512x7x7xf32>, tensor<1x512x9x9xf32>) -> tensor<1x512x9x9xf32>
    %3227 = tensor.empty() : tensor<512x3x3x1x7x7xf32>
    %3228 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3226 : tensor<1x512x9x9xf32>) outs(%3227 : tensor<512x3x3x1x7x7xf32>) attrs =  {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} {
    ^bb395(%3229: f32, %3230: f32):
      linalg.yield %3229 : f32
    } -> tensor<512x3x3x1x7x7xf32>
    %3231 = tensor.collapse_shape %3228 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<512x3x3x1x7x7xf32> into tensor<225792xf32>
    %3232 = tensor.expand_shape %3231 [[0 : i64, 1 : i64]] output_shape [4608, 49] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<225792xf32> into tensor<4608x49xf32>
    %3233 = tensor.collapse_shape %144 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<512x512x3x3xf32> into tensor<2359296xf32>
    %3234 = tensor.expand_shape %3233 [[0 : i64, 1 : i64]] output_shape [512, 4608] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<2359296xf32> into tensor<512x4608xf32>
    %3235 = arith.constant {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} 0.000000e+00 : f32
    %3236 = tensor.splat %3235 {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<512x49xf32>
    %3237 = linalg.matmul {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} ins(%3234, %3232 : tensor<512x4608xf32>, tensor<4608x49xf32>) outs(%3236 : tensor<512x49xf32>) -> tensor<512x49xf32>
    %3238 = tensor.collapse_shape %3237 [[0 : i64, 1 : i64]] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<512x49xf32> into tensor<25088xf32>
    %3239 = tensor.expand_shape %3238 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 7, 7] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<25088xf32> into tensor<512x1x7x7xf32>
    %3240 = tensor.collapse_shape %3239 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<512x1x7x7xf32> into tensor<25088xf32>
    %3241 = tensor.expand_shape %3240 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 7, 7] {prov.region_id = "conv_48", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv2"} : tensor<25088xf32> into tensor<1x512x7x7xf32>
    %3242 = tensor.empty() : tensor<1x512x7x7xf32>
    %3243 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3241, %305 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3242 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} {
    ^bb396(%3244: f32, %3245: f32, %3246: f32):
      %3247 = arith.subf %3244, %3245 : f32
      linalg.yield %3247 : f32
    } -> tensor<1x512x7x7xf32>
    %3248 = arith.constant {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} 1.000000e-05 : f32
    %3249 = tensor.splat %3248 {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} : tensor<512xf32>
    %3250 = tensor.empty() : tensor<512xf32>
    %3251 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%306, %3249 : tensor<512xf32>, tensor<512xf32>) outs(%3250 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} {
    ^bb397(%3252: f32, %3253: f32, %3254: f32):
      %3255 = arith.addf %3252, %3253 : f32
      linalg.yield %3255 : f32
    } -> tensor<512xf32>
    %3256 = tensor.empty() : tensor<512xf32>
    %3257 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3251 : tensor<512xf32>) outs(%3256 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} {
    ^bb398(%3258: f32, %3259: f32):
      %3260 = math.rsqrt %3258 : f32
      linalg.yield %3260 : f32
    } -> tensor<512xf32>
    %3261 = tensor.empty() : tensor<1x512x7x7xf32>
    %3262 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3243, %3257 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3261 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} {
    ^bb399(%3263: f32, %3264: f32, %3265: f32):
      %3266 = arith.mulf %3263, %3264 : f32
      linalg.yield %3266 : f32
    } -> tensor<1x512x7x7xf32>
    %3267 = tensor.empty() : tensor<1x512x7x7xf32>
    %3268 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3262, %145 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3267 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} {
    ^bb400(%3269: f32, %3270: f32, %3271: f32):
      %3272 = arith.mulf %3269, %3270 : f32
      linalg.yield %3272 : f32
    } -> tensor<1x512x7x7xf32>
    %3273 = tensor.empty() : tensor<1x512x7x7xf32>
    %3274 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3268, %146 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3273 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_48", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn2"} {
    ^bb401(%3275: f32, %3276: f32, %3277: f32):
      %3278 = arith.addf %3275, %3276 : f32
      linalg.yield %3278 : f32
    } -> tensor<1x512x7x7xf32>
    %3279 = tensor.empty() : tensor<1x512x7x7xf32>
    %3280 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3274 : tensor<1x512x7x7xf32>) outs(%3279 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "minmax_44", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.relu"} {
    ^bb402(%3281: f32, %3282: f32):
      %3283 = arith.constant 0.000000e+00 : f32
      %3284 = arith.maximumf %3281, %3283 : f32
      linalg.yield %3284 : f32
    } -> tensor<1x512x7x7xf32>
    %3285 = tensor.empty() : tensor<512x1x1x1x7x7xf32>
    %3286 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3280 : tensor<1x512x7x7xf32>) outs(%3285 : tensor<512x1x1x1x7x7xf32>) attrs =  {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} {
    ^bb403(%3287: f32, %3288: f32):
      linalg.yield %3287 : f32
    } -> tensor<512x1x1x1x7x7xf32>
    %3289 = tensor.collapse_shape %3286 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<512x1x1x1x7x7xf32> into tensor<25088xf32>
    %3290 = tensor.expand_shape %3289 [[0 : i64, 1 : i64]] output_shape [512, 49] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<25088xf32> into tensor<512x49xf32>
    %3291 = tensor.collapse_shape %147 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<2048x512x1x1xf32> into tensor<1048576xf32>
    %3292 = tensor.expand_shape %3291 [[0 : i64, 1 : i64]] output_shape [2048, 512] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<1048576xf32> into tensor<2048x512xf32>
    %3293 = arith.constant {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} 0.000000e+00 : f32
    %3294 = tensor.splat %3293 {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<2048x49xf32>
    %3295 = linalg.matmul {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} ins(%3292, %3290 : tensor<2048x512xf32>, tensor<512x49xf32>) outs(%3294 : tensor<2048x49xf32>) -> tensor<2048x49xf32>
    %3296 = tensor.collapse_shape %3295 [[0 : i64, 1 : i64]] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<2048x49xf32> into tensor<100352xf32>
    %3297 = tensor.expand_shape %3296 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [2048, 1, 7, 7] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<100352xf32> into tensor<2048x1x7x7xf32>
    %3298 = tensor.collapse_shape %3297 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<2048x1x7x7xf32> into tensor<100352xf32>
    %3299 = tensor.expand_shape %3298 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2048, 7, 7] {prov.region_id = "conv_49", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.conv3"} : tensor<100352xf32> into tensor<1x2048x7x7xf32>
    %3300 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3301 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3299, %308 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3300 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} {
    ^bb404(%3302: f32, %3303: f32, %3304: f32):
      %3305 = arith.subf %3302, %3303 : f32
      linalg.yield %3305 : f32
    } -> tensor<1x2048x7x7xf32>
    %3306 = arith.constant {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} 1.000000e-05 : f32
    %3307 = tensor.splat %3306 {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} : tensor<2048xf32>
    %3308 = tensor.empty() : tensor<2048xf32>
    %3309 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%309, %3307 : tensor<2048xf32>, tensor<2048xf32>) outs(%3308 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} {
    ^bb405(%3310: f32, %3311: f32, %3312: f32):
      %3313 = arith.addf %3310, %3311 : f32
      linalg.yield %3313 : f32
    } -> tensor<2048xf32>
    %3314 = tensor.empty() : tensor<2048xf32>
    %3315 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3309 : tensor<2048xf32>) outs(%3314 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} {
    ^bb406(%3316: f32, %3317: f32):
      %3318 = math.rsqrt %3316 : f32
      linalg.yield %3318 : f32
    } -> tensor<2048xf32>
    %3319 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3320 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3301, %3315 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3319 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} {
    ^bb407(%3321: f32, %3322: f32, %3323: f32):
      %3324 = arith.mulf %3321, %3322 : f32
      linalg.yield %3324 : f32
    } -> tensor<1x2048x7x7xf32>
    %3325 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3326 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3320, %148 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3325 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} {
    ^bb408(%3327: f32, %3328: f32, %3329: f32):
      %3330 = arith.mulf %3327, %3328 : f32
      linalg.yield %3330 : f32
    } -> tensor<1x2048x7x7xf32>
    %3331 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3332 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3326, %149 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3331 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_49", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.bn3"} {
    ^bb409(%3333: f32, %3334: f32, %3335: f32):
      %3336 = arith.addf %3333, %3334 : f32
      linalg.yield %3336 : f32
    } -> tensor<1x2048x7x7xf32>
    %3337 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3338 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3332, %3161 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%3337 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "add_14", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1"} {
    ^bb410(%3339: f32, %3340: f32, %3341: f32):
      %3342 = arith.addf %3339, %3340 : f32
      linalg.yield %3342 : f32
    } -> tensor<1x2048x7x7xf32>
    %3343 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3344 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3338 : tensor<1x2048x7x7xf32>) outs(%3343 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "minmax_45", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.1.relu"} {
    ^bb411(%3345: f32, %3346: f32):
      %3347 = arith.constant 0.000000e+00 : f32
      %3348 = arith.maximumf %3345, %3347 : f32
      linalg.yield %3348 : f32
    } -> tensor<1x2048x7x7xf32>
    %3349 = tensor.empty() : tensor<2048x1x1x1x7x7xf32>
    %3350 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3344 : tensor<1x2048x7x7xf32>) outs(%3349 : tensor<2048x1x1x1x7x7xf32>) attrs =  {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} {
    ^bb412(%3351: f32, %3352: f32):
      linalg.yield %3351 : f32
    } -> tensor<2048x1x1x1x7x7xf32>
    %3353 = tensor.collapse_shape %3350 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<2048x1x1x1x7x7xf32> into tensor<100352xf32>
    %3354 = tensor.expand_shape %3353 [[0 : i64, 1 : i64]] output_shape [2048, 49] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<100352xf32> into tensor<2048x49xf32>
    %3355 = tensor.collapse_shape %150 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<512x2048x1x1xf32> into tensor<1048576xf32>
    %3356 = tensor.expand_shape %3355 [[0 : i64, 1 : i64]] output_shape [512, 2048] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<1048576xf32> into tensor<512x2048xf32>
    %3357 = arith.constant {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} 0.000000e+00 : f32
    %3358 = tensor.splat %3357 {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<512x49xf32>
    %3359 = linalg.matmul {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} ins(%3356, %3354 : tensor<512x2048xf32>, tensor<2048x49xf32>) outs(%3358 : tensor<512x49xf32>) -> tensor<512x49xf32>
    %3360 = tensor.collapse_shape %3359 [[0 : i64, 1 : i64]] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<512x49xf32> into tensor<25088xf32>
    %3361 = tensor.expand_shape %3360 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 7, 7] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<25088xf32> into tensor<512x1x7x7xf32>
    %3362 = tensor.collapse_shape %3361 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<512x1x7x7xf32> into tensor<25088xf32>
    %3363 = tensor.expand_shape %3362 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 7, 7] {prov.region_id = "conv_50", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv1"} : tensor<25088xf32> into tensor<1x512x7x7xf32>
    %3364 = tensor.empty() : tensor<1x512x7x7xf32>
    %3365 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3363, %311 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3364 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} {
    ^bb413(%3366: f32, %3367: f32, %3368: f32):
      %3369 = arith.subf %3366, %3367 : f32
      linalg.yield %3369 : f32
    } -> tensor<1x512x7x7xf32>
    %3370 = arith.constant {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} 1.000000e-05 : f32
    %3371 = tensor.splat %3370 {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} : tensor<512xf32>
    %3372 = tensor.empty() : tensor<512xf32>
    %3373 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%312, %3371 : tensor<512xf32>, tensor<512xf32>) outs(%3372 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} {
    ^bb414(%3374: f32, %3375: f32, %3376: f32):
      %3377 = arith.addf %3374, %3375 : f32
      linalg.yield %3377 : f32
    } -> tensor<512xf32>
    %3378 = tensor.empty() : tensor<512xf32>
    %3379 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3373 : tensor<512xf32>) outs(%3378 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} {
    ^bb415(%3380: f32, %3381: f32):
      %3382 = math.rsqrt %3380 : f32
      linalg.yield %3382 : f32
    } -> tensor<512xf32>
    %3383 = tensor.empty() : tensor<1x512x7x7xf32>
    %3384 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3365, %3379 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3383 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} {
    ^bb416(%3385: f32, %3386: f32, %3387: f32):
      %3388 = arith.mulf %3385, %3386 : f32
      linalg.yield %3388 : f32
    } -> tensor<1x512x7x7xf32>
    %3389 = tensor.empty() : tensor<1x512x7x7xf32>
    %3390 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3384, %151 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3389 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} {
    ^bb417(%3391: f32, %3392: f32, %3393: f32):
      %3394 = arith.mulf %3391, %3392 : f32
      linalg.yield %3394 : f32
    } -> tensor<1x512x7x7xf32>
    %3395 = tensor.empty() : tensor<1x512x7x7xf32>
    %3396 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3390, %152 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3395 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_50", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn1"} {
    ^bb418(%3397: f32, %3398: f32, %3399: f32):
      %3400 = arith.addf %3397, %3398 : f32
      linalg.yield %3400 : f32
    } -> tensor<1x512x7x7xf32>
    %3401 = tensor.empty() : tensor<1x512x7x7xf32>
    %3402 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3396 : tensor<1x512x7x7xf32>) outs(%3401 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "minmax_46", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.relu"} {
    ^bb419(%3403: f32, %3404: f32):
      %3405 = arith.constant 0.000000e+00 : f32
      %3406 = arith.maximumf %3403, %3405 : f32
      linalg.yield %3406 : f32
    } -> tensor<1x512x7x7xf32>
    %3407 = arith.constant {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} 0.000000e+00 : f32
    %3408 = tensor.splat %3407 {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<1x512x9x9xf32>
    %3409 = "tensor.insert_slice"(%3402, %3408) <{static_offsets = array<i64: 0, 0, 1, 1>, static_sizes = array<i64: 1, 512, 7, 7>, static_strides = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1, 0, 0, 0>}> {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : (tensor<1x512x7x7xf32>, tensor<1x512x9x9xf32>) -> tensor<1x512x9x9xf32>
    %3410 = tensor.empty() : tensor<512x3x3x1x7x7xf32>
    %3411 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3409 : tensor<1x512x9x9xf32>) outs(%3410 : tensor<512x3x3x1x7x7xf32>) attrs =  {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} {
    ^bb420(%3412: f32, %3413: f32):
      linalg.yield %3412 : f32
    } -> tensor<512x3x3x1x7x7xf32>
    %3414 = tensor.collapse_shape %3411 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<512x3x3x1x7x7xf32> into tensor<225792xf32>
    %3415 = tensor.expand_shape %3414 [[0 : i64, 1 : i64]] output_shape [4608, 49] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<225792xf32> into tensor<4608x49xf32>
    %3416 = tensor.collapse_shape %153 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<512x512x3x3xf32> into tensor<2359296xf32>
    %3417 = tensor.expand_shape %3416 [[0 : i64, 1 : i64]] output_shape [512, 4608] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<2359296xf32> into tensor<512x4608xf32>
    %3418 = arith.constant {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} 0.000000e+00 : f32
    %3419 = tensor.splat %3418 {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<512x49xf32>
    %3420 = linalg.matmul {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} ins(%3417, %3415 : tensor<512x4608xf32>, tensor<4608x49xf32>) outs(%3419 : tensor<512x49xf32>) -> tensor<512x49xf32>
    %3421 = tensor.collapse_shape %3420 [[0 : i64, 1 : i64]] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<512x49xf32> into tensor<25088xf32>
    %3422 = tensor.expand_shape %3421 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [512, 1, 7, 7] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<25088xf32> into tensor<512x1x7x7xf32>
    %3423 = tensor.collapse_shape %3422 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<512x1x7x7xf32> into tensor<25088xf32>
    %3424 = tensor.expand_shape %3423 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 512, 7, 7] {prov.region_id = "conv_51", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv2"} : tensor<25088xf32> into tensor<1x512x7x7xf32>
    %3425 = tensor.empty() : tensor<1x512x7x7xf32>
    %3426 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3424, %314 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3425 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} {
    ^bb421(%3427: f32, %3428: f32, %3429: f32):
      %3430 = arith.subf %3427, %3428 : f32
      linalg.yield %3430 : f32
    } -> tensor<1x512x7x7xf32>
    %3431 = arith.constant {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} 1.000000e-05 : f32
    %3432 = tensor.splat %3431 {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} : tensor<512xf32>
    %3433 = tensor.empty() : tensor<512xf32>
    %3434 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%315, %3432 : tensor<512xf32>, tensor<512xf32>) outs(%3433 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} {
    ^bb422(%3435: f32, %3436: f32, %3437: f32):
      %3438 = arith.addf %3435, %3436 : f32
      linalg.yield %3438 : f32
    } -> tensor<512xf32>
    %3439 = tensor.empty() : tensor<512xf32>
    %3440 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3434 : tensor<512xf32>) outs(%3439 : tensor<512xf32>) attrs =  {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} {
    ^bb423(%3441: f32, %3442: f32):
      %3443 = math.rsqrt %3441 : f32
      linalg.yield %3443 : f32
    } -> tensor<512xf32>
    %3444 = tensor.empty() : tensor<1x512x7x7xf32>
    %3445 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3426, %3440 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3444 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} {
    ^bb424(%3446: f32, %3447: f32, %3448: f32):
      %3449 = arith.mulf %3446, %3447 : f32
      linalg.yield %3449 : f32
    } -> tensor<1x512x7x7xf32>
    %3450 = tensor.empty() : tensor<1x512x7x7xf32>
    %3451 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3445, %154 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3450 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} {
    ^bb425(%3452: f32, %3453: f32, %3454: f32):
      %3455 = arith.mulf %3452, %3453 : f32
      linalg.yield %3455 : f32
    } -> tensor<1x512x7x7xf32>
    %3456 = tensor.empty() : tensor<1x512x7x7xf32>
    %3457 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3451, %155 : tensor<1x512x7x7xf32>, tensor<512xf32>) outs(%3456 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "batch_norm_51", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn2"} {
    ^bb426(%3458: f32, %3459: f32, %3460: f32):
      %3461 = arith.addf %3458, %3459 : f32
      linalg.yield %3461 : f32
    } -> tensor<1x512x7x7xf32>
    %3462 = tensor.empty() : tensor<1x512x7x7xf32>
    %3463 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3457 : tensor<1x512x7x7xf32>) outs(%3462 : tensor<1x512x7x7xf32>) attrs =  {prov.region_id = "minmax_47", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.relu"} {
    ^bb427(%3464: f32, %3465: f32):
      %3466 = arith.constant 0.000000e+00 : f32
      %3467 = arith.maximumf %3464, %3466 : f32
      linalg.yield %3467 : f32
    } -> tensor<1x512x7x7xf32>
    %3468 = tensor.empty() : tensor<512x1x1x1x7x7xf32>
    %3469 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d0, (d4 + d1), (d5 + d2))>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%3463 : tensor<1x512x7x7xf32>) outs(%3468 : tensor<512x1x1x1x7x7xf32>) attrs =  {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} {
    ^bb428(%3470: f32, %3471: f32):
      linalg.yield %3470 : f32
    } -> tensor<512x1x1x1x7x7xf32>
    %3472 = tensor.collapse_shape %3469 [[0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64, 5 : i64]] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<512x1x1x1x7x7xf32> into tensor<25088xf32>
    %3473 = tensor.expand_shape %3472 [[0 : i64, 1 : i64]] output_shape [512, 49] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<25088xf32> into tensor<512x49xf32>
    %3474 = tensor.collapse_shape %156 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<2048x512x1x1xf32> into tensor<1048576xf32>
    %3475 = tensor.expand_shape %3474 [[0 : i64, 1 : i64]] output_shape [2048, 512] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<1048576xf32> into tensor<2048x512xf32>
    %3476 = arith.constant {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} 0.000000e+00 : f32
    %3477 = tensor.splat %3476 {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<2048x49xf32>
    %3478 = linalg.matmul {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} ins(%3475, %3473 : tensor<2048x512xf32>, tensor<512x49xf32>) outs(%3477 : tensor<2048x49xf32>) -> tensor<2048x49xf32>
    %3479 = tensor.collapse_shape %3478 [[0 : i64, 1 : i64]] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<2048x49xf32> into tensor<100352xf32>
    %3480 = tensor.expand_shape %3479 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [2048, 1, 7, 7] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<100352xf32> into tensor<2048x1x7x7xf32>
    %3481 = tensor.collapse_shape %3480 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<2048x1x7x7xf32> into tensor<100352xf32>
    %3482 = tensor.expand_shape %3481 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2048, 7, 7] {prov.region_id = "conv_52", prov.family = "contraction", prov.conv_path = "im2col_matmul", prov._pattern_hint = "convolution_im2col_matmul", prov.op = "convolution_im2col_matmul", prov.aten = "aten.conv2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.conv3"} : tensor<100352xf32> into tensor<1x2048x7x7xf32>
    %3483 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3484 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3482, %317 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3483 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} {
    ^bb429(%3485: f32, %3486: f32, %3487: f32):
      %3488 = arith.subf %3485, %3486 : f32
      linalg.yield %3488 : f32
    } -> tensor<1x2048x7x7xf32>
    %3489 = arith.constant {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} 1.000000e-05 : f32
    %3490 = tensor.splat %3489 {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} : tensor<2048xf32>
    %3491 = tensor.empty() : tensor<2048xf32>
    %3492 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%318, %3490 : tensor<2048xf32>, tensor<2048xf32>) outs(%3491 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} {
    ^bb430(%3493: f32, %3494: f32, %3495: f32):
      %3496 = arith.addf %3493, %3494 : f32
      linalg.yield %3496 : f32
    } -> tensor<2048xf32>
    %3497 = tensor.empty() : tensor<2048xf32>
    %3498 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3492 : tensor<2048xf32>) outs(%3497 : tensor<2048xf32>) attrs =  {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} {
    ^bb431(%3499: f32, %3500: f32):
      %3501 = math.rsqrt %3499 : f32
      linalg.yield %3501 : f32
    } -> tensor<2048xf32>
    %3502 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3503 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3484, %3498 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3502 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} {
    ^bb432(%3504: f32, %3505: f32, %3506: f32):
      %3507 = arith.mulf %3504, %3505 : f32
      linalg.yield %3507 : f32
    } -> tensor<1x2048x7x7xf32>
    %3508 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3509 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3503, %157 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3508 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} {
    ^bb433(%3510: f32, %3511: f32, %3512: f32):
      %3513 = arith.mulf %3510, %3511 : f32
      linalg.yield %3513 : f32
    } -> tensor<1x2048x7x7xf32>
    %3514 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3515 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3509, %158 : tensor<1x2048x7x7xf32>, tensor<2048xf32>) outs(%3514 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "batch_norm_52", prov.family = "normalization", prov._pattern_hint = "batch_norm", prov.op = "batch_norm", prov.aten = "aten.batch_norm.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.bn3"} {
    ^bb434(%3516: f32, %3517: f32, %3518: f32):
      %3519 = arith.addf %3516, %3517 : f32
      linalg.yield %3519 : f32
    } -> tensor<1x2048x7x7xf32>
    %3520 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3521 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3515, %3344 : tensor<1x2048x7x7xf32>, tensor<1x2048x7x7xf32>) outs(%3520 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "add_15", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2"} {
    ^bb435(%3522: f32, %3523: f32, %3524: f32):
      %3525 = arith.addf %3522, %3523 : f32
      linalg.yield %3525 : f32
    } -> tensor<1x2048x7x7xf32>
    %3526 = tensor.empty() : tensor<1x2048x7x7xf32>
    %3527 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%3521 : tensor<1x2048x7x7xf32>) outs(%3526 : tensor<1x2048x7x7xf32>) attrs =  {prov.region_id = "minmax_48", prov.family = "minmax", prov._pattern_hint = "minmax", prov.op = "minmax", prov.aten = "aten.relu_.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.layer4.2.relu"} {
    ^bb436(%3528: f32, %3529: f32):
      %3530 = arith.constant 0.000000e+00 : f32
      %3531 = arith.maximumf %3528, %3530 : f32
      linalg.yield %3531 : f32
    } -> tensor<1x2048x7x7xf32>
    %3532 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "adaptive_avg_pool2d", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} 0.000000e+00 : f32
    %3533 = tensor.splat %3532 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "adaptive_avg_pool2d", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} : tensor<1x2048xf32>
    %3534 = linalg.reduce ins(%3527:tensor<1x2048x7x7xf32>) outs(%3533:tensor<1x2048xf32>) dimensions = [2, 3]
    (%3535: f32, %3536: f32) {
      %3537 = arith.addf %3535, %3536 : f32
      linalg.yield %3537 : f32
    }
    %3538 = arith.constant {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "adaptive_avg_pool2d", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} 4.900000e+01 : f32
    %3539 = tensor.splat %3538 {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "adaptive_avg_pool2d", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} : tensor<1x2048xf32>
    %3540 = tensor.empty() : tensor<1x2048xf32>
    %3541 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3534, %3539 : tensor<1x2048xf32>, tensor<1x2048xf32>) outs(%3540 : tensor<1x2048xf32>) attrs =  {prov.region_id = "reduce_0", prov.family = "reduce", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "adaptive_avg_pool2d", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} {
    ^bb437(%3542: f32, %3543: f32, %3544: f32):
      %3545 = arith.divf %3542, %3543 : f32
      linalg.yield %3545 : f32
    } -> tensor<1x2048xf32>
    %3546 = tensor.collapse_shape %3541 [[0 : i64, 1 : i64]] {prov.region_id = "reduce_0", prov.family = "layout", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "reshape", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} : tensor<1x2048xf32> into tensor<2048xf32>
    %3547 = tensor.expand_shape %3546 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] output_shape [1, 2048, 1, 1] {prov.region_id = "reduce_0", prov.family = "layout", prov._pattern_hint = "adaptive_avg_pool2d", prov.op = "reshape", prov.aten = "aten.adaptive_avg_pool2d.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.avgpool"} : tensor<2048xf32> into tensor<1x2048x1x1xf32>
    %3548 = tensor.collapse_shape %3547 [[0 : i64, 1 : i64, 2 : i64, 3 : i64]] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<1x2048x1x1xf32> into tensor<2048xf32>
    %3549 = tensor.expand_shape %3548 [[0 : i64, 1 : i64]] output_shape [1, 2048] {prov.region_id = "view_0", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.flatten.using_ints", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model"} : tensor<2048xf32> into tensor<1x2048xf32>
    %3550 = tensor.empty() : tensor<1xf32>
    %3551 = arith.constant 0.000000e+00 : f32
    %3552 = linalg.fill ins(%3551 : f32) outs(%3550 : tensor<1xf32>) -> tensor<1xf32>
    %3553 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%3549 : tensor<1x2048xf32>) outs(%3552 : tensor<1xf32>) attrs =  {prov.region_id = "torchao_choose_qparams_affine_default_0", prov.dispatch_id = "torchao_choose_qparams_affine_default_0", prov.transforms = "torchao_choose_qparams_affine"} {
    ^bb438(%3554: f32, %3555: f32):
      %3556 = math.absf %3554 : f32
      %3557 = arith.maximumf %3555, %3556 : f32
      linalg.yield %3557 : f32
    } -> tensor<1xf32>
    %3558 = tensor.empty() : tensor<1xf32>
    %3559 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%3553 : tensor<1xf32>) outs(%3558 : tensor<1xf32>) attrs =  {prov.region_id = "torchao_choose_qparams_affine_default_0", prov.dispatch_id = "torchao_choose_qparams_affine_default_0", prov.transforms = "torchao_choose_qparams_affine"} {
    ^bb439(%3560: f32, %3561: f32):
      %3562 = arith.constant 1.270000e+02 : f32
      %3563 = arith.constant 1.000000e-05 : f32
      %3564 = arith.divf %3560, %3562 : f32
      %3565 = arith.maximumf %3564, %3563 : f32
      linalg.yield %3565 : f32
    } -> tensor<1xf32>
    %3566 = tensor.empty() : tensor<1x2048xi8>
    %3567 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3549, %3559 : tensor<1x2048xf32>, tensor<1xf32>) outs(%3566 : tensor<1x2048xi8>) attrs =  {prov.region_id = "torchao_quantize_affine_default_0", prov.dispatch_id = "torchao_quantize_affine_default_0", prov.transforms = "torchao_quantize_affine"} {
    ^bb440(%3568: f32, %3569: f32, %3570: i8):
      %3571 = arith.constant 1.000000e+00 : f32
      %3572 = arith.constant -1.270000e+02 : f32
      %3573 = arith.constant 1.270000e+02 : f32
      %3574 = arith.divf %3571, %3569 : f32
      %3575 = arith.mulf %3568, %3574 : f32
      %3576 = math.roundeven %3575 : f32
      %3577 = arith.maximumf %3576, %3572 : f32
      %3578 = arith.minimumf %3577, %3573 : f32
      %3579 = arith.fptosi %3578 : f32 to i8
      linalg.yield %3579 : i8
    } -> tensor<1x2048xi8>
    %3580 = tensor.empty() : tensor<2048x1000xi8>
    %3581 = linalg.transpose ins(%323:tensor<1000x2048xi8>) outs(%3580:tensor<2048x1000xi8>) permutation = [1, 0]
    %3582 = tensor.expand_shape %3559 [[0 : i64, 1 : i64]] output_shape [1, 1] {prov.region_id = "view_2", prov._pattern_hint = "view", prov.op = "view", prov.family = "layout", prov.aten = "aten.reshape.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.fc"} : tensor<1xf32> into tensor<1x1xf32>
    %3583 = tensor.empty() : tensor<1x1000xf32>
    %3584 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3582 : tensor<1x1xf32>) outs(%3583 : tensor<1x1000xf32>) attrs =  {prov.region_id = "expand_0", prov._pattern_hint = "expand", prov.op = "expand", prov.family = "layout", prov.aten = "aten.expand.default", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.fc"} {
    ^bb441(%3585: f32, %3586: f32):
      linalg.yield %3585 : f32
    } -> tensor<1x1000xf32>
    %3587 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32", prov.module = "model", prov.fqn = "model.fc"} 0 : i32
    %3588 = tensor.splat %3587 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32", prov.module = "model", prov.fqn = "model.fc"} : tensor<1x1000xi32>
    %3589 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3567, %3581 : tensor<1x2048xi8>, tensor<2048x1000xi8>) outs(%3588 : tensor<1x1000xi32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32", prov.module = "model", prov.fqn = "model.fc", prov.quant_inner_1 = "model.fc.weight.original_weight_tensor.tensor_impl.int_data"} {
    ^bb442(%3590: i8, %3591: i8, %3592: i32):
      %3593 = arith.extsi %3590 : i8 to i32
      %3594 = arith.extsi %3591 : i8 to i32
      %3595 = arith.muli %3593, %3594 : i32
      %3596 = arith.addi %3592, %3595 : i32
      linalg.yield %3596 : i32
    } -> tensor<1x1000xi32>
    %3597 = tensor.empty() : tensor<1x1000xf32>
    %3598 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3589 : tensor<1x1000xi32>) outs(%3597 : tensor<1x1000xf32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten.to.dtype", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.fc"} {
    ^bb443(%3599: i32, %3600: f32):
      %3601 = arith.sitofp %3599 : i32 to f32
      linalg.yield %3601 : f32
    } -> tensor<1x1000xf32>
    %3602 = tensor.empty() : tensor<1x1000xf32>
    %3603 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3598, %3584 : tensor<1x1000xf32>, tensor<1x1000xf32>) outs(%3602 : tensor<1x1000xf32>) attrs =  {prov.region_id = "mul_0", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.fc"} {
    ^bb444(%3604: f32, %3605: f32, %3606: f32):
      %3607 = arith.mulf %3604, %3605 : f32
      linalg.yield %3607 : f32
    } -> tensor<1x1000xf32>
    %3608 = tensor.empty() : tensor<1x1000xf32>
    %3609 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3603, %324 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%3608 : tensor<1x1000xf32>) attrs =  {prov.region_id = "mul_1", prov._pattern_hint = "mul", prov.op = "mul", prov.family = "elementwise", prov.aten = "aten.mul.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.fc", prov.quant_inner_1 = "model.fc.weight.original_weight_tensor.tensor_impl.scale"} {
    ^bb445(%3610: f32, %3611: f32, %3612: f32):
      %3613 = arith.mulf %3610, %3611 : f32
      linalg.yield %3613 : f32
    } -> tensor<1x1000xf32>
    %3614 = tensor.empty() : tensor<1x1000xf32>
    %3615 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3609, %160 : tensor<1x1000xf32>, tensor<1000xf32>) outs(%3614 : tensor<1x1000xf32>) attrs =  {prov.region_id = "add_16", prov._pattern_hint = "add", prov.op = "add", prov.family = "elementwise", prov.aten = "aten.add_.Tensor", prov.orig_dtype = "float32", prov.module = "model", prov.fqn = "model.fc"} {
    ^bb446(%3616: f32, %3617: f32, %3618: f32):
      %3619 = arith.addf %3616, %3617 : f32
      linalg.yield %3619 : f32
    } -> tensor<1x1000xf32>
    func.return %3615 : tensor<1x1000xf32>
  }
}