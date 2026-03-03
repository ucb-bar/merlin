module {
  func.func @main_graph(%arg0: !torch.vtensor<[1,10],f32>) -> !torch.vtensor<[1,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "onnx.quantize", torch.onnx_meta.producer_version = "0.1.0"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0251800455> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0155197214> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00248654163> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc1.weight_quantized> : tensor<32x10xsi8>} : () -> !torch.vtensor<[32,10],si8> 
    %7 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0155197214> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0078596957> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %12 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00139069092> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %13 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc2.weight_quantized> : tensor<32x32xsi8>} : () -> !torch.vtensor<[32,32],si8> 
    %14 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %15 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0078596957> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %16 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %17 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00234139641> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %18 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %19 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00138537993> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %20 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc3.weight_quantized> : tensor<2x32xsi8>} : () -> !torch.vtensor<[2,32],si8> 
    %21 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc1.bias_quantized> : tensor<32xsi32>} : () -> !torch.vtensor<[32],si32> 
    %22 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc1.bias_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %23 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32> 
    %24 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc2.bias_quantized> : tensor<32xsi32>} : () -> !torch.vtensor<[32],si32> 
    %25 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc2.bias_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %26 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32> 
    %27 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc3.bias_quantized> : tensor<2xsi32>} : () -> !torch.vtensor<[2],si32> 
    %28 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc3.bias_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %29 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32> 
    %none = torch.constant.none
    %30 = torch.operator "onnx.DequantizeLinear"(%21, %22, %23) : (!torch.vtensor<[32],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[32],f32> 
    %31 = torch.operator "onnx.DequantizeLinear"(%6, %5, %4) : (!torch.vtensor<[32,10],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[32,10],f32> 
    %32 = torch.operator "onnx.DequantizeLinear"(%24, %25, %26) : (!torch.vtensor<[32],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[32],f32> 
    %33 = torch.operator "onnx.DequantizeLinear"(%13, %12, %11) : (!torch.vtensor<[32,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[32,32],f32> 
    %34 = torch.operator "onnx.DequantizeLinear"(%27, %28, %29) : (!torch.vtensor<[2],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[2],f32> 
    %35 = torch.operator "onnx.DequantizeLinear"(%20, %19, %18) : (!torch.vtensor<[2,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[2,32],f32> 
    %36 = torch.operator "onnx.QuantizeLinear"(%arg0, %1, %0) : (!torch.vtensor<[1,10],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,10],si8> 
    %37 = torch.operator "onnx.DequantizeLinear"(%36, %1, %0) : (!torch.vtensor<[1,10],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,10],f32> 
    %38 = torch.operator "onnx.Gemm"(%37, %31, %30) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[1,10],f32>, !torch.vtensor<[32,10],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[1,32],f32> 
    %39 = torch.operator "onnx.QuantizeLinear"(%38, %3, %2) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],si8> 
    %40 = torch.operator "onnx.DequantizeLinear"(%39, %3, %2) : (!torch.vtensor<[1,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],f32> 
    %41 = torch.operator "onnx.Relu"(%40) : (!torch.vtensor<[1,32],f32>) -> !torch.vtensor<[1,32],f32> 
    %42 = torch.operator "onnx.QuantizeLinear"(%41, %8, %7) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],si8> 
    %43 = torch.operator "onnx.DequantizeLinear"(%42, %8, %7) : (!torch.vtensor<[1,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],f32> 
    %44 = torch.operator "onnx.Gemm"(%43, %33, %32) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[32,32],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[1,32],f32> 
    %45 = torch.operator "onnx.QuantizeLinear"(%44, %10, %9) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],si8> 
    %46 = torch.operator "onnx.DequantizeLinear"(%45, %10, %9) : (!torch.vtensor<[1,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],f32> 
    %47 = torch.operator "onnx.Relu"(%46) : (!torch.vtensor<[1,32],f32>) -> !torch.vtensor<[1,32],f32> 
    %48 = torch.operator "onnx.QuantizeLinear"(%47, %15, %14) : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],si8> 
    %49 = torch.operator "onnx.DequantizeLinear"(%48, %15, %14) : (!torch.vtensor<[1,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32],f32> 
    %50 = torch.operator "onnx.Gemm"(%49, %35, %34) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[1,32],f32>, !torch.vtensor<[2,32],f32>, !torch.vtensor<[2],f32>) -> !torch.vtensor<[1,2],f32> 
    %51 = torch.operator "onnx.QuantizeLinear"(%50, %17, %16) : (!torch.vtensor<[1,2],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,2],si8> 
    %52 = torch.operator "onnx.DequantizeLinear"(%51, %17, %16) : (!torch.vtensor<[1,2],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,2],f32> 
    return %52 : !torch.vtensor<[1,2],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _fc1.weight_quantized: "0x08000000F15E9B63DD9D9FD763409FB3B433F3484CB53E29320A8DDA1C3973C26FA2191E56BD698B33AAA3CD51549E31829ACB91892426B3CACD56AD6EAA1C3B6A8E9E64888BA914A1F48BD1F58FF0E54BD2DB2D08A173C80F891013656E433C82B22A209F732650AE305E3E87F8942890A14E01B9A5E5D0BFA166EFA5853BD1235577D306C304D095D5888707D37FFCBAAB5393F66659F829818808695E7EFBFA477512D81A382F3A102936C2488727BE71F1B6D447758A46D8045FA71560750176ADB6AE1893B72FD0ECB49EF01600FCACC9F0741322CD493ECE8BA4C6ED94AA98C9A6A0339A48F8A3D99D6356A83EEFC6EC3A233A8C7CBAA2329226A85BF73CC079B68F0B5AA6E2FA1328E00431D66EF107D3D7330562E4F1063CB3AA579819664DDC423BD2E5CEF114A86775A960A22A412A82CFACE0673EFC905E654FBFBAB0A369",
      _fc2.weight_quantized: "0x08000000310CD13842B6E8E2A9F7B5C01FCAFC9165EDD1A3819F6FADB7DCBF5C51E972252A1942882FB94D8EB574A31F0DABA44975F51BD789EC6EE29E7B204513F54CC4DF0D3FD68A15A99387527971D1FABADCEDCFD5ACE4C9666246B653D0DDD40FA75D4EDF151916DD2175E0B2D67B9AFB69A328B4E9F0A2B6DE2D58F86C448512563306B37C7E0850F9FF242146FD07BE3FF166F768932E9B4227ECEBE9EC4DEAF470518C06DA0A832DD8DA77B7D921121AED660A0187EE6A9711EAE6D268B99A58BE5AAC7E992F645FA0C5CC34C2DD3953D9B62406402DA030923C08245654B4C3E4CCE6066C56D9220EC493757A2E06A07CCA6D9BFDE0B630B22F40E08BD9B3921BE91E62A05E4A0F3CF878BA5FA70AABC814BD6EAF77752C55310F54174A81C1B693346E5AD762BB12C81790039284C7357E00DE8FA9507C8DF82BF909056553D19B66B10F9441FAEAC00DE1AD2272FC64C07AE3C64732CEFEB4F48FE78FE3BDE36DAB19237D86DEF832E5464604342AD7FFC1FE016D664EDA52FFE2253365002DCC994C79DA30FEC39B70527F435DDE8DCB47C8DFF05DDE7F35B02FD60528E5A820FD408ACC92436E08456E3D3B2A9AA55B0E8A85F838BF6A9AAC62E7DAE2869E327B1FB942D6C1B79274FA1F9D158D0E6D38B5CB8C88969D364F8B846FCC5B12B1737DB35E81180C05D8CB473CCAB359A13BDEA4F1DD45A3A09260451E836F0D30D6C41C08588BD79765CAB0D8F5E2DD105D9513DF41CA4C044D246A87C1F00758794CB48F17076265305BD210FAD4902F9B5289616D8AE0AD741EC9A305E18108124A881C0C09C6388AC20A3297247BF3585FAEACBC96B421C4C296ABC4D721DE52048A8DC1B112A93602696A759A79BEF7DA7E5955C25CE81B47F64DF1BE4014CCC396B82BE58FE1E890F7C52089BDAD5CFBEEFFD673F32BA8BED7F871A0C5E9E0360E496CAF9ED041A1B6C057BEFF92944E5E97D0F0F976AB69FFA88EA75B16EA0490B8CC3A88FCA7C47DC8A16AB8B0E5176FBFC72BD4A5F64C2847ED08370C0178048EB5DC3853C791CEB8A36104F32D5B3D72CCB6FDEE25A0C20987D67B826533A8EF97182663EBFB3630DFC8CD3D6552707963323211F2C5E0C919B2A731634D754261F05D6903694174B3D5B0031A542014D498DAA534286C6D70EC5982F133C4263DBE0634B638FFFB7DDE05CF1462FB9F8D8A702D06865A1AE31544B58F0C5A7B2A75171DC7A02F1652DE3D16F822B40CDF9D238CB6B56359671F69BC8D16C66B58D3B7F5870F936514BD8E8AC81D93A7B086F7E1ED6C355D4C3E178670BEA485DD1523D9507D2D7447855999299D8750AB287E0051360655D4B6C22643C4CF65C117C2E119433F141E9E654FDE99A4669F092D6348D37A0B6AE1E811D1312015AD97D826845FA18C97B74E77A2653C4A7E410FDB98B07BA038B6",
      _fc3.weight_quantized: "0x0800000009FF814A089E7B90127711A1A2FFCCEFEB6AA728AF5BCDEE6E5A3DEA4FDD0DD03306FEA8AFFD508453D10BF09AEF429FD770BBC9ED1879ADD5D27882545F75BE",
      _fc1.bias_quantized: "0x08000000BE10000038F3FFFFA507000099070000F5110000A8F1FFFF57040000C003000031010000AFF8FFFF8CFFFFFF4CEEFFFFA2EDFFFF97F7FFFF91050000B6040000ED0F00008BFAFFFFE8100000280C0000330500006A080000A30900001712000005EDFFFF9A0A0000D9F2FFFFA0FFFFFF84FFFFFF3E0600009309000084060000",
      _fc1.bias_quantized_scale: "0x08000000264E8338",
      _fc2.bias_quantized: "0x08000000950F0000F21C000066E6FFFF7E1C000029EAFFFF32F2FFFF82FBFFFF9FE6FFFFA91A0000E3E4FFFFE204000027FFFFFF6E06000005E3FFFF7316000027EDFFFF07FBFFFF6D08000034EBFFFF97E1FFFF67F4FFFFA0100000BB1B000083020000B8E4FFFF4DE9FFFFBDE3FFFFBF060000F00A0000511C0000EC09000012030000",
      _fc2.bias_quantized_scale: "0x080000006E0DB537",
      _fc3.bias_quantized: "0x0800000000140000FB0B0000",
      _fc3.bias_quantized_scale: "0x0800000075AE3637"
    }
  }
#-}

