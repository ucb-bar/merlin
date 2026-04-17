module {
  func.func @main_graph(%arg0: !torch.vtensor<[16,16],f32>) -> !torch.vtensor<[16,16],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "onnx.quantize", torch.onnx_meta.producer_version = "0.1.0"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0325061455> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0256808829> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00196764735> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc1.weight_quantized> : tensor<32x16xsi8>} : () -> !torch.vtensor<[32,16],si8>
    %7 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %8 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0256808829> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %10 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0100460853> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %11 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %12 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00138904818> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %13 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc2.weight_quantized> : tensor<32x32xsi8>} : () -> !torch.vtensor<[32,32],si8>
    %14 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %15 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0100460853> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %16 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %17 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00443977816> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %18 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %19 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00139102619> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %20 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc3.weight_quantized> : tensor<16x32xsi8>} : () -> !torch.vtensor<[16,32],si8>
    %21 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc1.bias_quantized> : tensor<32xsi32>} : () -> !torch.vtensor<[32],si32>
    %22 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc1.bias_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32>
    %23 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32>
    %24 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc2.bias_quantized> : tensor<32xsi32>} : () -> !torch.vtensor<[32],si32>
    %25 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc2.bias_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32>
    %26 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32>
    %27 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc3.bias_quantized> : tensor<16xsi32>} : () -> !torch.vtensor<[16],si32>
    %28 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_fc3.bias_quantized_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32>
    %29 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si32>} : () -> !torch.vtensor<[],si32>
    %none = torch.constant.none
    %30 = torch.operator "onnx.DequantizeLinear"(%21, %22, %23) : (!torch.vtensor<[32],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[32],f32>
    %31 = torch.operator "onnx.DequantizeLinear"(%6, %5, %4) : (!torch.vtensor<[32,16],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[32,16],f32>
    %32 = torch.operator "onnx.DequantizeLinear"(%24, %25, %26) : (!torch.vtensor<[32],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[32],f32>
    %33 = torch.operator "onnx.DequantizeLinear"(%13, %12, %11) : (!torch.vtensor<[32,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[32,32],f32>
    %34 = torch.operator "onnx.DequantizeLinear"(%27, %28, %29) : (!torch.vtensor<[16],si32>, !torch.vtensor<[1],f32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[16],f32>
    %35 = torch.operator "onnx.DequantizeLinear"(%20, %19, %18) : (!torch.vtensor<[16,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],f32>
    %36 = torch.operator "onnx.QuantizeLinear"(%arg0, %1, %0) : (!torch.vtensor<[16,16],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,16],si8>
    %37 = torch.operator "onnx.DequantizeLinear"(%36, %1, %0) : (!torch.vtensor<[16,16],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,16],f32>
    %38 = torch.operator "onnx.Gemm"(%37, %31, %30) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[16,16],f32>, !torch.vtensor<[32,16],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[16,32],f32>
    %39 = torch.operator "onnx.QuantizeLinear"(%38, %3, %2) : (!torch.vtensor<[16,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],si8>
    %40 = torch.operator "onnx.DequantizeLinear"(%39, %3, %2) : (!torch.vtensor<[16,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],f32>
    %41 = torch.operator "onnx.Relu"(%40) : (!torch.vtensor<[16,32],f32>) -> !torch.vtensor<[16,32],f32>
    %42 = torch.operator "onnx.QuantizeLinear"(%41, %8, %7) : (!torch.vtensor<[16,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],si8>
    %43 = torch.operator "onnx.DequantizeLinear"(%42, %8, %7) : (!torch.vtensor<[16,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],f32>
    %44 = torch.operator "onnx.Gemm"(%43, %33, %32) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[16,32],f32>, !torch.vtensor<[32,32],f32>, !torch.vtensor<[32],f32>) -> !torch.vtensor<[16,32],f32>
    %45 = torch.operator "onnx.QuantizeLinear"(%44, %10, %9) : (!torch.vtensor<[16,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],si8>
    %46 = torch.operator "onnx.DequantizeLinear"(%45, %10, %9) : (!torch.vtensor<[16,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],f32>
    %47 = torch.operator "onnx.Relu"(%46) : (!torch.vtensor<[16,32],f32>) -> !torch.vtensor<[16,32],f32>
    %48 = torch.operator "onnx.QuantizeLinear"(%47, %15, %14) : (!torch.vtensor<[16,32],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],si8>
    %49 = torch.operator "onnx.DequantizeLinear"(%48, %15, %14) : (!torch.vtensor<[16,32],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,32],f32>
    %50 = torch.operator "onnx.Gemm"(%49, %35, %34) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 1 : si64} : (!torch.vtensor<[16,32],f32>, !torch.vtensor<[16,32],f32>, !torch.vtensor<[16],f32>) -> !torch.vtensor<[16,16],f32>
    %51 = torch.operator "onnx.QuantizeLinear"(%50, %17, %16) : (!torch.vtensor<[16,16],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,16],si8>
    %52 = torch.operator "onnx.DequantizeLinear"(%51, %17, %16) : (!torch.vtensor<[16,16],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[16,16],f32>
    return %52 : !torch.vtensor<[16,16],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _fc1.weight_quantized: "0x08000000A3B506E87D6D48D422BC6274E702897613E26D6BE579E7303219D32083E51E83DA2D9C89E664641E3B7B441BD3147F3285D04785BE45C1863628AFE3D59179742DDAF2A63A6BF4FD67770DEB2E2BC63305F09809139A5485E359A84F251BB635EA79D4AD7B532CAEAA3BAB86928B92F1CB685CB1F861F3DFB8C1F8B1C29184C6B82303BD038A4A6935213589F78C509B221D0DDBE44733BC81592F7F6E04422E0E69E2E2262693967B3997B11B57B1E642BABC0AD106879BB75F3991B2B0FAA996D90656EBDE2A0725E90A416AFA8864ABCD59530B47A756F2DF6EF8FCD35BD7B8EA9D0DDE885E11E90375B5F36566282BD84CD24A65F7ED9B3AB0F2620BFD29AE7BEDC5070809124BA4E96D305DBEEE82DD50C77EF13AC7381FDE41B56FF779AB139DB4B2C804B3EE1DB852A32CE775E8CF716A32EE4963D482C4CBFD711679533268B10E08DF3758C5C331E9C6866991CF23C5A53F8ABBB914EABA423B2A0E5F4A2BF28B4C65D45132ECB9AAE17B1BE0AD04D6041079F6C8DCD5E1D6ABC3291B99E4A3025CACA66CB064618EA75D9EEF8D572DBFFA03F87F3BE529B9235494860F757EDCA7B2155EAC50236B9A8CA4DA7DF4655A9467754FA8B82EFB737D548F6BC594A9578B4B6D11008CDA0730E92DF41508E7FE5A747AC6038C6972ABFAD3C9854267449F3A296D950AFECAB94AE4FFA1F4A8FFAE7C3DD15B358EDB4E75",
      _fc2.weight_quantized: "0x0800000072C754C496D60BCD981F126BF18F52DF94A27028B6CFA9477EB8728C7CA45891CB3C2703A133B78A303483937D16A2C05BEC1F8C06A38615BD38D302096095A14B710E4279A0480BDB979442962BA02BAD916E37821CA0B74DCE134D03FEBD6290C4940593420A3F27DAF319F310A6EBA1D99288347B3F8B40B6A64BFFC8E5C203F6FFA14DB8BDEF5ED17C943E9813EA0C49AF8BEF923C1964CEA1BFFF7000A148B6419A16915F62A5BD28DD84C0E8E61333CF4A7F57DDDF38D42D3074E8854CD70C88CE8F931F76EC1F46E65E6456C2702C59B45CE5D38BA156292992AF24C5711C1D0A70C220AE44612069E57AB5FD1D296BF052FC5C8BA70147ECA3BB00CA46A7CF10D2E96070900045E62C83E209A0DBCFBED1052B17B06F68C3AAC034B29BA77303C11CB7FE8B32D40D790310D4A034CB52B89DFB4E1865E813BC1B4FBFC750141D695A3375175F91ACD6CFCB3F114E6F2963249514CC2CA0AD477A6FF1B97757B249C2E7AC26F5A5EE8D30D7E83FA73A70D25F608CBEEAC52938A4BD81E6741F41E720233C62874354913E65ACB1B569F8102822B88A62558FFE7378D0A9C100CE572DDA4EA070E0DE1C59C07F0FAA5E7BFF73E2746158EB093FE5812768DD4E650FECD971B5B51E1FBF2AD57CB0F0363A397B4234EA996F67AB38040FD776597997C33BBB554068184A24EDF3CD0F663C312829942EBFF3B52FAA1F69BE6AA7B481774929AAA49344B2C89D6F9F0EAA096FCF9C5274F47845A883E3EA871FD7B18CB466F57CDE2AAD108E277F8D2EE41F593FCC5D8F434770C57E47C472969F547A67D2D015A29A5E7C780EC85503BD510AF47800713AC9B1DE0836B5955BA58E60DDF3FFBD1F614F8ADAA336CB0A89B5F11D0E9C3153C02E1E206E99EAE75E15387EE73F3F1F72C8F9C667AC4D0ABE709C8BB61C30871FF4E47F942B9379DE1B27A5E2046DCAB7B5DFF9AD98874DC3CC7F8DBE78B66A3C613AB19C51B5AC14A74C680D1D9442ED91392EC50276574E2AF2D6571A23519457B019907695B087585CA8011FFAC9A50A84CFB34CA2EC0AF66832F03B5824AB355205D0E28600585ED1D995DE6F992AC46266B2A49235ECB8B158D2AB52165153A9A807C4EBD5372C35C7212B537B695661FBF01300560D9E3695875EEE040B25D6E1D25FACB465C290709E232B59784539AC602F5C90379EA8E6A026BC605805FC98E49524A84A390412F378F7640C793C0C6FCA95F7FC4D605E18518EBB57CFA81E58B7E83F522FCFA1746832CF7341A1F2E1F7AC14B1AD25A59DC429C3D28806233C5900461FC24B0BE4CD5155A42D1FB71F06D6B5DDFA61615162127EDAC2212382E3F2843E746E74831D37218839269C78E9026FBDF101DB50685AD83B1839DF0170DD153ABC4D46FF7E003A4315850C471BED60CB6E2A9976B32F42DB6E2A024E38",
      _fc3.weight_quantized: "0x08000000E36E5D927763D8CC458735EEA75370AD18C09A8ACBCC28281669E85EFDEFA7D3AF68B389F90DE0D08D6DB8E2778AE95F5C9D11B71BD6D35878328DEE3070CF48FE43246C5C1F390562D8D5AB88DDEAAB319722435D7843290B91FBEDF714F4A16BDC8762C78C6B325C0E5408E9F65E0D23B8B56A5ABEC7FC9DE4B9E11DCD09628E2E37F87ECD9B25039ECFF3DE073D935D778EA7D3DAB445D6A3A31C2C0F778A05822258D159FFD0F6825B29F9E3BF78D12188D53E3475714112CEA30D14929FDB3B64A542229F21D077D62FEADB3A0DDDCAF78BB4AB4C7AB7324EB923C315AABB8359BA72FEEB0229B66E1FEFF95F1BE619F28466F798085136F2469475F7D00C817CCCBA1105B378237E6CA6C49B02672CB3CB00C2C9C2E0C9743C15AAD276153D4D60E1E692B970F66E6BFA0B343F08C8F37618EC9162A4A75275A24133169FB80168434DF64BBFAC8C9FB7419AC8B26CAFE1AEEDD476923114D49988D5EA6311CDDE9930DB5AC65BAE0FFAD02740E7187BA52C04F2BCB8409A8D4B2F13D271F1E372B4B852D702AEC3A416A3993DD51CA28DFE1ED3FA0F746E4F7B0B16A08F7837A4F996575B6D325D6A198691C3EF8ADAE7DB5CD01CC2DD2333C6E8DE8E7F75B1DDD292AD676044C1F445A1FF4EBF5AE10E03946C2C66E061D30B59D910FA6CEE238998737607C5E86A03ECA72E82C9BAD8A730BC96B89A688A384BE732",
      _fc1.bias_quantized: "0x0800000098F2FFFFC10C00003DFBFFFFE5F2FFFF3F020000F80B000063FBFFFF71F5FFFF8002000071090000D80500005306000089020000E9FCFFFFD10A00001EFBFFFF60F2FFFF61F8FFFFEDF8FFFFDD060000E2F4FFFF1AF7FFFF16FFFFFFB80700004C0E00002B0600000E040000E2F5FFFF4E0D00006D080000D3F9FFFF39090000",
      _fc1.bias_quantized_scale: "0x080000009A228638",
      _fc2.bias_quantized: "0x0800000017FDFFFFFBF6FFFFC3F5FFFFE0EFFFFF7209000034EDFFFFFFF2FFFF51110000CA01000091070000ABEDFFFF4208000087FAFFFFC0FDFFFFA603000064FEFFFF0E0000004902000026FBFFFF6FFBFFFF47030000C303000043060000B6F9FFFFA50C000068000000A8020000FCFBFFFF560600009CF3FFFF3F100000B0120000",
      _fc2.bias_quantized_scale: "0x08000000809E1538",
      _fc3.bias_quantized: "0x0800000091300000D3E6FFFF4AECFFFF30DEFFFF93200000080F0000C7DCFFFF10D1FFFF921900006B140000B9EDFFFFADEFFFFFB2040000A3F0FFFFDED7FFFF43070000",
      _fc3.bias_quantized_scale: "0x0800000074736A37"
    }
  }
#-}
