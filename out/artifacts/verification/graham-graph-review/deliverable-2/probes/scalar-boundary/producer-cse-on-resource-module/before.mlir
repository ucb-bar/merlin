builtin.module {
  func.func @k(%a:i8, %b:i8) -> (i8, tensor<2xi8>) {
    %c = arith.constant dense_resource<blob1> : tensor<2xi8>
    %x = arith.addi %a, %b : i8
    %y = arith.addi %a, %b : i8
    %z = arith.muli %x, %y : i8
    func.return %z, %c : i8, tensor<2xi8>
  }
}
{-#
  dialect_resources: {
    builtin: {
      blob1: "0x010000000102"
    }
  }
#-}
