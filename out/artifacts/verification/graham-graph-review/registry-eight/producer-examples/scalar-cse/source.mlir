builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {
%first = arith.addi %a, %b : i8
%duplicate = arith.addi %a, %b : i8
%r = arith.muli %first, %duplicate : i8
func.return %r : i8
} }
