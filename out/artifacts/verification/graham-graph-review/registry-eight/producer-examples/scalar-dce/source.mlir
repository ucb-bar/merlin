builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {
%dead = arith.addi %a, %b : i8
%dead2 = arith.muli %dead, %b : i8
func.return %a : i8
} }
