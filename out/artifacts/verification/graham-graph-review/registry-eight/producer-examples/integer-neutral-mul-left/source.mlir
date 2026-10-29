builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {
%c = arith.constant 1 : i8
%r = arith.muli %c, %a : i8
func.return %r : i8
} }
