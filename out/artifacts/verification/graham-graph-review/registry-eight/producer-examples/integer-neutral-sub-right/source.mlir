builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {
%c = arith.constant 0 : i8
%r = arith.subi %a, %c : i8
func.return %r : i8
} }
