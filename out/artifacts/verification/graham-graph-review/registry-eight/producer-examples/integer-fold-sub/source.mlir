builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {
%x = arith.constant 127 : i8
%y = arith.constant 3 : i8
%r = arith.subi %x, %y : i8
func.return %r : i8
} }
