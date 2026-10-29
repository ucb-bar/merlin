builtin.module { func.func @kernel(%a:i8, %b:i8) -> i8 {
%x = arith.addi %a, %a : i8
%y = arith.muli %b, %b : i8
%r = arith.subi %x, %y : i8
func.return %r : i8
} func.func @untouched(%a:i8) -> i8 {func.return %a : i8}
}
