builtin.module {func.func @forward(%x:i8,%w:i8,%s:i8)->i8 {
%p=arith.muli %x,%w:i8
%r=arith.addi %p,%s:i8
func.return %r:i8
}}