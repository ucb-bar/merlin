builtin.module {func.func @forward(%x:i16,%w:i16,%s:i16)->i16 {
%p=arith.muli %x,%w:i16
%r=arith.addi %p,%s:i16
func.return %r:i16
}}