builtin.module {func.func @forward(%p:i16,%w:i16)->i16 {
%r=arith.addi %p,%w:i16
func.return %r:i16
}}