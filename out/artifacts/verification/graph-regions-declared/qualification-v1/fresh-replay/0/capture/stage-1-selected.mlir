builtin.module {func.func @forward(%p:i8,%w:i8)->i8 {
%r=arith.addi %p,%w:i8
func.return %r:i8
}}