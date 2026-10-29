builtin.module {
func.func private @step(i32, index) -> i32
func.func private @other(i32, index) -> i32
func.func private @observe(i32, i32)
func.func private @watch(i32)
func.func @main(%x:i32, %y:i32) -> (i32, i32, i32, i32) {
%lb = arith.constant 1 : index
%ub = arith.constant 11 : index
%st = arith.constant 1 : index
%r:4 = scf.for %i = %lb to %ub step %st iter_args(%c0 = %x, %c1 = %y, %c2 = %x, %c3 = %y) -> (i32, i32, i32, i32) {
%a0 = arith.addi %c0, %y : i32
%b0 = func.call @step(%a0, %i) : (i32,index) -> i32
%a1 = arith.addi %c1, %y : i32
%b1 = func.call @step(%a1, %i) : (i32,index) -> i32
%a2 = arith.addi %c2, %y : i32
%b2 = func.call @step(%a2, %i) : (i32,index) -> i32
%a3 = arith.addi %c3, %y : i32
%b3 = func.call @step(%a3, %i) : (i32,index) -> i32
func.call @observe(%c0, %y) : (i32,i32) -> ()
func.call @observe(%y, %c0) : (i32,i32) -> ()
scf.yield %b0, %b1, %b2, %b3 : i32, i32, i32, i32
}
func.call @watch(%r#0) : (i32) -> ()
func.return %r#0, %r#1, %r#2, %r#3 : i32, i32, i32, i32
}
func.func @untouched(%x:i32) -> i32 { func.return %x : i32 }
}