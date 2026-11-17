"""Content-derived compaction shared by host and bare-metal transport tests."""

COMPACT = """module {
func.func @forward(%x:tensor<?xi32> {bufferization.writable=false})
 -> (tensor<?xi32>,tensor<?xi32>,tensor<?xi32>) {
 %z=arith.constant 0:index
 %one=arith.constant 1:index
 %zero=arith.constant 0:i32
 %end=tensor.dim %x,%z:tensor<?xi32>
 %n=scf.for %i=%z to %end step %one iter_args(%count=%z)->(index) {
   %v=tensor.extract %x[%i]:tensor<?xi32>
   %positive=arith.cmpi sgt,%v,%zero:i32
   %inc=arith.select %positive,%one,%z:index
   %next=arith.addi %count,%inc:index
   scf.yield %next:index
 }
 %empty=tensor.empty(%n):tensor<?xi32>
 %packed:2=scf.for %i=%z to %end step %one
   iter_args(%destination=%empty,%at=%z)->(tensor<?xi32>,index) {
   %v=tensor.extract %x[%i]:tensor<?xi32>
   %positive=arith.cmpi sgt,%v,%zero:i32
   %selected:2=scf.if %positive -> (tensor<?xi32>,index) {
     %updated=tensor.insert %v into %destination[%at]:tensor<?xi32>
     %next=arith.addi %at,%one:index
     scf.yield %updated,%next:tensor<?xi32>,index
   } else {
     scf.yield %destination,%at:tensor<?xi32>,index
   }
   scf.yield %selected#0,%selected#1:tensor<?xi32>,index
 }
 return %packed#0,%packed#0,%x:tensor<?xi32>,tensor<?xi32>,tensor<?xi32>
}}"""
