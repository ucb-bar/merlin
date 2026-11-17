"""Content-sized and resizing source shared by host and target transport tests."""

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


RESIZE = """module {
func.func @forward(%x:tensor<?xi32> {bufferization.writable=false}, %steps:i32)
 -> tensor<?xi32> {
 %z=arith.constant 0:index
 %one=arith.constant 1:index
 %limit=arith.index_cast %steps:i32 to index
 %zero=arith.constant 0:i32
 %valid=arith.cmpi sge,%steps,%zero:i32
 cf.assert %valid,"source guard"
 %result=scf.for %i=%z to %limit step %one
   iter_args(%state=%x)->(tensor<?xi32>) {
   %size=tensor.dim %state,%z:tensor<?xi32>
   %empty=arith.cmpi eq,%size,%z:index
   %decreased=arith.subi %size,%one:index
   %nextsize=arith.select %empty,%z,%decreased:index
   %fresh=tensor.empty(%nextsize):tensor<?xi32>
   %next=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
      outs(%fresh:tensor<?xi32>) {
     ^bb0(%unused:i32):
       %index=linalg.index 0:index
       %value=tensor.extract %state[%index]:tensor<?xi32>
       linalg.yield %value:i32
   } -> tensor<?xi32>
   scf.yield %next:tensor<?xi32>
 }
 return %result:tensor<?xi32>
}}"""

MULTI_RESIZE = """module {
func.func @forward(%x:tensor<?xi32> {bufferization.writable=false},
 %y:tensor<?xf64> {bufferization.writable=false},%steps:i32)
 -> (tensor<?xi32>,tensor<?xf64>) {
 %z=arith.constant 0:index
 %one=arith.constant 1:index
 %two=arith.constant 2:index
 %limit=arith.index_cast %steps:i32 to index
 %zero=arith.constant 0:i32
 %valid=arith.cmpi sge,%steps,%zero:i32
 cf.assert %valid,"source guard"
 %result:2=scf.for %i=%z to %limit step %one
   iter_args(%a=%x,%b=%y)->(tensor<?xi32>,tensor<?xf64>) {
   %na=tensor.dim %a,%z:tensor<?xi32>
   %nb=tensor.dim %b,%z:tensor<?xf64>
   %ea=arith.cmpi eq,%na,%z:index
   %eb=arith.cmpi slt,%nb,%two:index
   %da=arith.subi %na,%one:index
   %db=arith.subi %nb,%two:index
   %sa=arith.select %ea,%z,%da:index
   %sb=arith.select %eb,%z,%db:index
   %afresh=tensor.empty(%sa):tensor<?xi32>
   %bfresh=tensor.empty(%sb):tensor<?xf64>
   %anext=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
      outs(%afresh:tensor<?xi32>) {
     ^bb0(%unused:i32):
       %index=linalg.index 0:index
       %value=tensor.extract %a[%index]:tensor<?xi32>
       linalg.yield %value:i32
   } -> tensor<?xi32>
   %bnext=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
      outs(%bfresh:tensor<?xf64>) {
     ^bb0(%unused:f64):
       %index=linalg.index 0:index
       %value=tensor.extract %b[%index]:tensor<?xf64>
       linalg.yield %value:f64
   } -> tensor<?xf64>
   scf.yield %anext,%bnext:tensor<?xi32>,tensor<?xf64>
 }
 return %result#0,%result#1:tensor<?xi32>,tensor<?xf64>
}}"""
