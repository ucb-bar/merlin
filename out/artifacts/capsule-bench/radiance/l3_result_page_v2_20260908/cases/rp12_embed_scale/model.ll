; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @radiance_kernel(ptr %0, ptr %1) {
  br label %3

3:                                                ; preds = %6, %2
  %4 = phi i32 [ %11, %6 ], [ 0, %2 ]
  %5 = icmp ult i32 %4, 256
  br i1 %5, label %6, label %12

6:                                                ; preds = %3
  %7 = getelementptr float, ptr %0, i32 %4
  %8 = load float, ptr %7, align 4
  %9 = fmul float %8, 4.000000e+00
  %10 = getelementptr float, ptr %1, i32 %4
  store float %9, ptr %10, align 4
  %11 = add i32 %4, 1
  br label %3

12:                                               ; preds = %3
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
