; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @radiance_kernel(ptr %0, ptr %1, ptr %2) {
  br label %4

4:                                                ; preds = %33, %3
  %5 = phi i32 [ %35, %33 ], [ 0, %3 ]
  %6 = icmp ult i32 %5, 32
  br i1 %6, label %7, label %12

7:                                                ; preds = %4
  %8 = udiv i32 %5, 16
  %9 = urem i32 %5, 16
  %10 = udiv i32 %9, 1
  %11 = urem i32 %9, 1
  br label %13

12:                                               ; preds = %4
  ret void

13:                                               ; preds = %17, %7
  %14 = phi i32 [ %32, %17 ], [ 0, %7 ]
  %15 = phi float [ %31, %17 ], [ 0.000000e+00, %7 ]
  %16 = icmp ult i32 %14, 16
  br i1 %16, label %17, label %33

17:                                               ; preds = %13
  %18 = mul i32 %8, 256
  %19 = mul i32 %10, 16
  %20 = add i32 %19, %14
  %21 = add i32 %18, %20
  %22 = mul i32 %8, 16
  %23 = mul i32 %14, 1
  %24 = add i32 %23, %11
  %25 = add i32 %22, %24
  %26 = getelementptr float, ptr %0, i32 %21
  %27 = load float, ptr %26, align 4
  %28 = getelementptr float, ptr %1, i32 %25
  %29 = load float, ptr %28, align 4
  %30 = fmul float %27, %29
  %31 = fadd float %15, %30
  %32 = add i32 %14, 1
  br label %13

33:                                               ; preds = %13
  %34 = getelementptr float, ptr %2, i32 %5
  store float %15, ptr %34, align 4
  %35 = add i32 %5, 1
  br label %4
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
