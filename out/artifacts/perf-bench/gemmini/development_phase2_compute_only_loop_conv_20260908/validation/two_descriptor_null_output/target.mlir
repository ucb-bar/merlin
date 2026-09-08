"builtin.module"() ({
  "llvm.mlir.global"() <{global_type = !llvm.array<64 x i8>, sym_name = "__gemmini_stage_0", linkage = #llvm.linkage<"internal">, addr_space = 0 : i32, alignment = 64 : i64}> ({
  }) : () -> ()
  "llvm.func"() <{sym_name = "gemmini_kernel", function_type = !llvm.func<void (!llvm.ptr, !llvm.ptr, !llvm.ptr)>, CConv = #llvm.cconv<ccc>, linkage = #llvm.linkage<"external">, visibility_ = 0 : i64}> ({
  ^bb0(%0: !llvm.ptr, %1: !llvm.ptr, %2: !llvm.ptr):
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> {merlin.global_task = -1 : i64} : () -> ()
    %3 = "llvm.mlir.constant"() <{value = 0 : i64}> {merlin.global_task = -1 : i64} : () -> i64
    "llvm.inline_asm"(%3, %3) <{asm_string = ".insn r 0x7b, 0x3, 0x7, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> {merlin.global_task = -1 : i64} : (i64, i64) -> ()
    %4 = "llvm.mlir.addressof"() <{global_name = @__gemmini_stage_0}> : () -> !llvm.ptr
    %5 = "llvm.mlir.constant"() <{value = 16 : i64}> : () -> i64
    %6 = "llvm.mlir.constant"() <{value = 1 : i64}> : () -> i64
    %7 = "llvm.mlir.constant"() <{value = 2 : i64}> : () -> i64
    %8 = "llvm.mlir.constant"() <{value = 4575657221408423968 : i64}> : () -> i64
    %9 = "llvm.mlir.constant"() <{value = 4575657221408489732 : i64}> : () -> i64
    %10 = "llvm.mlir.constant"() <{value = 281474976710656 : i64}> : () -> i64
    %11 = "llvm.mlir.constant"() <{value = 4575657221409472785 : i64}> : () -> i64
    %12 = "llvm.ptrtoint"(%4) : (!llvm.ptr) -> i64
    %13 = "llvm.mlir.constant"() <{value = 1125970773803008 : i64}> : () -> i64
    %14 = "llvm.mlir.constant"() <{value = 1125970773803012 : i64}> : () -> i64
    %15 = "llvm.mlir.constant"() <{value = 1125970773803016 : i64}> : () -> i64
    %16 = "llvm.mlir.constant"() <{value = 1125970773803020 : i64}> : () -> i64
    %17 = "llvm.mlir.constant"() <{value = 9007212139905025 : i64}> : () -> i64
    %18 = "llvm.mlir.constant"() <{value = 281492156841988 : i64}> : () -> i64
    %19 = "llvm.mlir.constant"() <{value = 281492156645632 : i64}> : () -> i64
    %20 = "llvm.mlir.constant"() <{value = 281492156842000 : i64}> : () -> i64
    %21 = "llvm.mlir.constant"() <{value = 281479271874560 : i64}> : () -> i64
    %22 = "llvm.mlir.constant"() <{value = 4 : i64}> : () -> i64
    %23 = "llvm.mlir.constant"() <{value = 1125899906842625 : i64}> : () -> i64
    %24 = "llvm.mlir.constant"() <{value = 1126037347893252 : i64}> : () -> i64
    %25 = "llvm.ptrtoint"(%1) : (!llvm.ptr) -> i64
    %26 = "llvm.ptrtoint"(%0) : (!llvm.ptr) -> i64
    %27 = "llvm.mlir.constant"() <{value = 289 : i64}> : () -> i64
    %28 = "llvm.mlir.constant"() <{value = 4575657221408424064 : i64}> : () -> i64
    %29 = "llvm.ptrtoint"(%2) : (!llvm.ptr) -> i64
    %30 = "llvm.mlir.constant"() <{value = 1125971310673920 : i64}> : () -> i64
    %31 = "llvm.mlir.constant"() <{value = 512 : i64}> : () -> i64
    %32 = "llvm.add"(%29, %31) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %33 = "llvm.mlir.constant"() <{value = 1125971310673924 : i64}> : () -> i64
    %34 = "llvm.mlir.constant"() <{value = 1024 : i64}> : () -> i64
    %35 = "llvm.add"(%29, %34) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %36 = "llvm.mlir.constant"() <{value = 1125971310673928 : i64}> : () -> i64
    %37 = "llvm.mlir.constant"() <{value = 1536 : i64}> : () -> i64
    %38 = "llvm.add"(%29, %37) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %39 = "llvm.mlir.constant"() <{value = 1125971310673932 : i64}> : () -> i64
    %40 = "llvm.add"(%25, %5) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %41 = "llvm.mlir.constant"() <{value = 64 : i64}> : () -> i64
    %42 = "llvm.add"(%29, %41) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %43 = "llvm.mlir.constant"() <{value = 576 : i64}> : () -> i64
    %44 = "llvm.add"(%29, %43) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %45 = "llvm.mlir.constant"() <{value = 1088 : i64}> : () -> i64
    %46 = "llvm.add"(%29, %45) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    %47 = "llvm.mlir.constant"() <{value = 1600 : i64}> : () -> i64
    %48 = "llvm.add"(%29, %47) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%3) [^bb1] : (i64) -> ()
  ^bb1(%49: i64):
    %50 = "llvm.icmp"(%49, %5) <{predicate = 2 : i64}> : (i64, i64) -> i1
    "llvm.cond_br"(%50) [^bb2, ^bb3] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb2:
    %51 = "llvm.trunc"(%3) <{overflowFlags = #llvm.overflow<none>}> : (i64) -> i32
    %52 = "llvm.getelementptr"(%4, %49) <{rawConstantIndices = array<i32: -2147483648>, elem_type = i32, noWrapFlags = 0 : i32}> : (!llvm.ptr, i64) -> !llvm.ptr
    "llvm.store"(%51, %52) <{ordering = 0 : i64}> : (i32, !llvm.ptr) -> ()
    %53 = "llvm.add"(%49, %6) <{overflowFlags = 0 : i32}> : (i64, i64) -> i64
    "llvm.br"(%53) [^bb1] : (i64) -> ()
  ^bb3:
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : () -> ()
    "llvm.inline_asm"(%7, %8) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%9, %10) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%11, %3) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %13) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %14) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %15) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %16) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%17, %18) <{asm_string = ".insn r 0x7b, 0x3, 0x10, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%19, %20) <{asm_string = ".insn r 0x7b, 0x3, 0x11, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%21, %22) <{asm_string = ".insn r 0x7b, 0x3, 0x12, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%23, %24) <{asm_string = ".insn r 0x7b, 0x3, 0x13, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%25, %3) <{asm_string = ".insn r 0x7b, 0x3, 0x14, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%3, %26) <{asm_string = ".insn r 0x7b, 0x3, 0x15, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%27, %6) <{asm_string = ".insn r 0x7b, 0x3, 0xf, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%7, %28) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%29, %30) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%32, %33) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%35, %36) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%38, %39) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%11, %3) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %13) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %14) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %15) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%12, %16) <{asm_string = ".insn r 0x7b, 0x3, 0xe, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%17, %18) <{asm_string = ".insn r 0x7b, 0x3, 0x10, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%19, %20) <{asm_string = ".insn r 0x7b, 0x3, 0x11, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%21, %22) <{asm_string = ".insn r 0x7b, 0x3, 0x12, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%23, %24) <{asm_string = ".insn r 0x7b, 0x3, 0x13, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%40, %3) <{asm_string = ".insn r 0x7b, 0x3, 0x14, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%3, %26) <{asm_string = ".insn r 0x7b, 0x3, 0x15, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%27, %6) <{asm_string = ".insn r 0x7b, 0x3, 0xf, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%42, %30) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%44, %33) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%46, %36) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"(%48, %39) <{asm_string = ".insn r 0x7b, 0x3, 0x3, x0, $0, $1", constraints = "r,r", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> : (i64, i64) -> ()
    "llvm.inline_asm"() <{asm_string = "fence", constraints = "~{memory}", has_side_effects, tail_call_kind = #llvm.tailcallkind<none>}> {merlin.global_task = -2 : i64} : () -> ()
    "llvm.return"() : () -> ()
  }) : () -> ()
}) {merlin.host_workspace_bytes = 0 : i64, merlin.host_stack_upper_bound_bytes = 0 : i64} : () -> ()
