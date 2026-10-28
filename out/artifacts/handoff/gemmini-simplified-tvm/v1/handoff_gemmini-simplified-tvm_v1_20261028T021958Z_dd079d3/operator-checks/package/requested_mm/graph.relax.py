# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_0_operator_fused_matmul_cast(x: T.Buffer((T.int64(2), T.int64(3)), "int8"), y: T.Buffer((T.int64(3), T.int64(2)), "int8"), compute_intermediate: T.Buffer((T.int64(2), T.int64(2)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.k": 3, "gemmini.m": 2, "gemmini.n": 2, "gemmini.tensorir": 1, "gemmini.workload": "42b0c6a09a909ac8", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.sketch": "ij.2x2x3.os.h0.p0.eroot.stile.tnone"})
            matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(2)), "int32")
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", y.data, T.int64(6), y.data, T.int64(6), compute_intermediate.data, T.int64(4))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(6), x.data, T.int64(6), compute_intermediate.data, T.int64(4))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    for i0_0_0, i1_0_0 in T.grid(T.int64(1), T.int64(1)):
                        with T.block(""):
                            T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)], y[T.int64(0):T.int64(3), T.int64(0):T.int64(2)])
                            T.writes(matmul_intermediate[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                            for i0_0_1_init, i1_0_1_init in T.grid(T.int64(1), T.int64(1)):
                                with T.block("matmul_init_o"):
                                    T.reads()
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_zero_acc", T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(2))
                            for k_0_0 in range(T.int64(1)):
                                with T.block(""):
                                    T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)], y[T.int64(0):T.int64(3), T.int64(0):T.int64(2)])
                                    T.writes()
                                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                        with T.block("x_local_gemmini_spad_o"):
                                            T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                            T.writes()
                                            T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x.data, T.int64(0), T.int64(6), 1), T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(3), T.int64(3), 0)
                                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                        with T.block("y_local_gemmini_spad_o"):
                                            T.reads(y[T.int64(0):T.int64(3), T.int64(0):T.int64(2)])
                                            T.writes()
                                            T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), y.data, T.int64(0), T.int64(6), 1), T.Cast("uint32", T.int64(16)), T.uint32(3), T.uint32(2), T.int64(2), 0)
                                    for k_0_1, i0_0_1, i1_0_1 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                        with T.block("matmul_update_o"):
                                            T.reads()
                                            T.writes()
                                            T.call_extern("void", "tvm_gemmini_compute", T.Cast("uint32", T.int64(0)), T.Cast("uint32", T.int64(16)), T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(2), T.uint32(3), 1)
                            for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("matmul_intermediate_local_gemmini_acc_o"):
                                    T.reads()
                                    T.writes(matmul_intermediate[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), matmul_intermediate.data, T.int64(0), T.int64(4), 2), T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(2), T.int64(2), 0)
                    T.call_extern("void", "tvm_gemmini_end")
                    for i0, i1 in T.grid(T.int64(2), T.int64(2)):
                        with T.block("compute"):
                            T.reads(matmul_intermediate[i0, i1])
                            T.writes(compute_intermediate[i0, i1])
                            compute_intermediate[i0, i1] = T.Cast("int8", matmul_intermediate[i0, i1])
                    T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int8"), y: R.Tensor((3, 2), dtype="int8"), w: R.Tensor((2,), dtype="int8")) -> R.Tensor((2, 2), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_0_operator_fused_matmul_cast, (x, y), out_sinfo=R.Tensor((2, 2), dtype="int8"))
            R.output(gv)
        return gv