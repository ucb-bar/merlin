# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_3_operator_fused_conv2d_cast_add_cast1(x: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "int8"), y: T.Buffer((T.int64(2), T.int64(1), T.int64(2), T.int64(2)), "int8"), lv1: T.Buffer((T.int64(1), T.int64(2), T.int64(1), T.int64(1)), "int8"), compute_intermediate_1: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.k": 4, "gemmini.m": 6, "gemmini.n": 2, "gemmini.tensorir": 1, "gemmini.workload": "a113c76f697b7a0c", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.sketch": "ij.6x2x4.os.h0.p0.eroot.stile.tnone"})
            conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)), "int32")
            compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(1), T.int64(1)), "int32")
            T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)), "int32")
            conv2d_nchw_intermediate_reindex = T.alloc_buffer((T.int64(6), T.int64(2)), "int32")
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", lv1.data, T.int64(2), lv1.data, T.int64(2), compute_intermediate_1.data, T.int64(12))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", y.data, T.int64(8), y.data, T.int64(8), compute_intermediate_1.data, T.int64(12))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    gemmini_admission_2: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(12), x.data, T.int64(12), compute_intermediate_1.data, T.int64(12))
                    if gemmini_admission_2 != 0:
                        T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                        T.tvm_throw_last_error()
                    else:
                        T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                        for ax0_0_0, ax1_0_0 in T.grid(T.int64(1), T.int64(1)):
                            with T.block(""):
                                T.reads(x[T.int64(0), T.int64(0), T.int64(0):T.int64(3), T.int64(0):T.int64(4)], y[T.int64(0):T.int64(2), T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                T.writes(conv2d_nchw_intermediate_reindex[T.int64(0):T.int64(6), T.int64(0):T.int64(2)])
                                for ax0_0_1_init, ax1_0_1_init in T.grid(T.int64(1), T.int64(1)):
                                    with T.block("conv2d_nchw_init_o"):
                                        T.reads()
                                        T.writes()
                                        T.call_extern("void", "tvm_gemmini_zero_acc", T.Cast("uint32", T.int64(0)), T.uint32(6), T.uint32(2))
                                for ax2_0_0 in range(T.int64(1)):
                                    with T.block(""):
                                        T.reads(x[T.int64(0), T.int64(0), T.int64(0):T.int64(3), T.int64(0):T.int64(4)], y[T.int64(0):T.int64(2), T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                        T.writes()
                                        pad_temp_reindex = T.alloc_buffer((T.int64(6), T.int64(4)), "int8")
                                        y_reindex = T.alloc_buffer((T.int64(4), T.int64(2)), "int8")
                                        for ax0, ax1 in T.grid(T.int64(6), T.int64(4)):
                                            with T.block("pad_temp_reindex_reindex"):
                                                T.reads(x[T.int64(0), T.int64(0), ax0 // T.int64(3) + ax1 // T.int64(2), ax0 % T.int64(3) + ax1 % T.int64(2)])
                                                T.writes(pad_temp_reindex[ax0, ax1])
                                                pad_temp_reindex[ax0, ax1] = x[T.int64(0), T.int64(0), ax0 // T.int64(3) + ax1 // T.int64(2), ax0 % T.int64(3) + ax1 % T.int64(2)]
                                        for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                            with T.block("pad_temp_reindex_local_gemmini_spad_o"):
                                                T.reads(pad_temp_reindex[T.int64(0):T.int64(6), T.int64(0):T.int64(4)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), pad_temp_reindex.data, T.int64(0), T.int64(24), 1), T.Cast("uint32", T.int64(0)), T.uint32(6), T.uint32(4), T.int64(4), 1)
                                        for ax0, ax1 in T.grid(T.int64(2), T.int64(4)):
                                            with T.block("y_reindex_reindex"):
                                                T.reads(y[ax0, T.int64(0), ax1 // T.int64(2), ax1 % T.int64(2)])
                                                T.writes(y_reindex[ax1, ax0])
                                                y_reindex[ax1, ax0] = y[ax0, T.int64(0), ax1 // T.int64(2), ax1 % T.int64(2)]
                                        for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                            with T.block("y_reindex_local_gemmini_spad_o"):
                                                T.reads(y_reindex[T.int64(0):T.int64(4), T.int64(0):T.int64(2)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), y_reindex.data, T.int64(0), T.int64(8), 1), T.Cast("uint32", T.int64(16)), T.uint32(4), T.uint32(2), T.int64(2), 1)
                                        for ax2_0_1, ax0_0_1, ax1_0_1 in T.grid(T.int64(1), T.int64(1), T.int64(1)):
                                            with T.block("conv2d_nchw_update_o"):
                                                T.reads()
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_compute", T.Cast("uint32", T.int64(0)), T.Cast("uint32", T.int64(16)), T.Cast("uint32", T.int64(0)), T.uint32(6), T.uint32(2), T.uint32(4), 1)
                                for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                                    with T.block("conv2d_nchw_intermediate_reindex_local_gemmini_acc_o"):
                                        T.reads()
                                        T.writes(conv2d_nchw_intermediate_reindex[T.int64(0):T.int64(6), T.int64(0):T.int64(2)])
                                        T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), conv2d_nchw_intermediate_reindex.data, T.int64(0), T.int64(12), 2), T.Cast("uint32", T.int64(0)), T.uint32(6), T.uint32(2), T.int64(2), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        for ax0, ax1 in T.grid(T.int64(6), T.int64(2)):
                            with T.block("conv2d_nchw_intermediate_reindex"):
                                T.reads(conv2d_nchw_intermediate_reindex[ax0, ax1])
                                T.writes(conv2d_nchw_intermediate[T.int64(0), ax1, ax0 // T.int64(3), ax0 % T.int64(3)])
                                conv2d_nchw_intermediate[T.int64(0), ax1, ax0 // T.int64(3), ax0 % T.int64(3)] = conv2d_nchw_intermediate_reindex[ax0, ax1]
                        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(2), T.int64(1), T.int64(1)):
                            with T.block("compute"):
                                T.reads(lv1[T.int64(0), i1, T.int64(0), T.int64(0)])
                                T.writes(compute_intermediate[T.int64(0), i1, T.int64(0), T.int64(0)])
                                compute_intermediate[T.int64(0), i1, T.int64(0), T.int64(0)] = T.Cast("int32", lv1[T.int64(0), i1, T.int64(0), T.int64(0)])
                        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(2), T.int64(2), T.int64(3)):
                            with T.block("T_add"):
                                T.reads(conv2d_nchw_intermediate[T.int64(0), ax1, ax2, ax3], compute_intermediate[T.int64(0), ax1, T.int64(0), T.int64(0)])
                                T.writes(T_add_intermediate[T.int64(0), ax1, ax2, ax3])
                                T_add_intermediate[T.int64(0), ax1, ax2, ax3] = conv2d_nchw_intermediate[T.int64(0), ax1, ax2, ax3] + compute_intermediate[T.int64(0), ax1, T.int64(0), T.int64(0)]
                        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(2), T.int64(2), T.int64(3)):
                            with T.block("compute_1"):
                                T.reads(T_add_intermediate[T.int64(0), i1, i2, i3])
                                T.writes(compute_intermediate_1[T.int64(0), i1, i2, i3])
                                compute_intermediate_1[T.int64(0), i1, i2, i3] = T.Cast("int8", T_add_intermediate[T.int64(0), i1, i2, i3])
                        T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((1, 1, 3, 4), dtype="int8"), y: R.Tensor((2, 1, 2, 2), dtype="int8"), w: R.Tensor((2,), dtype="int8")) -> R.Tensor((1, 2, 2, 3), dtype="int8"):
        cls = Module
        with R.dataflow():
            lv1: R.Tensor((1, 2, 1, 1), dtype="int8") = R.reshape(w, R.shape([1, 2, 1, 1]))
            gv = R.call_tir(cls.stage_3_operator_fused_conv2d_cast_add_cast1, (x, y, lv1), out_sinfo=R.Tensor((1, 2, 2, 3), dtype="int8"))
            R.output(gv)
        return gv