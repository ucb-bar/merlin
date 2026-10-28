# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_42_operator_fused_strided_slice_squeeze(x: T.Buffer((T.int64(2), T.int64(3), T.int64(4)), "float32"), T_squeeze_intermediate: T.Buffer((T.int64(2), T.int64(4)), "float32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "b728e8cdfa0a250d", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            T_strided_slice_with_axes_intermediate = T.alloc_buffer((T.int64(2), T.int64(1), T.int64(4)))
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(96), x.data, T.int64(96), T_squeeze_intermediate.data, T.int64(32))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                with T.block("T_strided_slice_with_axes_o"):
                    T.reads(x[T.int64(0):T.int64(2), T.int64(2), T.int64(0):T.int64(4)])
                    T.writes(T_strided_slice_with_axes_intermediate[T.int64(0):T.int64(2), T.int64(0), T.int64(0):T.int64(4)])
                    for ax0, ax1_0, ax2_0 in T.grid(T.int64(2), T.int64(1), T.int64(4)):
                        with T.block(""):
                            T.reads(x[ax0, T.int64(2), ax2_0])
                            T.writes(T_strided_slice_with_axes_intermediate[ax0, T.int64(0), ax2_0])
                            with T.block("T_strided_slice_with_axes_o_1"):
                                T.reads(x[ax0, T.int64(2), ax2_0])
                                T.writes()
                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), x.data, ax0 * T.int64(12) + ax2_0 + T.int64(8), T.int64(4), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(4), 0)
                            for ax0_1 in range(T.int64(1)):
                                with T.block("T_strided_slice_with_axes_intermediate_local_gemmini_acc_o"):
                                    T.reads()
                                    T.writes(T_strided_slice_with_axes_intermediate[ax0, T.int64(0), ax2_0])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), T_strided_slice_with_axes_intermediate.data, ax0 * T.int64(4) + ax2_0, T.int64(4), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(4), 0)
                T.call_extern("void", "tvm_gemmini_end")
                with T.block("T_squeeze_o"):
                    T.reads(T_strided_slice_with_axes_intermediate[T.int64(0):T.int64(2), T.int64(0), T.int64(0):T.int64(4)])
                    T.writes(T_squeeze_intermediate[T.int64(0):T.int64(2), T.int64(0):T.int64(4)])
                    for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(4)):
                        with T.block(""):
                            T.reads(T_strided_slice_with_axes_intermediate[ax0_0, T.int64(0), ax1_0])
                            T.writes(T_squeeze_intermediate[ax0_0, ax1_0])
                            with T.block("T_squeeze_o_1"):
                                T.reads(T_strided_slice_with_axes_intermediate[ax0_0, T.int64(0), ax1_0])
                                T.writes()
                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), T_strided_slice_with_axes_intermediate.data, ax0_0 * T.int64(4) + ax1_0, T.int64(4), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(4), 0)
                            with T.block("T_squeeze_intermediate_local_gemmini_acc_o"):
                                T.reads()
                                T.writes(T_squeeze_intermediate[ax0_0, ax1_0])
                                T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), T_squeeze_intermediate.data, ax0_0 * T.int64(4) + ax1_0, T.int64(4), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(4), 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3, 4), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_42_operator_fused_strided_slice_squeeze, (x,), out_sinfo=R.Tensor((2, 4), dtype="float32"))
            R.output(gv)
        return gv