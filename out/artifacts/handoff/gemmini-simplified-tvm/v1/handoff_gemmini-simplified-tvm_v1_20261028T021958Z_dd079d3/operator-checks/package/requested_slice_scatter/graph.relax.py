# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_26_operator_scatter_slice(x: T.Buffer((T.int64(2), T.int64(5)), "int32"), y: T.Buffer((T.int64(2), T.int64(2)), "int32"), strided_set: T.Buffer((T.int64(2), T.int64(5)), "int32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "77a431b861e53154", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", y.data, T.int64(16), y.data, T.int64(16), strided_set.data, T.int64(40))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(40), x.data, T.int64(40), strided_set.data, T.int64(40))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    with T.block("strided_set_i1_common"):
                        T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(0):T.int64(5)])
                        T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0):T.int64(5)])
                        with T.block("strided_set_i10_partition"):
                            T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(0)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0)])
                            with T.block("strided_set_global_o"):
                                T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(0)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0)])
                                with T.block("strided_set_global_o_5"):
                                    T.reads(x[T.int64(0):T.int64(2), T.int64(0)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0)])
                                    for ax0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block(""):
                                            T.reads(x[ax0, T.int64(0)])
                                            T.writes(strided_set[ax0, T.int64(0)])
                                            with T.block("strided_set_global_o_6"):
                                                T.reads(x[ax0, T.int64(0)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), x.data, ax0 * T.int64(5), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("strided_set_local_gemmini_acc_o"):
                                                    T.reads()
                                                    T.writes(strided_set[ax0, T.int64(0)])
                                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), strided_set.data, ax0 * T.int64(5), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        with T.block("strided_set_i11_partition"):
                            T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(1)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(1)])
                            with T.block("strided_set_global_o_1"):
                                T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(1)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(1)])
                                with T.block("strided_set_global_o_7"):
                                    T.reads(y[T.int64(0):T.int64(2), T.int64(0)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(1)])
                                    for ax0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block(""):
                                            T.reads(y[ax0, T.int64(0)])
                                            T.writes(strided_set[ax0, T.int64(1)])
                                            with T.block("strided_set_global_o_8"):
                                                T.reads(y[ax0, T.int64(0)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), y.data, ax0 * T.int64(2), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("strided_set_local_gemmini_acc_o_1"):
                                                    T.reads()
                                                    T.writes(strided_set[ax0, T.int64(1)])
                                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), strided_set.data, ax0 * T.int64(5) + T.int64(1), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        with T.block("strided_set_i12_partition"):
                            T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(2)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(2)])
                            with T.block("strided_set_global_o_2"):
                                T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(2)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(2)])
                                with T.block("strided_set_global_o_9"):
                                    T.reads(x[T.int64(0):T.int64(2), T.int64(2)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(2)])
                                    for ax0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block(""):
                                            T.reads(x[ax0, T.int64(2)])
                                            T.writes(strided_set[ax0, T.int64(2)])
                                            with T.block("strided_set_global_o_10"):
                                                T.reads(x[ax0, T.int64(2)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), x.data, ax0 * T.int64(5) + T.int64(2), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("strided_set_local_gemmini_acc_o_2"):
                                                    T.reads()
                                                    T.writes(strided_set[ax0, T.int64(2)])
                                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), strided_set.data, ax0 * T.int64(5) + T.int64(2), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        with T.block("strided_set_i13_partition"):
                            T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(3)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(3)])
                            with T.block("strided_set_global_o_3"):
                                T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(3)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(3)])
                                with T.block("strided_set_global_o_11"):
                                    T.reads(y[T.int64(0):T.int64(2), T.int64(1)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(3)])
                                    for ax0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block(""):
                                            T.reads(y[ax0, T.int64(1)])
                                            T.writes(strided_set[ax0, T.int64(3)])
                                            with T.block("strided_set_global_o_12"):
                                                T.reads(y[ax0, T.int64(1)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), y.data, ax0 * T.int64(2) + T.int64(1), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("strided_set_local_gemmini_acc_o_3"):
                                                    T.reads()
                                                    T.writes(strided_set[ax0, T.int64(3)])
                                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), strided_set.data, ax0 * T.int64(5) + T.int64(3), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        with T.block("strided_set_i14_partition"):
                            T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(4)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(4)])
                            with T.block("strided_set_global_o_4"):
                                T.reads(y[0:T.int64(2), 0:T.int64(2)], x[T.int64(0):T.int64(2), T.int64(4)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(4)])
                                with T.block("strided_set_global_o_13"):
                                    T.reads(x[T.int64(0):T.int64(2), T.int64(4)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(4)])
                                    for ax0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block(""):
                                            T.reads(x[ax0, T.int64(4)])
                                            T.writes(strided_set[ax0, T.int64(4)])
                                            with T.block("strided_set_global_o_14"):
                                                T.reads(x[ax0, T.int64(4)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), x.data, ax0 * T.int64(5) + T.int64(4), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("strided_set_local_gemmini_acc_o_4"):
                                                    T.reads()
                                                    T.writes(strided_set[ax0, T.int64(4)])
                                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), strided_set.data, ax0 * T.int64(5) + T.int64(4), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                    T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 5), dtype="int32"), y: R.Tensor((2, 2), dtype="int32"), w: R.Tensor((3,), dtype="int32")) -> R.Tensor((2, 5), dtype="int32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_26_operator_scatter_slice, (x, y), out_sinfo=R.Tensor((2, 5), dtype="int32"))
            R.output(gv)
        return gv