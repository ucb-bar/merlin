# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_25_operator_concatenate(x: T.Buffer((T.int64(2), T.int64(3)), "int8"), y: T.Buffer((T.int64(1), T.int64(3)), "int8"), T_concat: T.Buffer((T.int64(3), T.int64(3)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "f42e29d6bd93294f", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", y.data, T.int64(3), y.data, T.int64(3), T_concat.data, T.int64(9))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(6), x.data, T.int64(6), T_concat.data, T.int64(9))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    with T.block("T_concat_ax0_common"):
                        T.reads(y[T.int64(-2):T.int64(1), T.int64(0):T.int64(3)], x[T.int64(0):T.int64(3), T.int64(0):T.int64(3)])
                        T.writes(T_concat[T.int64(0):T.int64(3), T.int64(0):T.int64(3)])
                        with T.block("T_concat_ax00_partition"):
                            T.reads(y[T.int64(-2):T.int64(0), T.int64(0):T.int64(3)], x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                            T.writes(T_concat[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                            with T.block("T_concat_global_o"):
                                T.reads(y[T.int64(-2):T.int64(0), T.int64(0):T.int64(3)], x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                T.writes(T_concat[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                with T.block("T_concat_global_o_2"):
                                    T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                    T.writes(T_concat[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                    for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(3)):
                                        with T.block(""):
                                            T.reads(x[ax0_0, ax1_0])
                                            T.writes(T_concat[ax0_0, ax1_0])
                                            with T.block("T_concat_global_o_3"):
                                                T.reads(x[ax0_0, ax1_0])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                                            with T.block("T_concat_local_gemmini_spad_o"):
                                                T.reads()
                                                T.writes(T_concat[ax0_0, ax1_0])
                                                T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), T_concat.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        with T.block("T_concat_ax01_partition"):
                            T.reads(y[T.int64(0), T.int64(0):T.int64(3)], x[T.int64(2), T.int64(0):T.int64(3)])
                            T.writes(T_concat[T.int64(2), T.int64(0):T.int64(3)])
                            with T.block("T_concat_global_o_1"):
                                T.reads(y[T.int64(0), T.int64(0):T.int64(3)], x[T.int64(2), T.int64(0):T.int64(3)])
                                T.writes(T_concat[T.int64(2), T.int64(0):T.int64(3)])
                                with T.block("T_concat_global_o_4"):
                                    T.reads(y[T.int64(0), T.int64(0):T.int64(3)])
                                    T.writes(T_concat[T.int64(2), T.int64(0):T.int64(3)])
                                    for ax0, ax1_0 in T.grid(T.int64(1), T.int64(3)):
                                        with T.block(""):
                                            T.reads(y[T.int64(0), ax1_0])
                                            T.writes(T_concat[T.int64(2), ax1_0])
                                            with T.block("T_concat_global_o_5"):
                                                T.reads(y[T.int64(0), ax1_0])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), y.data, ax1_0, T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("T_concat_local_gemmini_spad_o_1"):
                                                    T.reads()
                                                    T.writes(T_concat[T.int64(2), ax1_0])
                                                    T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), T_concat.data, ax1_0 + T.int64(6), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                    T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int8"), y: R.Tensor((1, 3), dtype="int8"), w: R.Tensor((3,), dtype="int8")) -> R.Tensor((3, 3), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_25_operator_concatenate, (x, y), out_sinfo=R.Tensor((3, 3), dtype="int8"))
            R.output(gv)
        return gv