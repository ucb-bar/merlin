# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_27_operator_scatter_slice(x: T.Buffer((T.int64(2), T.int64(3)), "float32"), lv: T.Buffer((T.int64(2), T.int64(1)), "float32"), strided_set: T.Buffer((T.int64(2), T.int64(3)), "float32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "32960c760a6653c4", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", lv.data, T.int64(8), lv.data, T.int64(8), strided_set.data, T.int64(24))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(24), x.data, T.int64(24), strided_set.data, T.int64(24))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    with T.block("strided_set_i1_common"):
                        T.reads(lv[0:T.int64(2), 0], x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        with T.block("strided_set_i10_partition"):
                            T.reads(lv[0:T.int64(2), 0], x[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                            with T.block("strided_set_global_o"):
                                T.reads(lv[0:T.int64(2), 0], x[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                with T.block("strided_set_global_o_2"):
                                    T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                                    for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                                        with T.block(""):
                                            T.reads(x[ax0_0, ax1_0])
                                            T.writes(strided_set[ax0_0, ax1_0])
                                            with T.block("strided_set_global_o_3"):
                                                T.reads(x[ax0_0, ax1_0])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), x.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                                            with T.block("strided_set_local_gemmini_acc_o"):
                                                T.reads()
                                                T.writes(strided_set[ax0_0, ax1_0])
                                                T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), strided_set.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        with T.block("strided_set_i11_partition"):
                            T.reads(lv[0:T.int64(2), 0], x[T.int64(0):T.int64(2), T.int64(2)])
                            T.writes(strided_set[T.int64(0):T.int64(2), T.int64(2)])
                            with T.block("strided_set_global_o_1"):
                                T.reads(lv[0:T.int64(2), 0], x[T.int64(0):T.int64(2), T.int64(2)])
                                T.writes(strided_set[T.int64(0):T.int64(2), T.int64(2)])
                                with T.block("strided_set_global_o_4"):
                                    T.reads(lv[T.int64(0):T.int64(2), T.int64(0)])
                                    T.writes(strided_set[T.int64(0):T.int64(2), T.int64(2)])
                                    for ax0, ax1_0 in T.grid(T.int64(2), T.int64(1)):
                                        with T.block(""):
                                            T.reads(lv[ax0, T.int64(0)])
                                            T.writes(strided_set[ax0, T.int64(2)])
                                            with T.block("strided_set_global_o_5"):
                                                T.reads(lv[ax0, T.int64(0)])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), lv.data, ax0, T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                                            for ax0_1 in range(T.int64(1)):
                                                with T.block("strided_set_local_gemmini_acc_o_1"):
                                                    T.reads()
                                                    T.writes(strided_set[ax0, T.int64(2)])
                                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), strided_set.data, ax0 * T.int64(3) + T.int64(2), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                    T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2,), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((2, 1), dtype="float32") = R.reshape(y, R.shape([2, 1]))
            gv = R.call_tir(cls.stage_27_operator_scatter_slice, (x, lv), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            R.output(gv)
        return gv