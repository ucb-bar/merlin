# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_50_operator_fused_cast_sum(x: T.Buffer((T.int64(2), T.int64(3)), "int8"), lv_red_intermediate: T.Buffer((T.int64(2),), "int64")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "7bc17b3189b58296", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.sum_chunk": 3, "gemmini.sum_rows": 2})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(6), x.data, T.int64(6), lv_red_intermediate.data, T.int64(16))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for ax0_0 in range(T.int64(1)):
                    with T.block(""):
                        T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        T.writes(lv_red_intermediate[T.int64(0):T.int64(2)])
                        lv_red_intermediate_global = T.alloc_buffer((T.int64(2), T.int64(1)), "int32")
                        with T.block("lv_red_init_o"):
                            T.reads()
                            T.writes()
                            T.call_extern("void", "tvm_gemmini_zero_acc", T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(1))
                        for ax1_0 in range(T.int64(1)):
                            with T.block(""):
                                T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                T.writes()
                                x_global = T.alloc_buffer((T.int64(2), T.int64(3)), "int8")
                                for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                                    with T.block("x_global"):
                                        T.reads(x[ax0, ax1])
                                        T.writes(x_global[ax0, ax1])
                                        x_global[ax0, ax1] = x[ax0, ax1]
                                with T.block("x_global_local_gemmini_spad_o"):
                                    T.reads(x_global[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x_global.data, T.int64(0), T.int64(6), 1), T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(3), T.int64(3), 1)
                                with T.block("lv_red_update_o"):
                                    T.reads()
                                    T.writes()
                                    with T.block("ones_product"):
                                        T.reads()
                                        T.writes()
                                        one = T.alloc_buffer((), "int8")
                                        one[()] = T.int8(1)
                                        T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), one.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", 16), T.uint32(3), T.uint32(1), T.int64(0), 1)
                                        T.call_extern("void", "tvm_gemmini_compute", T.Cast("uint32", T.int64(0)), T.Cast("uint32", 16), T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(1), T.uint32(3), 1)
                        with T.block("lv_red_intermediate_global_local_gemmini_acc_o"):
                            T.reads()
                            T.writes(lv_red_intermediate_global[T.int64(0):T.int64(2), T.int64(0)])
                            T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), lv_red_intermediate_global.data, T.int64(0), T.int64(2), 2), T.Cast("uint32", T.int64(0)), T.uint32(2), T.uint32(1), T.int64(1), 1)
                        for ax0 in range(T.int64(2)):
                            with T.block("lv_red_intermediate_global"):
                                T.reads(lv_red_intermediate_global[ax0, T.int64(0)])
                                T.writes(lv_red_intermediate[ax0])
                                lv_red_intermediate[ax0] = T.Cast("int64", lv_red_intermediate_global[ax0, T.int64(0)])
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int8"), y: R.Tensor((2, 3), dtype="int8"), w: R.Tensor((3,), dtype="int8")) -> R.Tensor((2,), dtype="int64"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_50_operator_fused_cast_sum, (x,), out_sinfo=R.Tensor((2,), dtype="int64"))
            R.output(gv)
        return gv