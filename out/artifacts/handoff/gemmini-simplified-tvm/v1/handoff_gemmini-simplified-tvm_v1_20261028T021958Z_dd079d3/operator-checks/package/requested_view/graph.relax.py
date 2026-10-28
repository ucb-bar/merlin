# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_48_operator_reshape(x: T.Buffer((T.int64(2), T.int64(3)), "int8"), T_reshape: T.Buffer((T.int64(3), T.int64(2)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "6c4286e8d7a9a795", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(6), x.data, T.int64(6), T_reshape.data, T.int64(6))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for ax0_0, ax1_0 in T.grid(T.int64(3), T.int64(2)):
                    with T.block(""):
                        T.reads(x[(ax0_0 * T.int64(2) + ax1_0) // T.int64(3), (ax0_0 * T.int64(2) + ax1_0) % T.int64(3)])
                        T.writes(T_reshape[ax0_0, ax1_0])
                        with T.block("T_reshape_o"):
                            T.reads(x[(ax0_0 * T.int64(2) + ax1_0) // T.int64(3), (ax0_0 * T.int64(2) + ax1_0) % T.int64(3)])
                            T.writes()
                            T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x.data, ax0_0 * T.int64(2) + ax1_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                        with T.block("T_reshape_local_gemmini_spad_o"):
                            T.reads()
                            T.writes(T_reshape[ax0_0, ax1_0])
                            T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), T_reshape.data, ax0_0 * T.int64(2) + ax1_0, T.int64(2), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(2), 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int8"), y: R.Tensor((2, 3), dtype="int8"), w: R.Tensor((3,), dtype="int8")) -> R.Tensor((3, 2), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_48_operator_reshape, (x,), out_sinfo=R.Tensor((3, 2), dtype="int8"))
            R.output(gv)
        return gv