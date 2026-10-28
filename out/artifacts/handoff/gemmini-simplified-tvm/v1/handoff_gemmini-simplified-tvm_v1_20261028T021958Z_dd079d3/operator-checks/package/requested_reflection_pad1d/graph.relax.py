# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_32_operator_pad(x: T.Buffer((T.int64(1), T.int64(1), T.int64(3)), "int8"), MirrorPadInput: T.Buffer((T.int64(1), T.int64(1), T.int64(5)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "2afbebc717dc5106", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(3), x.data, T.int64(3), MirrorPadInput.data, T.int64(5))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for i0, i1_0, i2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                    with T.block(""):
                        T.reads(x[T.int64(0), T.int64(0), 0:T.int64(3)])
                        T.writes(MirrorPadInput[T.int64(0), T.int64(0), i2_0])
                        with T.block("MirrorPadInput_o"):
                            T.reads(x[T.int64(0), T.int64(0), T.if_then_else(i2_0 == T.int64(4), T.int64(5) - i2_0, T.if_then_else(i2_0 < T.int64(1), T.int64(1) - i2_0, i2_0 - T.int64(1)))])
                            T.writes()
                            T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x.data, T.if_then_else(i2_0 == T.int64(4), T.int64(5) - i2_0, T.if_then_else(i2_0 < T.int64(1), T.int64(1) - i2_0, i2_0 - T.int64(1))), T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                        for ax0 in range(T.int64(1)):
                            with T.block("MirrorPadInput_local_gemmini_spad_o"):
                                T.reads()
                                T.writes(MirrorPadInput[T.int64(0), T.int64(0), i2_0])
                                T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), MirrorPadInput.data, i2_0, T.int64(5), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((1, 1, 3), dtype="int8"), y: R.Tensor((2, 3), dtype="int8"), w: R.Tensor((3,), dtype="int8")) -> R.Tensor((1, 1, 5), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_32_operator_pad, (x,), out_sinfo=R.Tensor((1, 1, 5), dtype="int8"))
            R.output(gv)
        return gv