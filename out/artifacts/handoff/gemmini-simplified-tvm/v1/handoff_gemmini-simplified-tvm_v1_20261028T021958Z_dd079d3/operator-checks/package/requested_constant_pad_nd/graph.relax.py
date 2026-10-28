# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_31_operator_pad(x: T.Buffer((T.int64(2), T.int64(3)), "int8"), PadInput: T.Buffer((T.int64(2), T.int64(5)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "6be9f43373af1ab0", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(6), x.data, T.int64(6), PadInput.data, T.int64(10))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                with T.block("PadInput_pad_const_o"):
                    T.reads()
                    T.writes(PadInput[T.int64(0):T.int64(2), T.int64(0):T.int64(5)])
                    for i0_0, i1_0 in T.grid(T.int64(2), T.int64(5)):
                        with T.block(""):
                            T.reads()
                            T.writes(PadInput[i0_0, i1_0])
                            with T.block("PadInput_pad_const_o_1"):
                                T.reads()
                                T.writes()
                                with T.block("constant_source"):
                                    T.reads()
                                    T.writes()
                                    scalar = T.alloc_buffer((), "int8")
                                    scalar[()] = T.int8(-17)
                                    T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), scalar.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(0), 1)
                            with T.block("PadInput_local_gemmini_spad_o"):
                                T.reads()
                                T.writes(PadInput[i0_0, i1_0])
                                T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), PadInput.data, i0_0 * T.int64(5) + i1_0, T.int64(5), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                T.call_extern("void", "tvm_gemmini_end")
                with T.block("PadInput_o"):
                    T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                    T.writes(PadInput[T.int64(0):T.int64(2), T.int64(1):T.int64(4)])
                    for i0_0, i1_0 in T.grid(T.int64(2), T.int64(3)):
                        with T.block(""):
                            T.reads(x[i0_0, i1_0])
                            T.writes(PadInput[i0_0, i1_0 + T.int64(1)])
                            with T.block("PadInput_o_1"):
                                T.reads(x[i0_0, i1_0])
                                T.writes()
                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x.data, i0_0 * T.int64(3) + i1_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                            with T.block("PadInput_local_gemmini_spad_o_1"):
                                T.reads()
                                T.writes(PadInput[i0_0, i1_0 + T.int64(1)])
                                T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), PadInput.data, i0_0 * T.int64(5) + i1_0 + T.int64(1), T.int64(5), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int8"), y: R.Tensor((2, 3), dtype="int8"), w: R.Tensor((3,), dtype="int8")) -> R.Tensor((2, 5), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_31_operator_pad, (x,), out_sinfo=R.Tensor((2, 5), dtype="int8"))
            R.output(gv)
        return gv