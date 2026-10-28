# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def stage_58_operator_kernel(source: T.Buffer((3, 17), "bool"), permuted: T.Buffer((17, 3), "bool")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "62ed33fd186e4e4f", "global_symbol": "main", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", source.data, T.int64(51), source.data, T.int64(51), permuted.data, T.int64(51))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for i_0, j_0 in T.grid(17, 3):
                    with T.block("permuted_o"):
                        T.reads(source[j_0, i_0])
                        T.writes(permuted[i_0, j_0])
                        with T.block("byte_copy"):
                            T.reads(source[j_0, i_0])
                            T.writes(permuted[i_0, j_0])
                            T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("uint8"), source.data, j_0 * 17 + i_0, 17, 1), T.Cast("uint32", 0), T.uint32(1), T.uint32(1), 17, 0)
                            T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("uint8"), permuted.data, i_0 * 3 + j_0, 3, 2), T.Cast("uint32", 0), T.uint32(1), T.uint32(1), 3, 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(input0: R.Tensor((3, 17), dtype="bool")) -> R.Tensor((17, 3), dtype="bool"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_58_operator_kernel, (input0,), out_sinfo=R.Tensor((17, 3), dtype="bool"))
            R.output(gv)
        return gv