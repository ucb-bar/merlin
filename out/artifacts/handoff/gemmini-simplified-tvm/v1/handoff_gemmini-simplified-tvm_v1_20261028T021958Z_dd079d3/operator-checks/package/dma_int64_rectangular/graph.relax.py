# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def stage_64_operator_kernel(source: T.Buffer((3, 18), "int64"), copied: T.Buffer((3, 18), "int64")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "725ab025bcda2ab7", "global_symbol": "main", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", source.data, T.int64(432), source.data, T.int64(432), copied.data, T.int64(432))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for i_0, j_0 in T.grid(1, 9):
                    with T.block("copied_o"):
                        T.reads(source[0:3, j_0 * 2:j_0 * 2 + 2])
                        T.writes(copied[0:3, j_0 * 2:j_0 * 2 + 2])
                        with T.block("byte_copy"):
                            T.reads(source[0:3, j_0 * 2:j_0 * 2 + 2])
                            T.writes(copied[0:3, j_0 * 2:j_0 * 2 + 2])
                            T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("uint8"), source.data, j_0 * 16, T.int64(432), 1), T.Cast("uint32", 0), T.uint32(3), T.uint32(16), T.int64(144), 0)
                            T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("uint8"), copied.data, j_0 * 16, T.int64(432), 2), T.Cast("uint32", 0), T.uint32(3), T.uint32(16), T.int64(144), 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(input0: R.Tensor((3, 18), dtype="int64")) -> R.Tensor((3, 18), dtype="int64"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_64_operator_kernel, (input0,), out_sinfo=R.Tensor((3, 18), dtype="int64"))
            R.output(gv)
        return gv