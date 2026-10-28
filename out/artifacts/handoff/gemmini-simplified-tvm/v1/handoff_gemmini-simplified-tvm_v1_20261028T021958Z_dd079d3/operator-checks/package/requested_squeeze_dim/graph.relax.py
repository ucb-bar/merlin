# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_45_operator_squeeze(x: T.Buffer((T.int64(1), T.int64(2), T.int64(3)), "int32"), T_squeeze: T.Buffer((T.int64(2), T.int64(3)), "int32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "0aed5e21f2bc9efb", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(24), x.data, T.int64(24), T_squeeze.data, T.int64(24))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(3)):
                    with T.block(""):
                        T.reads(x[T.int64(0), ax0_0, ax1_0])
                        T.writes(T_squeeze[ax0_0, ax1_0])
                        with T.block("T_squeeze_o"):
                            T.reads(x[T.int64(0), ax0_0, ax1_0])
                            T.writes()
                            T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), x.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                        with T.block("T_squeeze_local_gemmini_acc_o"):
                            T.reads()
                            T.writes(T_squeeze[ax0_0, ax1_0])
                            T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), T_squeeze.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(x: R.Tensor((1, 2, 3), dtype="int32"), y: R.Tensor((2, 3), dtype="int32"), w: R.Tensor((3,), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_45_operator_squeeze, (x,), out_sinfo=R.Tensor((2, 3), dtype="int32"))
            R.output(gv)
        return gv