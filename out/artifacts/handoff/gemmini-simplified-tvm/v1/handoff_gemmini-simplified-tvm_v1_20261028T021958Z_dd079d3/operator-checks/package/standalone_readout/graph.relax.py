# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def stage_69_operator_kernel(accumulator: T.Buffer((3, 17), "int32"), readout: T.Buffer((3, 17), "int8")):
        T.func_attr({"gemmini.workload": "693993567a6065d1", "global_symbol": "main", "op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(3, 17):
            with T.block("readout"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(accumulator[v_i, v_j])
                T.writes(readout[v_i, v_j])
                readout[v_i, v_j] = T.Cast("int8", T.min(T.max(T.nearbyint(T.max(T.Cast("float32", accumulator[v_i, v_j]) * T.float32(0.0031129), T.float32(0.0))), T.float32(-128.0)), T.float32(127.0)))

    @R.function
    def main(input0: R.Tensor((3, 17), dtype="int32")) -> R.Tensor((3, 17), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_69_operator_kernel, (input0,), out_sinfo=R.Tensor((3, 17), dtype="int8"))
            R.output(gv)
        return gv