# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_11_operator_wrapping_subtract(A: T.Buffer((), "int32"), x: T.Buffer((T.int64(2), T.int64(3)), "int32"), wrapping_subtract: T.Buffer((T.int64(2), T.int64(3)), "int32")):
        T.func_attr({"gemmini.workload": "a3c23a3dc0811e91", "op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("wrapping_subtract"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(A[()], x[v_i0, v_i1])
                T.writes(wrapping_subtract[v_i0, v_i1])
                wrapping_subtract[v_i0, v_i1] = T.Cast("int32", T.Cast("int64", A[()]) - T.Cast("int64", x[v_i0, v_i1]))

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int32"), y: R.Tensor((2, 3), dtype="int32"), w: R.Tensor((3,), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_11_operator_wrapping_subtract, (R.const(0, "int32"), x), out_sinfo=R.Tensor((2, 3), dtype="int32"))
            R.output(gv)
        return gv