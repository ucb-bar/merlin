# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_9_operator_wrapping_multiply(x: T.Buffer((T.int64(2), T.int64(3)), "int32"), B: T.Buffer((), "int32"), wrapping_multiply: T.Buffer((T.int64(2), T.int64(3)), "int32")):
        T.func_attr({"gemmini.workload": "6b15de769cb37b5a", "op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("wrapping_multiply"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(x[v_i0, v_i1], B[()])
                T.writes(wrapping_multiply[v_i0, v_i1])
                wrapping_multiply[v_i0, v_i1] = T.Cast("int32", T.Cast("int64", x[v_i0, v_i1]) * T.Cast("int64", B[()]))

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int32"), y: R.Tensor((2, 3), dtype="int32"), w: R.Tensor((3,), dtype="int32")) -> R.Tensor((2, 3), dtype="int32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_9_operator_wrapping_multiply, (x, R.const(-3, "int32")), out_sinfo=R.Tensor((2, 3), dtype="int32"))
            R.output(gv)
        return gv