# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_10_operator_fused_cast_divide(x: T.Buffer((T.int64(2), T.int64(3)), "int32"), T_divide_intermediate: T.Buffer((T.int64(2), T.int64(3)), "float32")):
        T.func_attr({"gemmini.workload": "b77ddc32443a9ce2", "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("T_divide"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(x[v_ax0, v_ax1])
                T.writes(T_divide_intermediate[v_ax0, v_ax1])
                T_divide_intermediate[v_ax0, v_ax1] = T.Cast("float32", x[v_ax0, v_ax1]) / T.float32(3.0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="int32"), y: R.Tensor((2, 3), dtype="int32"), w: R.Tensor((3,), dtype="int32")) -> R.Tensor((2, 3), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_10_operator_fused_cast_divide, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
            R.output(gv)
        return gv