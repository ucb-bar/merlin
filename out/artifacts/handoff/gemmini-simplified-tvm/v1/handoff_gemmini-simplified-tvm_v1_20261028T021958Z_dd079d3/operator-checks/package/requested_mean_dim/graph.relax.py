# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_13_operator_mean(x: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"gemmini.workload": "8de82f5be5c4c91e", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        x_red = T.alloc_buffer((T.int64(2),))
        for ax0, k1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("x_red"):
                v_ax0, v_k1 = T.axis.remap("SR", [ax0, k1])
                T.reads(x[v_ax0, v_k1])
                T.writes(x_red[v_ax0])
                with T.init():
                    x_red[v_ax0] = T.float32(0.0)
                x_red[v_ax0] = x_red[v_ax0] + x[v_ax0, v_k1]
        for ax0 in range(T.int64(2)):
            with T.block("T_divide"):
                v_ax0 = T.axis.spatial(T.int64(2), ax0)
                T.reads(x_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = x_red[v_ax0] / T.float32(3.0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_13_operator_mean, (x,), out_sinfo=R.Tensor((2,), dtype="float32"))
            R.output(gv)
        return gv