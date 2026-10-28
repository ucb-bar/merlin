# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_20_operator_avg_pool2d_backward(x: T.Buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(3)), "float32"), y: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32"), T_pool_grad: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.workload": "0624a03000d003a1", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, wh, ww in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4), T.int64(2), T.int64(2)):
            with T.block("T_pool_grad"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_wh, v_ww = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, wh, ww])
                T.reads(x[v_ax0, v_ax1, v_ax2 - v_wh, v_ax3 - v_ww])
                T.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.0)
                T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(2), T.int64(0), v_ax2 - T.int64(1)) <= v_ax2 - v_wh and v_ax2 - v_wh < T.int64(2) and T.Select(v_ax3 < T.int64(2), T.int64(0), v_ax3 - T.int64(1)) <= v_ax3 - v_ww and v_ax3 - v_ww < T.int64(3), x[v_ax0, v_ax1, v_ax2 - v_wh, v_ax3 - v_ww] / T.float32(4.0), T.float32(0.0))

    @R.function
    def main(x: R.Tensor((1, 1, 2, 3), dtype="float32"), y: R.Tensor((1, 1, 3, 4), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((1, 1, 3, 4), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_20_operator_avg_pool2d_backward, (x, y), out_sinfo=R.Tensor((1, 1, 3, 4), dtype="float32"))
            R.output(gv)
        return gv