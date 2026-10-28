# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_17_operator_adaptive_avg_pool1d(x: T.Buffer((T.int64(1), T.int64(1), T.int64(3)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(1), T.int64(1), T.int64(2)), "float32")):
        T.func_attr({"gemmini.workload": "9a9a4e2c555b74cd", "op_pattern": 8, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        adaptive_pool_sum = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2)))
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
            with T.block("adaptive_pool_sum_l1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(x[v_ax0, v_ax1, v_ax2 * T.int64(3) // T.int64(2):v_ax2 * T.int64(3) // T.int64(2) + (v_ax2 % T.int64(2) + T.int64(2))])
                T.writes(adaptive_pool_sum[v_ax0, v_ax1, v_ax2])
                for rv0 in range(T.Select((v_ax2 + T.int64(1)) % T.int64(2) == T.int64(0), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2) + T.int64(1)) - v_ax2 * T.int64(3) // T.int64(2)):
                    with T.block("adaptive_pool_sum"):
                        v_ax0_1 = T.axis.spatial((v_ax0, v_ax0 + T.int64(1)), v_ax0)
                        v_ax1_1 = T.axis.spatial((v_ax1, v_ax1 + T.int64(1)), v_ax1)
                        v_ax2_1 = T.axis.spatial((v_ax2, v_ax2 + T.int64(1)), v_ax2)
                        v_rv0 = T.axis.reduce(T.Select((v_ax2 + T.int64(1)) % T.int64(2) == T.int64(0), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2) + T.int64(1)) - v_ax2 * T.int64(3) // T.int64(2), rv0)
                        T.reads(x[v_ax0_1, v_ax1_1, v_ax2_1 * T.int64(3) // T.int64(2) + v_rv0])
                        T.writes(adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1])
                        with T.init():
                            adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1] = T.float32(0.0)
                        adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1] = adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1] + x[v_ax0_1, v_ax1_1, v_ax2_1 * T.int64(3) // T.int64(2) + v_rv0]
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
            with T.block("adaptive_pool_avg"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(adaptive_pool_sum[v_ax0, v_ax1, v_ax2])
                T.writes(adaptive_pool_avg[v_ax0, v_ax1, v_ax2])
                T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                adaptive_pool_avg[v_ax0, v_ax1, v_ax2] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2] / T.Cast("float32", T.Select((v_ax2 + T.int64(1)) % T.int64(2) == T.int64(0), (T.Cast("int64", v_ax2) * T.int64(3) + T.int64(3)) // T.int64(2), (T.Cast("int64", v_ax2) * T.int64(3) + T.int64(3)) // T.int64(2) + T.int64(1)) - T.Cast("int64", v_ax2) * T.int64(3) // T.int64(2))

    @R.function
    def main(x: R.Tensor((1, 1, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((1, 1, 2), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_17_operator_adaptive_avg_pool1d, (x,), out_sinfo=R.Tensor((1, 1, 2), dtype="float32"))
            R.output(gv)
        return gv