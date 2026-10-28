# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_19_operator_adaptive_avg_pool3d(x: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4), T.int64(5)), "float32"), adaptive_pool_avg: T.Buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(2), T.int64(2)), "float32")):
        T.func_attr({"gemmini.workload": "c7158268ee9419db", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        adaptive_pool_sum = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(2), T.int64(2)))
        for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(2), T.int64(2)):
            with T.block("adaptive_pool_sum_init"):
                v_ax0_init, v_ax1_init, v_ax2_init, v_ax3_init, v_ax4_init = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads()
                T.writes(adaptive_pool_sum[v_ax0_init, v_ax1_init, v_ax2_init, v_ax3_init, v_ax4_init])
                adaptive_pool_sum[v_ax0_init, v_ax1_init, v_ax2_init, v_ax3_init, v_ax4_init] = T.float32(0.0)
            for rv1 in range(T.int64(2)):
                with T.block("adaptive_pool_sum_l1"):
                    v_ax0, v_ax1, v_ax2, v_ax3, v_ax4, v_rv1 = T.axis.remap("SSSSSR", [ax0, ax1, ax2, ax3, ax4, rv1])
                    T.reads(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], x[v_ax0, v_ax1, v_ax2 * T.int64(3) // T.int64(2):v_ax2 * T.int64(3) // T.int64(2) + (v_ax2 % T.int64(2) + T.int64(2)), v_ax3 * T.int64(2) + v_rv1, v_ax4 * T.int64(5) // T.int64(2):v_ax4 * T.int64(5) // T.int64(2) + (v_ax4 % T.int64(2) + T.int64(3))])
                    T.writes(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                    for rv0, rv2 in T.grid(T.Select((v_ax2 + T.int64(1)) % T.int64(2) == T.int64(0), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2) + T.int64(1)) - v_ax2 * T.int64(3) // T.int64(2), T.Select((v_ax4 + T.int64(1)) % T.int64(2) == T.int64(0), (v_ax4 * T.int64(5) + T.int64(5)) // T.int64(2), (v_ax4 * T.int64(5) + T.int64(5)) // T.int64(2) + T.int64(1)) - v_ax4 * T.int64(5) // T.int64(2)):
                        with T.block("adaptive_pool_sum"):
                            v_ax0_1 = T.axis.spatial((v_ax0, v_ax0 + T.int64(1)), v_ax0)
                            v_ax1_1 = T.axis.spatial((v_ax1, v_ax1 + T.int64(1)), v_ax1)
                            v_ax2_1 = T.axis.spatial((v_ax2, v_ax2 + T.int64(1)), v_ax2)
                            v_ax3_1 = T.axis.spatial((v_ax3, v_ax3 + T.int64(1)), v_ax3)
                            v_ax4_1 = T.axis.spatial((v_ax4, v_ax4 + T.int64(1)), v_ax4)
                            v_rv0 = T.axis.reduce(T.Select((v_ax2 + T.int64(1)) % T.int64(2) == T.int64(0), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2), (v_ax2 * T.int64(3) + T.int64(3)) // T.int64(2) + T.int64(1)) - v_ax2 * T.int64(3) // T.int64(2), rv0)
                            v_rv1_1 = T.axis.reduce((v_rv1, v_rv1 + T.int64(1)), v_rv1)
                            v_rv2 = T.axis.reduce(T.Select((v_ax4 + T.int64(1)) % T.int64(2) == T.int64(0), (v_ax4 * T.int64(5) + T.int64(5)) // T.int64(2), (v_ax4 * T.int64(5) + T.int64(5)) // T.int64(2) + T.int64(1)) - v_ax4 * T.int64(5) // T.int64(2), rv2)
                            T.reads(adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1, v_ax4_1], x[v_ax0_1, v_ax1_1, v_ax2_1 * T.int64(3) // T.int64(2) + v_rv0, v_ax3_1 * T.int64(2) + v_rv1_1, v_ax4_1 * T.int64(5) // T.int64(2) + v_rv2])
                            T.writes(adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1, v_ax4_1])
                            adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1, v_ax4_1] = adaptive_pool_sum[v_ax0_1, v_ax1_1, v_ax2_1, v_ax3_1, v_ax4_1] + x[v_ax0_1, v_ax1_1, v_ax2_1 * T.int64(3) // T.int64(2) + v_rv0, v_ax3_1 * T.int64(2) + v_rv1_1, v_ax4_1 * T.int64(5) // T.int64(2) + v_rv2]
        for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(2), T.int64(2)):
            with T.block("adaptive_pool_avg"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
                T.reads(adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                T.writes(adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
                T.block_attr({"schedule_rule": "meta_schedule.adaptive_pool_avg"})
                adaptive_pool_avg[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = adaptive_pool_sum[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] / (T.Cast("float32", T.Select((v_ax2 + T.int64(1)) % T.int64(2) == T.int64(0), (T.Cast("int64", v_ax2) * T.int64(3) + T.int64(3)) // T.int64(2), (T.Cast("int64", v_ax2) * T.int64(3) + T.int64(3)) // T.int64(2) + T.int64(1)) - T.Cast("int64", v_ax2) * T.int64(3) // T.int64(2)) * T.float32(2.0) * T.Cast("float32", T.Select((v_ax4 + T.int64(1)) % T.int64(2) == T.int64(0), (T.Cast("int64", v_ax4) * T.int64(5) + T.int64(5)) // T.int64(2), (T.Cast("int64", v_ax4) * T.int64(5) + T.int64(5)) // T.int64(2) + T.int64(1)) - T.Cast("int64", v_ax4) * T.int64(5) // T.int64(2)))

    @R.function
    def main(x: R.Tensor((1, 1, 3, 4, 5), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((1, 1, 2, 2, 2), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_19_operator_adaptive_avg_pool3d, (x,), out_sinfo=R.Tensor((1, 1, 2, 2, 2), dtype="float32"))
            R.output(gv)
        return gv