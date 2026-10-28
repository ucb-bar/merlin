# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_21_operator_adaptive_pool_backward(x: T.Buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(2)), "float32"), adaptive_input_gradient: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.workload": "fd5d0d7be546a57e", "op_pattern": 8, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        adaptive_gradient_delta = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(2)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(2)):
            with T.block("adaptive_gradient_delta"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2, v_i3])
                T.writes(adaptive_gradient_delta[v_i0, v_i1, v_i2, v_i3])
                adaptive_gradient_delta[v_i0, v_i1, v_i2, v_i3] = x[v_i0, v_i1, v_i2, v_i3] / T.float32(2.0) / T.float32(2.0)
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4)):
            with T.block("adaptive_input_gradient_l1"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(adaptive_gradient_delta[v_i0, v_i1, v_i2 * T.int64(2) // T.int64(3):v_i2 * T.int64(2) // T.int64(3) + (v_i2 * T.int64(2) % T.int64(3) + T.int64(4)) // T.int64(3), v_i3 // T.int64(2)])
                T.writes(adaptive_input_gradient[v_i0, v_i1, v_i2, v_i3])
                for window_0, window_1 in T.grid((v_i2 * T.int64(2) % T.int64(3) + T.int64(4)) // T.int64(3), T.int64(1)):
                    with T.block("adaptive_input_gradient"):
                        v_i0_1 = T.axis.spatial((v_i0, v_i0 + T.int64(1)), v_i0)
                        v_i1_1 = T.axis.spatial((v_i1, v_i1 + T.int64(1)), v_i1)
                        v_i2_1 = T.axis.spatial((v_i2, v_i2 + T.int64(1)), v_i2)
                        v_i3_1 = T.axis.spatial((v_i3, v_i3 + T.int64(1)), v_i3)
                        v_window_0 = T.axis.reduce((v_i2 * T.int64(2) // T.int64(3), v_i2 * T.int64(2) // T.int64(3) + (v_i2 * T.int64(2) % T.int64(3) + T.int64(4)) // T.int64(3)), window_0 + v_i2 * T.int64(2) // T.int64(3))
                        v_window_1 = T.axis.reduce((v_i3 // T.int64(2), v_i3 // T.int64(2) + T.int64(1)), window_1 + v_i3 // T.int64(2))
                        T.reads(adaptive_gradient_delta[v_i0_1, v_i1_1, v_window_0, v_window_1])
                        T.writes(adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1])
                        with T.init():
                            adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1] = T.float32(0.0)
                        adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1] = adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1] + adaptive_gradient_delta[v_i0_1, v_i1_1, v_window_0, v_window_1]

    @R.function
    def main(x: R.Tensor((1, 1, 2, 2), dtype="float32"), y: R.Tensor((1, 1, 3, 4), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((1, 1, 3, 4), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_21_operator_adaptive_pool_backward, (x,), out_sinfo=R.Tensor((1, 1, 3, 4), dtype="float32"))
            R.output(gv)
        return gv