# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_22_operator_col2im(x: T.Buffer((T.int64(1), T.int64(4), T.int64(6)), "float32"), overlap_sum: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.workload": "66bae785669c939d", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, offset_0, offset_1 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4), T.int64(2), T.int64(2)):
            with T.block("overlap_sum"):
                v_i0, v_i1, v_i2, v_i3, v_offset_0, v_offset_1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, offset_0, offset_1])
                T.reads(x[v_i0, v_i1 * T.int64(4) + v_offset_0 * T.int64(2) + v_offset_1, v_i2 * T.int64(3) + v_i3 - v_offset_1 - v_offset_0 * T.int64(3)])
                T.writes(overlap_sum[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    overlap_sum[v_i0, v_i1, v_i2, v_i3] = T.float32(0.0)
                overlap_sum[v_i0, v_i1, v_i2, v_i3] = overlap_sum[v_i0, v_i1, v_i2, v_i3] + T.if_then_else(T.int64(0) <= v_i2 - v_offset_0 and v_i2 - v_offset_0 < T.int64(2) and T.int64(0) <= v_i3 - v_offset_1 and v_i3 - v_offset_1 < T.int64(3), x[v_i0, v_i1 * T.int64(4) + v_offset_0 * T.int64(2) + v_offset_1, v_i2 * T.int64(3) + v_i3 - v_offset_1 - v_offset_0 * T.int64(3)], T.float32(0.0))

    @R.function
    def main(x: R.Tensor((1, 4, 6), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((1, 1, 3, 4), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_22_operator_col2im, (x,), out_sinfo=R.Tensor((1, 1, 3, 4), dtype="float32"))
            R.output(gv)
        return gv