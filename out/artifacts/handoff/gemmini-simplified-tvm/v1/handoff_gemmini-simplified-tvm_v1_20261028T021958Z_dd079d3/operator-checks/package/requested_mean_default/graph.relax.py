# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_49_operator_mean(x: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((), "float32")):
        T.func_attr({"gemmini.workload": "45df7bfae6b7fb5d", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        x_red = T.alloc_buffer(())
        for k0, k1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("x_red"):
                v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                T.reads(x[v_k0, v_k1])
                T.writes(x_red[()])
                with T.init():
                    x_red[()] = T.float32(0.0)
                x_red[()] = x_red[()] + x[v_k0, v_k1]
        with T.block("T_divide"):
            vi = T.axis.spatial(1, T.int64(0))
            T.reads(x_red[()])
            T.writes(T_divide[()])
            T_divide[()] = x_red[()] / T.float32(6.0)

    @R.function
    def main(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32"), w: R.Tensor((3,), dtype="float32")) -> R.Tensor((), dtype="float32"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_49_operator_mean, (x,), out_sinfo=R.Tensor((), dtype="float32"))
            R.output(gv)
        return gv