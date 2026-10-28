# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func
    def stage_65_operator_kernel(image: T.Buffer((1, 3, 5, 7), "int8"), pool_max: T.Buffer((1, 3, 2, 3), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "6de801774bd62046", "global_symbol": "main", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.pool_scheduled": 1})
            pool_max_global = T.alloc_buffer((18, 1), "int8")
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", image.data, T.int64(105), image.data, T.int64(105), pool_max.data, T.int64(18))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for ax0_0_init in range(18):
                    with T.block("pool_max_o_init"):
                        T.reads()
                        T.writes(pool_max_global[ax0_0_init, 0])
                        with T.block("pool_max_init"):
                            T.reads()
                            T.writes(pool_max_global[ax0_0_init, 0])
                            pool_max_global[ax0_0_init, 0] = T.int8(-128)
                for ax0_0 in range(18):
                    with T.block(""):
                        T.reads(image[0, ax0_0 // 6, ax0_0 % 6 // 3 * 2:ax0_0 % 6 // 3 * 2 + 2, ax0_0 % 3 * 2:ax0_0 % 3 * 2 + 2], pool_max_global[ax0_0, 0])
                        T.writes(pool_max_global[ax0_0, 0], pool_max[0, ax0_0 // 6, ax0_0 % 6 // 3, ax0_0 % 3])
                        image_global = T.alloc_buffer((1, 4), "int8")
                        for ax0, ax1 in T.grid(1, 4):
                            with T.block("image_global"):
                                T.reads(image[0, ax0_0 // 6, ax0_0 % 6 // 3 * 2 + ax1 // 2, ax0_0 % 3 * 2 + ax1 % 2])
                                T.writes(image_global[0, ax1])
                                image_global[0, ax1] = image[0, ax0_0 // 6, ax0_0 % 6 // 3 * 2 + ax1 // 2, ax0_0 % 3 * 2 + ax1 % 2]
                        with T.block("pool_max_o_update"):
                            T.reads(image_global[0, 0:4])
                            T.writes(pool_max_global[ax0_0, 0])
                            with T.block("pooled_readout"):
                                T.reads(image_global[0, 0:4])
                                T.writes(pool_max_global[ax0_0, 0])
                                T.call_extern("void", "tvm_gemmini_dma_widen", T.tvm_access_ptr(T.type_annotation("int8"), image_global.data, 0, 4, 1), T.Cast("uint32", 0), T.uint32(4), T.uint32(1), T.int64(1), 1)
                                T.call_extern("void", "tvm_gemmini_dma_pool_out", T.tvm_access_ptr(T.type_annotation("int8"), pool_max_global.data, ax0_0, 1, 2), T.Cast("uint32", 0), T.uint32(2), 1)
                        for ax0 in range(1):
                            with T.block("pool_max_global"):
                                T.reads(pool_max_global[ax0_0, 0])
                                T.writes(pool_max[0, ax0_0 // 6, ax0_0 % 6 // 3, ax0_0 % 3])
                                pool_max[0, ax0_0 // 6, ax0_0 % 6 // 3, ax0_0 % 3] = pool_max_global[ax0_0, 0]
                T.call_extern("void", "tvm_gemmini_end")

    @R.function
    def main(input0: R.Tensor((1, 3, 5, 7), dtype="int8")) -> R.Tensor((1, 3, 2, 3), dtype="int8"):
        cls = Module
        with R.dataflow():
            gv = R.call_tir(cls.stage_65_operator_kernel, (input0,), out_sinfo=R.Tensor((1, 3, 2, 3), dtype="int8"))
            R.output(gv)
        return gv