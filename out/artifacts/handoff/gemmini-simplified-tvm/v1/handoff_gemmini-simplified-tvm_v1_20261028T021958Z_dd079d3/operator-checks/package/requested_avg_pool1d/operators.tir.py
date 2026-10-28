metadata = tvm.ir.load_json("""{
  \"root\": 1, 
  \"nodes\": [
    {
      \"type_key\": \"\"
    }, 
    {
      \"type_key\": \"Map\", 
      \"keys\": [
        \"Executor\"
      ], 
      \"data\": [2]
    }, 
    {
      \"type_key\": \"Array\", 
      \"data\": [3]
    }, 
    {
      \"type_key\": \"Executor\", 
      \"attrs\": {
        \"attrs\": \"5\", 
        \"name\": \"4\"
      }
    }, 
    {
      \"type_key\": \"runtime.String\", 
      \"repr_str\": \"aot\"
    }, 
    {
      \"type_key\": \"DictAttrs\", 
      \"attrs\": {\"__dict__\": \"6\"}
    }, 
    {
      \"type_key\": \"Map\", 
      \"keys\": [
        \"interface-api\", 
        \"unpacked-api\", 
        \"link-params\"
      ], 
      \"data\": [7, 8, 9]
    }, 
    {
      \"type_key\": \"runtime.String\", 
      \"repr_str\": \"c\"
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }, 
    {
      \"type_key\": \"IntImm\", 
      \"attrs\": {
        \"dtype\": \"bool\", 
        \"span\": \"0\", 
        \"value\": \"1\"
      }
    }
  ], 
  \"b64ndarrays\": [], 
  \"attrs\": {\"tvm_version\": \"0.19.0\"}
}""")
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    I.module_attrs({"executor": metadata["Executor"][0]})
    @T.prim_func
    def stage_14_operator_avg_pool1d(x: T.Buffer((T.int64(1), T.int64(1), T.int64(4)), "float32"), pool_avg: T.Buffer((T.int64(1), T.int64(1), T.int64(2)), "float32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "5182ba65def5d639", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            pad_temp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(6)))
            pool_sum = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2)))
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(16), x.data, T.int64(16), pool_avg.data, T.int64(8))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                with T.block("pad_temp_pad_const_o"):
                    T.reads()
                    T.writes(pad_temp[T.int64(0), T.int64(0), T.int64(0):T.int64(6)])
                    for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(6)):
                        with T.block(""):
                            T.reads()
                            T.writes(pad_temp[T.int64(0), T.int64(0), ax2_0])
                            with T.block("pad_temp_pad_const_o_1"):
                                T.reads()
                                T.writes()
                                with T.block("constant_source"):
                                    T.reads()
                                    T.writes()
                                    scalar = T.alloc_buffer(())
                                    scalar[()] = T.reinterpret("float32", T.uint32(0))
                                    T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), scalar.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(0), 1)
                            for ax0_1, ax1_1 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("pad_temp_local_gemmini_acc_o"):
                                    T.reads()
                                    T.writes(pad_temp[T.int64(0), T.int64(0), ax2_0])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), pad_temp.data, ax2_0, T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                T.call_extern("void", "tvm_gemmini_end")
                with T.block("pad_temp_o"):
                    T.reads(x[T.int64(0), T.int64(0), T.int64(0):T.int64(4)])
                    T.writes(pad_temp[T.int64(0), T.int64(0), T.int64(1):T.int64(5)])
                    for ax0, ax1, ax2_0 in T.grid(T.int64(1), T.int64(1), T.int64(4)):
                        with T.block(""):
                            T.reads(x[T.int64(0), T.int64(0), ax2_0])
                            T.writes(pad_temp[T.int64(0), T.int64(0), ax2_0 + T.int64(1)])
                            with T.block("pad_temp_o_1"):
                                T.reads(x[T.int64(0), T.int64(0), ax2_0])
                                T.writes()
                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), x.data, ax2_0, T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                            for ax0_1, ax1_1 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("pad_temp_local_gemmini_acc_o_1"):
                                    T.reads()
                                    T.writes(pad_temp[T.int64(0), T.int64(0), ax2_0 + T.int64(1)])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), pad_temp.data, ax2_0 + T.int64(1), T.int64(1), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(1), 0)
                T.call_extern("void", "tvm_gemmini_end")
                for ax0, ax1, ax2, rv0 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(3)):
                    with T.block("pool_sum"):
                        T.reads(pad_temp[T.int64(0), T.int64(0), ax2 * T.int64(2) + rv0])
                        T.writes(pool_sum[T.int64(0), T.int64(0), ax2])
                        if rv0 == T.int64(0):
                            pool_sum[T.int64(0), T.int64(0), ax2] = T.float32(0.0)
                        pool_sum[T.int64(0), T.int64(0), ax2] = pool_sum[T.int64(0), T.int64(0), ax2] + pad_temp[T.int64(0), T.int64(0), ax2 * T.int64(2) + rv0]
                for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(2)):
                    with T.block("pool_avg"):
                        T.reads(pool_sum[T.int64(0), T.int64(0), ax2])
                        T.writes(pool_avg[T.int64(0), T.int64(0), ax2])
                        T.block_attr({"schedule_rule": "meta_schedule.pool_avg"})
                        pool_avg[T.int64(0), T.int64(0), ax2] = pool_sum[T.int64(0), T.int64(0), ax2] / T.float32(3.0)
                T.call_extern("void", "tvm_gemmini_end")