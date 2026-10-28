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
    def stage_66_operator_kernel(image: T.Buffer((1, 3, 5, 7), "int8"), pool_max: T.Buffer((1, 3, 3, 4), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "10d9b80f676fb7b6", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.pool_scheduled": 1})
            pad_temp = T.alloc_buffer((1, 3, 7, 9), "int8")
            pool_max_global = T.alloc_buffer((36, 1), "int8")
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", image.data, T.int64(105), image.data, T.int64(105), pool_max.data, T.int64(36))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for ax0, ax1, ax2, ax3 in T.grid(1, 3, 7, 9):
                    with T.block("pad_temp"):
                        T.reads(image[0, ax1, ax2 - 1, ax3 - 1])
                        T.writes(pad_temp[0, ax1, ax2, ax3])
                        pad_temp[0, ax1, ax2, ax3] = T.if_then_else(1 <= ax2 and ax2 < 6 and 1 <= ax3 and ax3 < 8, image[0, ax1, ax2 - 1, ax3 - 1], T.int8(-128))
                for ax0_0_init in range(36):
                    with T.block("pool_max_o_init"):
                        T.reads()
                        T.writes(pool_max_global[ax0_0_init, 0])
                        with T.block("pool_max_init"):
                            T.reads()
                            T.writes(pool_max_global[ax0_0_init, 0])
                            pool_max_global[ax0_0_init, 0] = T.int8(-128)
                for ax0_0 in range(36):
                    with T.block(""):
                        T.reads(pad_temp[0, ax0_0 // 12, ax0_0 % 12 // 4 * 2:ax0_0 % 12 // 4 * 2 + 3, ax0_0 % 4 * 2:ax0_0 % 4 * 2 + 3], pool_max_global[ax0_0, 0])
                        T.writes(pool_max_global[ax0_0, 0], pool_max[0, ax0_0 // 12, ax0_0 % 12 // 4, ax0_0 % 4])
                        pad_temp_global = T.alloc_buffer((1, 9), "int8")
                        for ax0, ax1 in T.grid(1, 9):
                            with T.block("pad_temp_global"):
                                T.reads(pad_temp[0, ax0_0 // 12, ax0_0 % 12 // 4 * 2 + ax1 // 3, ax0_0 % 4 * 2 + ax1 % 3])
                                T.writes(pad_temp_global[0, ax1])
                                pad_temp_global[0, ax1] = pad_temp[0, ax0_0 // 12, ax0_0 % 12 // 4 * 2 + ax1 // 3, ax0_0 % 4 * 2 + ax1 % 3]
                        with T.block("pool_max_o_update"):
                            T.reads(pad_temp_global[0, 0:9])
                            T.writes(pool_max_global[ax0_0, 0])
                            with T.block("pooled_readout"):
                                T.reads(pad_temp_global[0, 0:9])
                                T.writes(pool_max_global[ax0_0, 0])
                                T.call_extern("void", "tvm_gemmini_dma_widen", T.tvm_access_ptr(T.type_annotation("int8"), pad_temp_global.data, 0, 9, 1), T.Cast("uint32", 0), T.uint32(9), T.uint32(1), T.int64(1), 1)
                                T.call_extern("void", "tvm_gemmini_dma_pool_out", T.tvm_access_ptr(T.type_annotation("int8"), pool_max_global.data, ax0_0, 1, 2), T.Cast("uint32", 0), T.uint32(3), 1)
                        for ax0 in range(1):
                            with T.block("pool_max_global"):
                                T.reads(pool_max_global[ax0_0, 0])
                                T.writes(pool_max[0, ax0_0 // 12, ax0_0 % 12 // 4, ax0_0 % 4])
                                pool_max[0, ax0_0 // 12, ax0_0 % 12 // 4, ax0_0 % 4] = pool_max_global[ax0_0, 0]
                T.call_extern("void", "tvm_gemmini_end")