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
    def stage_67_operator_kernel(windows: T.Buffer((3, 5, 4), "int8"), window_average: T.Buffer((3, 5), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "2f36f4aa777361c9", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.readout_fused": 1, "gemmini.readout_proof": "{\"input_domain\": [-2147483648, 2147483647], \"proof\": \"equal monotone output-code thresholds over every signed INT32 input\", \"relu\": false, \"scale_bits\": 1048576000, \"thresholds_sha256\": \"97bb5d49f59976e11420db1fbb865ae6e07ca6c1181386ff064621fe00101635\", \"transitions_checked\": 255}", "gemmini.sum_chunk": 4, "gemmini.sum_rows": 15})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", windows.data, T.int64(60), windows.data, T.int64(60), window_average.data, T.int64(15))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for ax0_0 in range(1):
                    with T.block(""):
                        T.reads(windows[0:3, 0:5, 0:4])
                        T.writes(window_average[0:3, 0:5])
                        window_average_global = T.alloc_buffer((15, 1), "int8")
                        with T.block("window_sum_init_o"):
                            T.reads()
                            T.writes()
                            T.call_extern("void", "tvm_gemmini_zero_acc", T.Cast("uint32", 0), T.uint32(15), T.uint32(1))
                        for ax1_0 in range(1):
                            with T.block(""):
                                T.reads(windows[0:3, 0:5, 0:4])
                                T.writes()
                                windows_global = T.alloc_buffer((15, 4), "int8")
                                for ax0, ax1 in T.grid(15, 4):
                                    with T.block("windows_global"):
                                        T.reads(windows[ax0 // 5, ax0 % 5, ax1])
                                        T.writes(windows_global[ax0, ax1])
                                        windows_global[ax0, ax1] = windows[ax0 // 5, ax0 % 5, ax1]
                                with T.block("windows_global_local_gemmini_spad_o"):
                                    T.reads(windows_global[0:15, 0:4])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), windows_global.data, 0, T.int64(60), 1), T.Cast("uint32", 0), T.uint32(15), T.uint32(4), 4, 1)
                                with T.block("window_sum_update_o"):
                                    T.reads()
                                    T.writes()
                                    with T.block("ones_product"):
                                        T.reads()
                                        T.writes()
                                        one = T.alloc_buffer((), "int8")
                                        one[()] = T.int8(1)
                                        T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), one.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", 16), T.uint32(4), T.uint32(1), T.int64(0), 1)
                                        T.call_extern("void", "tvm_gemmini_compute", T.Cast("uint32", 0), T.Cast("uint32", 16), T.Cast("uint32", 0), T.uint32(15), T.uint32(1), T.uint32(4), 1)
                        with T.block("window_sum_global_local_gemmini_acc_o"):
                            T.reads()
                            T.writes(window_average_global[0:15, 0])
                            T.call_extern("void", "tvm_gemmini_dma_scaled_out", T.tvm_access_ptr(T.type_annotation("int8"), window_average_global.data, 0, T.int64(15), 2), T.Cast("uint32", 0), T.uint32(15), T.uint32(1), 1, T.uint32(1048576000), 0, 1)
                        for ax0 in range(15):
                            with T.block("window_average_global"):
                                T.reads(window_average_global[ax0, 0])
                                T.writes(window_average[ax0 // 5, ax0 % 5])
                                window_average[ax0 // 5, ax0 % 5] = window_average_global[ax0, 0]
                T.call_extern("void", "tvm_gemmini_end")