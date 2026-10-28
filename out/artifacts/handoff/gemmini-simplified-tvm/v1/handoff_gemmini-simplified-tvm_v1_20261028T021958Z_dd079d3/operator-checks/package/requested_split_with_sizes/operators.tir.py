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
    def stage_44_operator_split(x: T.Buffer((T.int64(2), T.int64(5)), "int32"), T_split: T.Buffer((T.int64(2), T.int64(2)), "int32"), T_split_1: T.Buffer((T.int64(2), T.int64(3)), "int32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "4bb22e4161806982", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(40), x.data, T.int64(40), T_split_1.data, T.int64(24))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(40), x.data, T.int64(40), T_split.data, T.int64(16))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    with T.block("T_split_o"):
                        T.reads(x[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                        T.writes(T_split[T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(2)):
                            with T.block(""):
                                T.reads(x[ax0_0, ax1_0])
                                T.writes(T_split[ax0_0, ax1_0])
                                with T.block("T_split_o_1"):
                                    T.reads(x[ax0_0, ax1_0])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), x.data, ax0_0 * T.int64(5) + ax1_0, T.int64(5), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                                with T.block("T_split_local_gemmini_acc_o"):
                                    T.reads()
                                    T.writes(T_split[ax0_0, ax1_0])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), T_split.data, ax0_0 * T.int64(2) + ax1_0, T.int64(2), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(2), 0)
                    with T.block("T_split_1_o"):
                        T.reads(x[T.int64(0):T.int64(2), T.int64(2):T.int64(5)])
                        T.writes(T_split_1[T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        for ax0_0, ax1_0 in T.grid(T.int64(2), T.int64(3)):
                            with T.block(""):
                                T.reads(x[ax0_0, ax1_0 + T.int64(2)])
                                T.writes(T_split_1[ax0_0, ax1_0])
                                with T.block("T_split_1_o_1"):
                                    T.reads(x[ax0_0, ax1_0 + T.int64(2)])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), x.data, ax0_0 * T.int64(5) + ax1_0 + T.int64(2), T.int64(5), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                                with T.block("T_split_local_gemmini_acc_o_1"):
                                    T.reads()
                                    T.writes(T_split_1[ax0_0, ax1_0])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), T_split_1.data, ax0_0 * T.int64(3) + ax1_0, T.int64(3), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                    T.call_extern("void", "tvm_gemmini_end")