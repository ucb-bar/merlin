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
    def stage_32_operator_pad(x: T.Buffer((T.int64(1), T.int64(1), T.int64(3)), "int8"), MirrorPadInput: T.Buffer((T.int64(1), T.int64(1), T.int64(5)), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "2afbebc717dc5106", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(3), x.data, T.int64(3), MirrorPadInput.data, T.int64(5))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                for i0, i1_0, i2_0 in T.grid(T.int64(1), T.int64(1), T.int64(5)):
                    with T.block(""):
                        T.reads(x[T.int64(0), T.int64(0), 0:T.int64(3)])
                        T.writes(MirrorPadInput[T.int64(0), T.int64(0), i2_0])
                        with T.block("MirrorPadInput_o"):
                            T.reads(x[T.int64(0), T.int64(0), T.if_then_else(i2_0 == T.int64(4), T.int64(5) - i2_0, T.if_then_else(i2_0 < T.int64(1), T.int64(1) - i2_0, i2_0 - T.int64(1)))])
                            T.writes()
                            T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), x.data, T.if_then_else(i2_0 == T.int64(4), T.int64(5) - i2_0, T.if_then_else(i2_0 < T.int64(1), T.int64(1) - i2_0, i2_0 - T.int64(1))), T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                        for ax0 in range(T.int64(1)):
                            with T.block("MirrorPadInput_local_gemmini_spad_o"):
                                T.reads()
                                T.writes(MirrorPadInput[T.int64(0), T.int64(0), i2_0])
                                T.call_extern("void", "tvm_gemmini_dma_spad_out", T.tvm_access_ptr(T.type_annotation("int8"), MirrorPadInput.data, i2_0, T.int64(5), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                T.call_extern("void", "tvm_gemmini_end")