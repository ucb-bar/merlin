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
    def stage_56_operator_kernel(a: T.Buffer((256, 256), "int8"), b: T.Buffer((256, 256), "int8"), residual: T.Buffer((256, 256), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "a091c3a74f9318db", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", b.data, T.int64(65536), b.data, T.int64(65536), residual.data, T.int64(65536))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", a.data, T.int64(65536), a.data, T.int64(65536), residual.data, T.int64(65536))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    for i_0, j_0 in T.grid(16, 256):
                        with T.block("residual_o"):
                            T.reads(a[i_0 * 16:i_0 * 16 + 16, j_0], b[i_0 * 16:i_0 * 16 + 16, j_0])
                            T.writes(residual[i_0 * 16:i_0 * 16 + 16, j_0])
                            with T.block("affine_commands"):
                                T.reads(a[i_0 * 16:i_0 * 16 + 16, j_0], b[i_0 * 16:i_0 * 16 + 16, j_0])
                                T.writes(residual[i_0 * 16:i_0 * 16 + 16, j_0])
                                T.call_extern("void", "tvm_gemmini_dma_scaled_in", T.tvm_access_ptr(T.type_annotation("int8"), a.data, i_0 * 4096 + j_0, T.int64(4096), 1), T.Cast("uint32", 0), 16, 1, 256, T.uint32(1056964608), 0, 0)
                                T.call_extern("void", "tvm_gemmini_dma_scaled_in", T.tvm_access_ptr(T.type_annotation("int8"), b.data, i_0 * 4096 + j_0, T.int64(4096), 1), T.Cast("uint32", 0), 16, 1, 256, T.uint32(1048576000), 1, 0)
                                T.call_extern("void", "tvm_gemmini_dma_scaled_out", T.tvm_access_ptr(T.type_annotation("int8"), residual.data, i_0 * 4096 + j_0, T.int64(4096), 2), T.Cast("uint32", 0), 16, 1, 256, T.uint32(1065353216), 0, 0)
                    T.call_extern("void", "tvm_gemmini_end")