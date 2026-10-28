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
    def stage_55_operator_kernel(a: T.Buffer((17, 19), "int8"), b: T.Buffer((17, 19), "int8"), affine: T.Buffer((17, 19), "int8")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "cceeb08a87f7ce26", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", b.data, T.int64(323), b.data, T.int64(323), affine.data, T.int64(323))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", a.data, T.int64(323), a.data, T.int64(323), affine.data, T.int64(323))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    for i_0, j_0 in T.grid(17, 19):
                        with T.block("affine_o"):
                            T.reads(b[i_0, j_0], a[i_0, j_0])
                            T.writes(affine[i_0, j_0])
                            with T.block("affine_commands"):
                                T.reads(b[i_0, j_0], a[i_0, j_0])
                                T.writes(affine[i_0, j_0])
                                coefficient = T.alloc_buffer((), "int8")
                                offset = T.alloc_buffer((), "int32")
                                offset[()] = 19
                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("int32"), offset.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", 0), 1, 1, T.int64(0), 1)
                                coefficient[()] = T.int8(2)
                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), b.data, i_0 * 19 + j_0, 19, 1), T.Cast("uint32", 0), 1, 1, 19, 0)
                                T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), coefficient.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", 16), 1, 1, T.int64(1), 1)
                                T.call_extern("void", "tvm_gemmini_compute_ws", T.Cast("uint32", 0), T.Cast("uint32", 16), T.Cast("uint32", 0), 1, 1, 1, 1)
                                coefficient[()] = T.int8(-3)
                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), a.data, i_0 * 19 + j_0, 19, 1), T.Cast("uint32", 0), 1, 1, 19, 0)
                                T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), coefficient.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", 16), 1, 1, T.int64(1), 1)
                                T.call_extern("void", "tvm_gemmini_compute_ws", T.Cast("uint32", 0), T.Cast("uint32", 16), T.Cast("uint32", 0), 1, 1, 1, 1)
                                T.call_extern("void", "tvm_gemmini_dma_scaled_out", T.tvm_access_ptr(T.type_annotation("int8"), affine.data, i_0 * 19 + j_0, 19, 2), T.Cast("uint32", 0), 1, 1, 19, T.uint32(1040187392), 1, 0)
                    T.call_extern("void", "tvm_gemmini_end")