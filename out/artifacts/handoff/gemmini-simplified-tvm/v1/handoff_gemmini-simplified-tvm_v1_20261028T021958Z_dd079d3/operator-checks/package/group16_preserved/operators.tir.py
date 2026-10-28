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
    def stage_70_operator_kernel(a: T.Buffer((3, 5), "int8"), b: T.Buffer((5, 33), "int8"), scales: T.Buffer((33,), "float32"), compute: T.Buffer((3, 33), "float32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.k": 5, "gemmini.m": 3, "gemmini.n": 33, "gemmini.tensorir": 1, "gemmini.workload": "1904746a4ea817fe", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini.sketch": "ij.3x11x5.os.h0.p0.eroot.stile.tnone"})
            compute_1 = T.alloc_buffer((3, 33), "int32")
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", scales.data, T.int64(132), scales.data, T.int64(132), compute.data, T.int64(396))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", b.data, T.int64(165), b.data, T.int64(165), compute.data, T.int64(396))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    gemmini_admission_2: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", a.data, T.int64(15), a.data, T.int64(15), compute.data, T.int64(396))
                    if gemmini_admission_2 != 0:
                        T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                        T.tvm_throw_last_error()
                    else:
                        T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                        for i_0_0, j_0_0 in T.grid(1, 3):
                            with T.block(""):
                                T.reads(a[0:3, 0:5], b[0:5, j_0_0 * 11:j_0_0 * 11 + 11])
                                T.writes(compute_1[0:3, j_0_0 * 11:j_0_0 * 11 + 11])
                                for i_0_1_init, j_0_1_init in T.grid(1, 1):
                                    with T.block("compute_init_o"):
                                        T.reads()
                                        T.writes()
                                        T.call_extern("void", "tvm_gemmini_zero_acc", T.Cast("uint32", 0), T.uint32(3), T.uint32(11))
                                for k_0_0 in range(1):
                                    with T.block(""):
                                        T.reads(a[0:3, 0:5], b[0:5, j_0_0 * 11:j_0_0 * 11 + 11])
                                        T.writes()
                                        for ax0_0, ax1_0 in T.grid(1, 1):
                                            with T.block("a_local_gemmini_spad_o"):
                                                T.reads(a[0:3, 0:5])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_a", T.tvm_access_ptr(T.type_annotation("int8"), a.data, 0, T.int64(15), 1), T.Cast("uint32", 0), T.uint32(3), T.uint32(5), 5, 0)
                                        for ax0_0, ax1_0 in T.grid(1, 1):
                                            with T.block("b_local_gemmini_spad_o"):
                                                T.reads(b[0:5, j_0_0 * 11:j_0_0 * 11 + 11])
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_dma_b", T.tvm_access_ptr(T.type_annotation("int8"), b.data, j_0_0 * 11, T.int64(165), 1), T.Cast("uint32", 16), T.uint32(5), T.uint32(11), 33, 0)
                                        for k_0_1, i_0_1, j_0_1 in T.grid(1, 1, 1):
                                            with T.block("compute_update_o"):
                                                T.reads()
                                                T.writes()
                                                T.call_extern("void", "tvm_gemmini_compute", T.Cast("uint32", 0), T.Cast("uint32", 16), T.Cast("uint32", 0), T.uint32(3), T.uint32(11), T.uint32(5), 1)
                                for ax0_0, ax1_0 in T.grid(1, 1):
                                    with T.block("compute_local_gemmini_acc_o"):
                                        T.reads()
                                        T.writes(compute_1[0:3, j_0_0 * 11:j_0_0 * 11 + 11])
                                        T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("int32"), compute_1.data, j_0_0 * 11, T.int64(99), 2), T.Cast("uint32", 0), T.uint32(3), T.uint32(11), 33, 0)
                        T.call_extern("void", "tvm_gemmini_end")
                        for i, j in T.grid(3, 33):
                            with T.block("compute_1"):
                                T.reads(compute_1[i, j], scales[j])
                                T.writes(compute[i, j])
                                compute[i, j] = T.Cast("float32", compute_1[i, j]) * scales[j]
                        T.call_extern("void", "tvm_gemmini_end")