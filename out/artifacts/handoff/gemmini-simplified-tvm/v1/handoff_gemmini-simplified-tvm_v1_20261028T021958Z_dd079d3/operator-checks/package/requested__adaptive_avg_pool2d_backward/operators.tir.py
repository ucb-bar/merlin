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
    def stage_21_operator_adaptive_pool_backward(x: T.Buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(2)), "float32"), adaptive_input_gradient: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.workload": "fd5d0d7be546a57e", "op_pattern": 8, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        adaptive_gradient_delta = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(2)))
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(2), T.int64(2)):
            with T.block("adaptive_gradient_delta"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(x[v_i0, v_i1, v_i2, v_i3])
                T.writes(adaptive_gradient_delta[v_i0, v_i1, v_i2, v_i3])
                adaptive_gradient_delta[v_i0, v_i1, v_i2, v_i3] = x[v_i0, v_i1, v_i2, v_i3] / T.float32(2.0) / T.float32(2.0)
        for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4)):
            with T.block("adaptive_input_gradient_l1"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(adaptive_gradient_delta[v_i0, v_i1, v_i2 * T.int64(2) // T.int64(3):v_i2 * T.int64(2) // T.int64(3) + (v_i2 * T.int64(2) % T.int64(3) + T.int64(4)) // T.int64(3), v_i3 // T.int64(2)])
                T.writes(adaptive_input_gradient[v_i0, v_i1, v_i2, v_i3])
                for window_0, window_1 in T.grid((v_i2 * T.int64(2) % T.int64(3) + T.int64(4)) // T.int64(3), T.int64(1)):
                    with T.block("adaptive_input_gradient"):
                        v_i0_1 = T.axis.spatial((v_i0, v_i0 + T.int64(1)), v_i0)
                        v_i1_1 = T.axis.spatial((v_i1, v_i1 + T.int64(1)), v_i1)
                        v_i2_1 = T.axis.spatial((v_i2, v_i2 + T.int64(1)), v_i2)
                        v_i3_1 = T.axis.spatial((v_i3, v_i3 + T.int64(1)), v_i3)
                        v_window_0 = T.axis.reduce((v_i2 * T.int64(2) // T.int64(3), v_i2 * T.int64(2) // T.int64(3) + (v_i2 * T.int64(2) % T.int64(3) + T.int64(4)) // T.int64(3)), window_0 + v_i2 * T.int64(2) // T.int64(3))
                        v_window_1 = T.axis.reduce((v_i3 // T.int64(2), v_i3 // T.int64(2) + T.int64(1)), window_1 + v_i3 // T.int64(2))
                        T.reads(adaptive_gradient_delta[v_i0_1, v_i1_1, v_window_0, v_window_1])
                        T.writes(adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1])
                        with T.init():
                            adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1] = T.float32(0.0)
                        adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1] = adaptive_input_gradient[v_i0_1, v_i1_1, v_i2_1, v_i3_1] + adaptive_gradient_delta[v_i0_1, v_i1_1, v_window_0, v_window_1]