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
    def stage_22_operator_col2im(x: T.Buffer((T.int64(1), T.int64(4), T.int64(6)), "float32"), overlap_sum: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.workload": "66bae785669c939d", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3, offset_0, offset_1 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4), T.int64(2), T.int64(2)):
            with T.block("overlap_sum"):
                v_i0, v_i1, v_i2, v_i3, v_offset_0, v_offset_1 = T.axis.remap("SSSSRR", [i0, i1, i2, i3, offset_0, offset_1])
                T.reads(x[v_i0, v_i1 * T.int64(4) + v_offset_0 * T.int64(2) + v_offset_1, v_i2 * T.int64(3) + v_i3 - v_offset_1 - v_offset_0 * T.int64(3)])
                T.writes(overlap_sum[v_i0, v_i1, v_i2, v_i3])
                with T.init():
                    overlap_sum[v_i0, v_i1, v_i2, v_i3] = T.float32(0.0)
                overlap_sum[v_i0, v_i1, v_i2, v_i3] = overlap_sum[v_i0, v_i1, v_i2, v_i3] + T.if_then_else(T.int64(0) <= v_i2 - v_offset_0 and v_i2 - v_offset_0 < T.int64(2) and T.int64(0) <= v_i3 - v_offset_1 and v_i3 - v_offset_1 < T.int64(3), x[v_i0, v_i1 * T.int64(4) + v_offset_0 * T.int64(2) + v_offset_1, v_i2 * T.int64(3) + v_i3 - v_offset_1 - v_offset_0 * T.int64(3)], T.float32(0.0))