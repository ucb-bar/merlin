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
    def stage_69_operator_kernel(accumulator: T.Buffer((3, 17), "int32"), readout: T.Buffer((3, 17), "int8")):
        T.func_attr({"gemmini.workload": "693993567a6065d1", "op_pattern": 0, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(3, 17):
            with T.block("readout"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(accumulator[v_i, v_j])
                T.writes(readout[v_i, v_j])
                readout[v_i, v_j] = T.Cast("int8", T.min(T.max(T.nearbyint(T.max(T.Cast("float32", accumulator[v_i, v_j]) * T.float32(0.0031129), T.float32(0.0))), T.float32(-128.0)), T.float32(127.0)))