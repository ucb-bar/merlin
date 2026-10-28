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
    def stage_49_operator_mean(x: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((), "float32")):
        T.func_attr({"gemmini.workload": "45df7bfae6b7fb5d", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        x_red = T.alloc_buffer(())
        for k0, k1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("x_red"):
                v_k0, v_k1 = T.axis.remap("RR", [k0, k1])
                T.reads(x[v_k0, v_k1])
                T.writes(x_red[()])
                with T.init():
                    x_red[()] = T.float32(0.0)
                x_red[()] = x_red[()] + x[v_k0, v_k1]
        with T.block("T_divide"):
            vi = T.axis.spatial(1, T.int64(0))
            T.reads(x_red[()])
            T.writes(T_divide[()])
            T_divide[()] = x_red[()] / T.float32(6.0)