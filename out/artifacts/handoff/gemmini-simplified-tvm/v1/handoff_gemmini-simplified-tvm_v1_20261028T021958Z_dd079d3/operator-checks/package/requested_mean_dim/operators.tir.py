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
    def stage_13_operator_mean(x: T.Buffer((T.int64(2), T.int64(3)), "float32"), T_divide: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"gemmini.workload": "8de82f5be5c4c91e", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        x_red = T.alloc_buffer((T.int64(2),))
        for ax0, k1 in T.grid(T.int64(2), T.int64(3)):
            with T.block("x_red"):
                v_ax0, v_k1 = T.axis.remap("SR", [ax0, k1])
                T.reads(x[v_ax0, v_k1])
                T.writes(x_red[v_ax0])
                with T.init():
                    x_red[v_ax0] = T.float32(0.0)
                x_red[v_ax0] = x_red[v_ax0] + x[v_ax0, v_k1]
        for ax0 in range(T.int64(2)):
            with T.block("T_divide"):
                v_ax0 = T.axis.spatial(T.int64(2), ax0)
                T.reads(x_red[v_ax0])
                T.writes(T_divide[v_ax0])
                T_divide[v_ax0] = x_red[v_ax0] / T.float32(3.0)