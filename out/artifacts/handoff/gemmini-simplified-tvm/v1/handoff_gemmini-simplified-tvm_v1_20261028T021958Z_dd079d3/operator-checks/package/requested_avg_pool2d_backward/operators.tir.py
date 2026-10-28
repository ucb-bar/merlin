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
    def stage_20_operator_avg_pool2d_backward(x: T.Buffer((T.int64(1), T.int64(1), T.int64(2), T.int64(3)), "float32"), y: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32"), T_pool_grad: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.workload": "0624a03000d003a1", "op_pattern": 4, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2, ax3, wh, ww in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4), T.int64(2), T.int64(2)):
            with T.block("T_pool_grad"):
                v_ax0, v_ax1, v_ax2, v_ax3, v_wh, v_ww = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, wh, ww])
                T.reads(x[v_ax0, v_ax1, v_ax2 - v_wh, v_ax3 - v_ww])
                T.writes(T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3])
                with T.init():
                    T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(0.0)
                T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] = T_pool_grad[v_ax0, v_ax1, v_ax2, v_ax3] + T.if_then_else(T.Select(v_ax2 < T.int64(2), T.int64(0), v_ax2 - T.int64(1)) <= v_ax2 - v_wh and v_ax2 - v_wh < T.int64(2) and T.Select(v_ax3 < T.int64(2), T.int64(0), v_ax3 - T.int64(1)) <= v_ax3 - v_ww and v_ax3 - v_ww < T.int64(3), x[v_ax0, v_ax1, v_ax2 - v_wh, v_ax3 - v_ww] / T.float32(4.0), T.float32(0.0))