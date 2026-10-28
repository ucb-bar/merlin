# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((2, 3), dtype="int8"), y: R.Tensor((2, 3), dtype="int8"), w: R.Tensor((3,), dtype="int8")) -> R.Tensor((2, 3), dtype="int8"):
        return x