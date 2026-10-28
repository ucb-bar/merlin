# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def stage_4_operator_conv2d1(y: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32"), x: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)), "float32"), conv2d_cnhw: T.Buffer((T.int64(2), T.int64(1), T.int64(2), T.int64(2)), "float32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "ab74fedb3cbe57fa", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            pad_temp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)))
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(48), x.data, T.int64(48), conv2d_cnhw.data, T.int64(32))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", y.data, T.int64(48), y.data, T.int64(48), conv2d_cnhw.data, T.int64(32))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    for i0, i1, i2_0, i3_0 in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4)):
                        with T.block(""):
                            T.reads(y[T.int64(0), T.int64(0), i2_0, i3_0])
                            T.writes(pad_temp[T.int64(0), T.int64(0), i2_0, i3_0])
                            with T.block("pad_temp_o"):
                                T.reads(y[T.int64(0), T.int64(0), i2_0, i3_0])
                                T.writes()
                                T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), y.data, i2_0 * T.int64(4) + i3_0, T.int64(4), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(4), 0)
                            for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("pad_temp_local_gemmini_acc_o"):
                                    T.reads()
                                    T.writes(pad_temp[T.int64(0), T.int64(0), i2_0, i3_0])
                                    T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), pad_temp.data, i2_0 * T.int64(4) + i3_0, T.int64(4), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(4), 0)
                    T.call_extern("void", "tvm_gemmini_end")
                    for ff, nn, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1), T.int64(2), T.int64(2), T.int64(1), T.int64(2), T.int64(3)):
                        with T.block("conv2d_cnhw"):
                            T.reads(pad_temp[T.int64(0), T.int64(0), yy + ry, xx + rx], x[T.int64(0), ff, ry, rx])
                            T.writes(conv2d_cnhw[ff, T.int64(0), yy, xx])
                            if ry == T.int64(0) and rx == T.int64(0):
                                conv2d_cnhw[ff, T.int64(0), yy, xx] = T.float32(0.0)
                            conv2d_cnhw[ff, T.int64(0), yy, xx] = conv2d_cnhw[ff, T.int64(0), yy, xx] + pad_temp[T.int64(0), T.int64(0), yy + ry, xx + rx] * x[T.int64(0), ff, ry, rx]
                    T.call_extern("void", "tvm_gemmini_end")

    @T.prim_func(private=True)
    def stage_4_operator_conv2d_transpose(x: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)), "float32"), w: T.Buffer((T.int64(2), T.int64(1), T.int64(2), T.int64(2)), "float32"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(3), T.int64(4)), "float32")):
        T.func_attr({"gemmini.capability_id": "fb2281d666401792a7b5813387e82468de9832712cdcabe7d8ee47077638543d", "gemmini.tensorir": 1, "gemmini.workload": "2f7d628356448165", "op_pattern": 8, "tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            T.block_attr({"gemmini_regions_scheduled": 1})
            data_dilate = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)))
            data_pad = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(4), T.int64(5)))
            kernel_transform = T.alloc_buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(2)))
            gemmini_admission: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", w.data, T.int64(32), w.data, T.int64(32), compute.data, T.int64(48))
            if gemmini_admission != 0:
                T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                T.tvm_throw_last_error()
            else:
                gemmini_admission_1: T.int32 = T.call_extern("int32", "tvm_gemmini_validate_tensor_buffers", x.data, T.int64(48), x.data, T.int64(48), compute.data, T.int64(48))
                if gemmini_admission_1 != 0:
                    T.call_extern("void", "TVMAPISetLastError", "Gemmini tensor buffers rejected")
                    T.tvm_throw_last_error()
                else:
                    T.call_extern("void", "tvm_gemmini_begin", T.int64(1), T.int64(1), T.int64(1))
                    with T.block("data_dilate_o"):
                        T.reads(x[T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        T.writes(data_dilate[T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        for i0, i1, i2_0, i3_0 in T.grid(T.int64(1), T.int64(2), T.int64(2), T.int64(3)):
                            with T.block(""):
                                T.reads(x[T.int64(0), i1, i2_0, i3_0])
                                T.writes(data_dilate[T.int64(0), i1, i2_0, i3_0])
                                with T.block("data_dilate_o_1"):
                                    T.reads(x[T.int64(0), i1, i2_0, i3_0])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), x.data, i1 * T.int64(6) + i2_0 * T.int64(3) + i3_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                    with T.block("data_dilate_local_gemmini_acc_o"):
                                        T.reads()
                                        T.writes(data_dilate[T.int64(0), i1, i2_0, i3_0])
                                        T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), data_dilate.data, i1 * T.int64(6) + i2_0 * T.int64(3) + i3_0, T.int64(3), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                    with T.block("data_pad_pad_const_o"):
                        T.reads()
                        T.writes(data_pad[T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(4), T.int64(0):T.int64(5)])
                        for i0, i1, i2_0, i3_0 in T.grid(T.int64(1), T.int64(2), T.int64(4), T.int64(5)):
                            with T.block(""):
                                T.reads()
                                T.writes(data_pad[T.int64(0), i1, i2_0, i3_0])
                                with T.block("data_pad_pad_const_o_1"):
                                    T.reads()
                                    T.writes()
                                    with T.block("constant_source"):
                                        T.reads()
                                        T.writes()
                                        scalar = T.alloc_buffer(())
                                        scalar[()] = T.reinterpret("float32", T.uint32(0))
                                        T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), scalar.data, T.int64(0), T.int64(1), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(0), 1)
                                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                    with T.block("data_pad_local_gemmini_acc_o"):
                                        T.reads()
                                        T.writes(data_pad[T.int64(0), i1, i2_0, i3_0])
                                        T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), data_pad.data, i1 * T.int64(20) + i2_0 * T.int64(5) + i3_0, T.int64(5), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                    T.call_extern("void", "tvm_gemmini_end")
                    with T.block("data_pad_o"):
                        T.reads(data_dilate[T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2), T.int64(0):T.int64(3)])
                        T.writes(data_pad[T.int64(0), T.int64(0):T.int64(2), T.int64(1):T.int64(3), T.int64(1):T.int64(4)])
                        for i0, i1, i2_0, i3_0 in T.grid(T.int64(1), T.int64(2), T.int64(2), T.int64(3)):
                            with T.block(""):
                                T.reads(data_dilate[T.int64(0), i1, i2_0, i3_0])
                                T.writes(data_pad[T.int64(0), i1, i2_0 + T.int64(1), i3_0 + T.int64(1)])
                                with T.block("data_pad_o_1"):
                                    T.reads(data_dilate[T.int64(0), i1, i2_0, i3_0])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), data_dilate.data, i1 * T.int64(6) + i2_0 * T.int64(3) + i3_0, T.int64(3), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(3), 0)
                                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                    with T.block("data_pad_local_gemmini_acc_o_1"):
                                        T.reads()
                                        T.writes(data_pad[T.int64(0), i1, i2_0 + T.int64(1), i3_0 + T.int64(1)])
                                        T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), data_pad.data, i1 * T.int64(20) + i2_0 * T.int64(5) + i3_0 + T.int64(6), T.int64(5), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(5), 0)
                    with T.block("kernel_transform_o"):
                        T.reads(w[T.int64(0):T.int64(2), T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                        T.writes(kernel_transform[T.int64(0), T.int64(0):T.int64(2), T.int64(0):T.int64(2), T.int64(0):T.int64(2)])
                        for o, i, h_0, w_0 in T.grid(T.int64(1), T.int64(2), T.int64(2), T.int64(2)):
                            with T.block(""):
                                T.reads(w[i, T.int64(0), T.int64(1) - h_0, T.int64(1) - w_0])
                                T.writes(kernel_transform[T.int64(0), i, h_0, w_0])
                                with T.block("kernel_transform_o_1"):
                                    T.reads(w[i, T.int64(0), T.int64(1) - h_0, T.int64(1) - w_0])
                                    T.writes()
                                    T.call_extern("void", "tvm_gemmini_dma_acc_write", T.tvm_access_ptr(T.type_annotation("float32"), w.data, i * T.int64(4) + T.int64(3) - w_0 - h_0 * T.int64(2), T.int64(2), 1), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(2), 0)
                                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                                    with T.block("kernel_transform_local_gemmini_acc_o"):
                                        T.reads()
                                        T.writes(kernel_transform[T.int64(0), i, h_0, w_0])
                                        T.call_extern("void", "tvm_gemmini_dma_out", T.tvm_access_ptr(T.type_annotation("float32"), kernel_transform.data, i * T.int64(4) + h_0 * T.int64(2) + w_0, T.int64(2), 2), T.Cast("uint32", T.int64(0)), T.uint32(1), T.uint32(1), T.int64(2), 0)
                    T.call_extern("void", "tvm_gemmini_end")
                    for b, c, h, w_1, dc, dh, dw in T.grid(T.int64(1), T.int64(1), T.int64(3), T.int64(4), T.int64(2), T.int64(2), T.int64(2)):
                        with T.block("compute"):
                            T.reads(data_pad[T.int64(0), dc, h + dh, w_1 + dw], kernel_transform[T.int64(0), dc, dh, dw])
                            T.writes(compute[T.int64(0), T.int64(0), h, w_1])
                            if dc == T.int64(0) and dh == T.int64(0) and dw == T.int64(0):
                                compute[T.int64(0), T.int64(0), h, w_1] = T.float32(0.0)
                            compute[T.int64(0), T.int64(0), h, w_1] = compute[T.int64(0), T.int64(0), h, w_1] + data_pad[T.int64(0), dc, h + dh, w_1 + dw] * kernel_transform[T.int64(0), dc, dh, dw]
                    T.call_extern("void", "tvm_gemmini_end")

    @T.prim_func(private=True)
    def stage_4_operator_sum(x: T.Buffer((T.int64(1), T.int64(2), T.int64(2), T.int64(3)), "float32"), x_red: T.Buffer((T.int64(2),), "float32")):
        T.func_attr({"gemmini.workload": "304321ba7c8ac45d", "op_pattern": 3, "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, k0, k2, k3 in T.grid(T.int64(2), T.int64(1), T.int64(2), T.int64(3)):
            with T.block("x_red"):
                v_ax0, v_k0, v_k2, v_k3 = T.axis.remap("SRRR", [ax0, k0, k2, k3])
                T.reads(x[v_k0, v_ax0, v_k2, v_k3])
                T.writes(x_red[v_ax0])
                with T.init():
                    x_red[v_ax0] = T.float32(0.0)
                x_red[v_ax0] = x_red[v_ax0] + x[v_k0, v_ax0, v_k2, v_k3]

    @R.function
    def main(x: R.Tensor((1, 2, 2, 3), dtype="float32"), y: R.Tensor((1, 1, 3, 4), dtype="float32"), w: R.Tensor((2, 1, 2, 2), dtype="float32")) -> R.Tuple(R.Tensor((1, 1, 3, 4), dtype="float32"), R.Tensor((2, 1, 2, 2), dtype="float32"), R.Tensor((2,), dtype="float32")):
        cls = Module
        with R.dataflow():
            lv1 = R.call_tir(cls.stage_4_operator_conv2d_transpose, (x, w), out_sinfo=R.Tensor((1, 1, 3, 4), dtype="float32"))
            lv2 = R.call_tir(cls.stage_4_operator_conv2d1, (y, x), out_sinfo=R.Tensor((2, 1, 2, 2), dtype="float32"))
            lv3 = R.call_tir(cls.stage_4_operator_sum, (x,), out_sinfo=R.Tensor((2,), dtype="float32"))
            gv: R.Tuple(R.Tensor((1, 1, 3, 4), dtype="float32"), R.Tensor((2, 1, 2, 2), dtype="float32"), R.Tensor((2,), dtype="float32")) = lv1, lv2, lv3
            R.output(gv)
        return gv