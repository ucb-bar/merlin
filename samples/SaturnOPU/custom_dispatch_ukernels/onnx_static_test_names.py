import onnx

def main():
        
    model_path = "samples/SaturnOPU/custom_dispatch_ukernels/pretrained_models/quantize_test_export/LSTMnet_fp8.onnx"  # or LSTMnet_fp32.onnx
    model = onnx.load(model_path)

    fp8_op_types = ("MatMul", "Gemm", "QLinearMatMul", "QLinearGemm")
    QDQ_op_types = ("QuantizeLinear", "DequantizeLinear")

    to_exclude = [n.name for n in model.graph.node if n.op_type in QDQ_op_types]
    print("Node names to exclude (MatMul/Gemm and QLinear*):")
    for name in to_exclude:
        print(name)
    # One line for pasting into CLI (space-separated):
    print("\n--nodes_to_exclude", " ".join(to_exclude))



if __name__ == "__main__":
    main()