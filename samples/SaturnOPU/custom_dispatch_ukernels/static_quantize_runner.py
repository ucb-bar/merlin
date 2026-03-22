import argparse
import json
import os
import torch
import numpy as np
import onnx

import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, StaticQuantConfig, quantize
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod


class MultiInputCalibrationDataReader(CalibrationDataReader):
    """
    Generates calibration data for models with multiple inputs.
    """
    def __init__(self, example_inputs_tuple, input_names, num_samples=10):
        self.input_names = input_names
        # Create a list of shapes from the example inputs
        self.input_shapes = [tuple(tensor.shape) for tensor in example_inputs_tuple]
        self.num_samples = num_samples

        print(f"\nInitializing calibration data reader with {len(input_names)} inputs:")
        for name, shape in zip(self.input_names, self.input_shapes):
            print(f"  Input '{name}' with shape={shape}")
        # This will be our iterator
        self.data_generator = self._data_generator()

    def _data_generator(self):
        """A generator function that yields a dictionary of {input_name: data}."""
        for _ in range(self.num_samples):
            # Create a dictionary of {input_name: numpy_array}
            sample = {}
            for name, shape in zip(self.input_names, self.input_shapes):
                sample[name] = np.random.randn(*shape).astype(np.float32)
            yield sample

    def get_next(self):
        """Returns the next sample from the generator."""
        return next(self.data_generator, None)
    
    def rewind(self):
        """Resets the iterator to the beginning."""
        self.data_generator = self._data_generator()


def parse_arguments():
    parser = argparse.ArgumentParser(description="The arguments for static quantization")
    parser.add_argument("-i", "--input_model_path", required=True, help="Path to the input onnx model")
    parser.add_argument(
        "-o", "--output_quantized_model_path", required=True, help="Path to the output quantized onnx model"
    )
    parser.add_argument(
        "--activation_type",
        choices=["qint8", "quint8", "qint16", "quint16", "qint4", "quint4", "qfloat8e4m3fn"],
        default="quint8",
        help="Activation quantization type used",
    )
    parser.add_argument(
        "--weight_type",
        choices=["qint8", "quint8", "qint16", "quint16", "qint4", "quint4", "qfloat8e4m3fn"],
        default="qint8",
        help="Weight quantization type used",
    )
    parser.add_argument("--enable_subgraph", action="store_true", help="If set, subgraph will be quantized.")
    parser.add_argument(
        "--force_quantize_no_input_check",
        action="store_true",
        help="By default, some latent operators like maxpool, transpose, do not quantize if their input is not"
        " quantized already. Setting to True to force such operator always quantize input and so generate"
        " quantized output. Also the True behavior could be disabled per node using the nodes_to_exclude.",
    )
    parser.add_argument(
        "--matmul_const_b_only",
        action="store_true",
        help="If set, only MatMul with const B will be quantized.",
    )
    parser.add_argument(
        "--add_qdq_pair_to_weight",
        action="store_true",
        help="If set, it remains floating-point weight and inserts both QuantizeLinear/DeQuantizeLinear"
        " nodes to weight.",
    )
    parser.add_argument(
        "--dedicated_qdq_pair",
        action="store_true",
        help="If set, it will create identical and dedicated QDQ pair for each node.",
    )
    parser.add_argument(
        "--op_types_to_exclude_output_quantization",
        nargs="+",
        default=[],
        help="If any op type is specified, it won't quantize the output of ops with this specific op types.",
    )
    parser.add_argument(
        "--calibration_method",
        default="minmax",
        choices=["minmax", "entropy", "percentile", "distribution"],
        help="Calibration method used",
    )
    parser.add_argument("--quant_format", default="qdq", choices=["qdq", "qoperator"], help="Quantization format used")
    parser.add_argument(
        "--calib_tensor_range_symmetric",
        action="store_true",
        help="If enabled, the final range of tensor during calibration will be explicitly"
        " set to symmetric to central point 0",
    )
    # TODO: --calib_strided_minmax"
    # TODO: --calib_moving_average_constant"
    # TODO: --calib_max_intermediate_outputs"
    parser.add_argument(
        "--calib_moving_average",
        action="store_true",
        help="If enabled, the moving average of"
        " the minimum and maximum values will be computed when the calibration method selected is MinMax.",
    )
    parser.add_argument(
        "--disable_quantize_bias",
        action="store_true",
        help="Whether to quantize floating-point biases by solely inserting a DeQuantizeLinear node"
        " If not set, it remains floating-point bias and does not insert any quantization nodes"
        " associated with biases.",
    )

    # TODO: Add arguments related to Smooth Quant

    parser.add_argument(
        "--use_qdq_contrib_ops",
        action="store_true",
        help="If set, the inserted QuantizeLinear and DequantizeLinear ops will have the com.microsoft domain,"
        " which forces use of ONNX Runtime's QuantizeLinear and DequantizeLinear contrib op implementations.",
    )
    parser.add_argument(
        "--minimum_real_range",
        type=float,
        default=0.0001,
        help="If set to a floating-point value, the calculation of the quantization parameters"
        " (i.e., scale and zero point) will enforce a minimum range between rmin and rmax. If (rmax-rmin)"
        " is less than the specified minimum range, rmax will be set to rmin + MinimumRealRange. This is"
        " necessary for EPs like QNN that require a minimum floating-point range when determining "
        " quantization parameters.",
    )
    parser.add_argument(
        "--qdq_keep_removable_activations",
        action="store_true",
        help="If set, removable activations (e.g., Clip or Relu) will not be removed,"
        " and will be explicitly represented in the QDQ model.",
    )
    parser.add_argument(
        "--qdq_disable_weight_adjust_for_int32_bias",
        action="store_true",
        help="If set, QDQ quantizer will not adjust the weight's scale when the bias"
        " has a scale (input_scale * weight_scale) that is too small.",
    )
    parser.add_argument("--per_channel", action="store_true", help="Whether using per-channel quantization")
    parser.add_argument(
        "--nodes_to_quantize",
        nargs="+",
        default=None,
        help="List of nodes names to quantize. When this list is not None only the nodes in this list are quantized.",
    )
    parser.add_argument(
        "--nodes_to_exclude",
        nargs="+",
        default=None,
        help="List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.",
    )
    parser.add_argument(
        "--op_per_channel_axis",
        nargs=2,
        action="append",
        metavar=("OP_TYPE", "PER_CHANNEL_AXIS"),
        default=[],
        help="Set channel axis for specific op type, for example: --op_per_channel_axis MatMul 1, and it's"
        " effective only when per channel quantization is supported and per_channel is True. If specific"
        " op type supports per channel quantization but not explicitly specified with channel axis,"
        " default channel axis will be used.",
    )
    parser.add_argument("--tensor_quant_overrides", help="Set the json file for tensor quantization overrides.")
    return parser.parse_args()


def get_tensor_quant_overrides(file):
    # TODO: Enhance the function to handle more real cases of json file
    if not file:
        return {}
    with open(file) as f:
        quant_override_dict = json.load(f)
    for tensor in quant_override_dict:
        for enc_dict in quant_override_dict[tensor]:
            enc_dict["scale"] = np.array(enc_dict["scale"], dtype=np.float32)
            enc_dict["zero_point"] = np.array(enc_dict["zero_point"])
    return quant_override_dict


def main():
    input_names = ["image", "velocity", "quaternion", "h_in", "c_in"]
    forward_args = ((torch.randn(5, 1, 60, 90), torch.randn(5, 1), torch.cat([torch.ones(5, 1), torch.zeros(5, 3)], dim=1)),)
    args = parse_arguments()
    data_reader = MultiInputCalibrationDataReader(forward_args[0], input_names, num_samples=10)
    arg2quant_type = {
        "qint8": QuantType.QInt8,
        "quint8": QuantType.QUInt8,
        "qint16": QuantType.QInt16,
        "quint16": QuantType.QUInt16,
        "qint4": QuantType.QInt4,
        "quint4": QuantType.QUInt4,
        "qfloat8e4m3fn": QuantType.QFLOAT8E4M3FN,
    }
    activation_type = arg2quant_type[args.activation_type]
    weight_type = arg2quant_type[args.weight_type]
    qdq_op_type_per_channel_support_to_axis = dict(args.op_per_channel_axis)
    extra_options = {
        "EnableSubgraph": args.enable_subgraph,
        "ForceQuantizeNoInputCheck": args.force_quantize_no_input_check,
        "MatMulConstBOnly": args.matmul_const_b_only,
        "AddQDQPairToWeight": args.add_qdq_pair_to_weight,
        "OpTypesToExcludeOutputQuantization": args.op_types_to_exclude_output_quantization,
        "DedicatedQDQPair": args.dedicated_qdq_pair,
        "QDQOpTypePerChannelSupportToAxis": qdq_op_type_per_channel_support_to_axis,
        "CalibTensorRangeSymmetric": args.calib_tensor_range_symmetric,
        "CalibMovingAverage": args.calib_moving_average,
        "QuantizeBias": not args.disable_quantize_bias,
        "UseQDQContribOps": args.use_qdq_contrib_ops,
        "MinimumRealRange": args.minimum_real_range,
        "QDQKeepRemovableActivations": args.qdq_keep_removable_activations,
        "QDQDisableWeightAdjustForInt32Bias": args.qdq_disable_weight_adjust_for_int32_bias,
        # Load json file for encoding override
        "TensorQuantOverrides": get_tensor_quant_overrides(args.tensor_quant_overrides),
    }
    arg2calib_method = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
        "distribution": CalibrationMethod.Distribution,
    }
    arg2quant_format = {
        "qdq": QuantFormat.QDQ,
        "qoperator": QuantFormat.QOperator,
    }
    sqc = StaticQuantConfig(
        calibration_data_reader=data_reader,
        calibrate_method=arg2calib_method[args.calibration_method],
        quant_format=arg2quant_format[args.quant_format],
        activation_type=activation_type,
        weight_type=weight_type,
        op_types_to_quantize=None,
        # op_types_to_quantize=["MatMul", "Gemm"],
        nodes_to_quantize=args.nodes_to_quantize,
        nodes_to_exclude=args.nodes_to_exclude,
        per_channel=args.per_channel,
        reduce_range=False,
        use_external_data_format=False,
        calibration_providers=None,  # Use CPUExecutionProvider
        extra_options=extra_options,
    )
    quantize(model_input=args.input_model_path, model_output=args.output_quantized_model_path, quant_config=sqc)


if __name__ == "__main__":
    main()
