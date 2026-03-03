import torch
import torchvision.models as models
import argparse
import os
import numpy as np
import onnx

from torchvision.models.squeezenet import SqueezeNet1_0_Weights
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.models.resnet import ResNet50_Weights
from custom_models.vitfly_models import LSTMNetVIT
from custom_models.FC import SimpleFCModel

# --- ONNX Runtime Imports ---
try:
    import onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
except ImportError:
    raise ImportError(
        "onnxruntime is not installed. Please install it with "
        "'pip install onnx onnxruntime'"
    )

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Export a PyTorch FP32 model to ONNX, then perform post-training"
                " static quantization using the ONNX Runtime."
)
parser.add_argument(
    "--output_onnx_file",
    type=str,
    default="model_quantized_ort.onnx",
    help="Path to output the final quantized ONNX file.",
)
parser.add_argument(
    "--model",
    type=str,
    choices=[
        "mobilenet",
        "squeezenet",
        "alexnet",
        "resnet",
        "mobilenetv3small",
        "mobilenetv3large",
        "transformer", 
        "lstmvit",
        "fc",
    ],
    default="mobilenet",
    help="Choose the model to export and quantize.",
)
args = parser.parse_args()

# --- MODIFIED SECTION: Directory and Path Management ---

# 1. Define the model-specific output directory name
output_dir = f"compilation_phases_{args.model}"

# 2. Create the directory if it doesn't exist
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory set to: ./{output_dir}")
except OSError as e:
    print(f"Error creating directory {output_dir}: {e}")
    exit(1)

# 3. Re-define output paths to be inside the new directory
# Use os.path.basename to get just the filename from the argument
quantized_basename = os.path.basename(args.output_onnx_file)
quantized_output_path = os.path.join(output_dir, quantized_basename)

# Derive the FP32 path from the new, full quantized path
fp32_output_path = f"{os.path.splitext(quantized_output_path)[0]}_fp32.onnx"

# --- END MODIFIED SECTION ---


print(f"Torch version: {torch.__version__}")
print(f"Selected Model: {args.model}")

# --- Step 1: Model Loading ---
model = None
# Default input for torchvision models
sample_inputs = (torch.randn(1, 3, 224, 224),)
input_names = ["input"]
output_names = ["output"]

if args.model == "squeezenet":
    model = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).eval()
elif args.model == "alexnet":
    model = models.alexnet(weights=AlexNet_Weights.DEFAULT).eval()
elif args.model == "resnet":
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
elif args.model == "mobilenetv3small":
    model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
elif args.model == "mobilenetv3large":
    model = models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
elif args.model == "transformer":
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, d_model=64, nhead=8, dim_feedforward=128, seq_len=16):
            super(SimpleTransformer, self).__init__()
            self.pos_embedding = torch.nn.Parameter(torch.randn(1, seq_len, d_model))
            encoder_layer = torch.nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
            self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        def forward(self, x):
            x = x + self.pos_embedding  # Add positional encoding
            return self.encoder(x)
    
    model = SimpleTransformer().eval()
    # Override sample_inputs for this specific model
    sample_inputs = (torch.randn(1, 16, 64),)  # (batch_size, seq_len, d_model)

elif args.model == "fc":
    model = SimpleFCModel(input_size=1024, hidden_size=128, output_size=10).eval()
    sample_inputs = (torch.randn(16, 1024),)  # (batch_size, input_size)
    
elif args.model == "lstmvit":
    model = LSTMNetVIT().eval()
    
    batch_size = 1 # Static batch size for quantization
    
    num_layers = model.lstm_num_layers
    hidden_size = model.lstm_hidden_size

    # 1. Define all sample inputs
    img_in = torch.randn(batch_size, 1, 60, 90)
    vel_in = torch.randn(batch_size, 1)
    quat_in = torch.randn(batch_size, 4)
    h_in = torch.randn(num_layers, batch_size, hidden_size)
    c_in = torch.randn(num_layers, batch_size, hidden_size)
    
    # This tuple will be unpacked by the exporter
    sample_inputs = (img_in, vel_in, quat_in, h_in, c_in)
    
    # 2. Define input names (must match forward() args)
    input_names = ["image", "velocity", "quaternion", "h_in", "c_in"]
    
    # 3. Define output names (must match forward() return)
    output_names = ["output", "h_out", "c_out"]

elif args.model == "mobilenet":  # Default (MobileNetV2)
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
else:
    raise ValueError(f"Unsupported model: {args.model}")

print(f"Loaded {args.model} model.")
print(f"Using sample input shape: {sample_inputs[0].shape}")


# --- Step 2: Export the FP32 model to ONNX ---
# This print statement will now show the full path
print(f"Exporting FP32 model to ONNX file: {fp32_output_path}...")
try:
    torch.onnx.export(
        model,
        sample_inputs,  # <-- Use the sample_inputs tuple directly
        fp32_output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=20, # <-- Set to 17 for better IREE compatibility
        dynamic_axes=None, # Explicitly disable dynamic axes for quantization
    )
    print(f"Successfully exported FP32 model to {fp32_output_path}")
except Exception as e:
    print(f"Error during FP32 ONNX export: {e}")
    exit(1)

# --- Step 3: Post-Training Static Quantization with ONNX Runtime ---

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

print("\nStarting post-training static quantization with ONNX Runtime...")
calibration_data_reader = MultiInputCalibrationDataReader(
    example_inputs_tuple=sample_inputs,  # <-- Pass the *entire tuple* of inputs
    input_names=input_names,              # <-- Pass the *entire list* of names
    num_samples=10 
)

# --- Find nodes to exclude (as requested) ---
print(f"Loading {fp32_output_path} to find nodes for exclusion...")
model_for_exclusion = onnx.load(fp32_output_path)
graph = model_for_exclusion.graph

nodes_to_exclude = []
op_types_to_exclude = {'Add', 'Gemm'} # Define the op_types you want to skip

for node in graph.node:
    if node.op_type in op_types_to_exclude:
        nodes_to_exclude.append(node.name)

print(f"Found {len(nodes_to_exclude)} nodes to exclude (if enabled):")
# print(nodes_to_exclude) # Uncomment to debug

try:
    quantize_static(
        model_input=fp32_output_path,
        model_output=quantized_output_path,
        calibration_data_reader=calibration_data_reader,
        
        # nodes_to_exclude=nodes_to_exclude,
        
        quant_format=onnxruntime.quantization.QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False, 
        extra_options={
        'ActivationSymmetric': True,
        'WeightSymmetric': True
        }
    )
    print(f"Successfully generated quantized ONNX model: {quantized_output_path}")
    print("\nNext step: Verify the ONNX file with a tool like Netron to see the Q/DQ nodes,")
    print(f"then import into MLIR using 'iree-import-onnx {quantized_output_path}'.")

except Exception as e:
    print(f"Error during ONNX Runtime quantization: {e}")