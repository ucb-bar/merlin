
import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import copy
import onnx
import numpy as np
import traceback
from contextlib import suppress
from custom_models.vitfly_models_july import ConvNet, LSTMNet, LSTMNetVIT, ViT, UNetConvLSTMNet
from torch.nn.utils.spectral_norm import remove_spectral_norm, SpectralNorm
from torch.export import export, ExportedProgram

# --- MLIR and Turbine Imports ---
try:
    from torch_mlir.fx import export_and_import as torch_mlir_export
    from torch_mlir import ir
    from torch_mlir.dialects.torch import register_dialect as register_torch_dialect
    print("Successfully imported torch-mlir.")
except ImportError:
    print("Warning: torch-mlir is not installed. 'torch-mlir' export path will be unavailable.")
    torch_mlir_export = None

try:
    import iree.turbine.aot as turbine_aot
    print("Successfully imported iree-turbine.")
except ImportError:
    print("Warning: iree-turbine is not installed. 'turbine_cpu' backend and 'iree-turbine' export path will be unavailable.")
    turbine_aot = None

# --- New Export Path Imports ---
try:
    import torch.export as torch_export
    print("Successfully imported torch.export.")
except ImportError:
    print("Warning: torch.export is not available. 'torch.export' path will be unavailable.")
    torch_export = None

try:
    import torch.onnx as torch_onnx
    print("Successfully imported torch.onnx.")
except ImportError:
    print("Warning: torch.onnx is not available. 'torch.onnx' path will be unavailable.")
    torch_onnx = None

# --- ONNX Runtime Imports ---
try:
    import onnxruntime
    from onnxruntime.quantization import static_quantize_runner, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod
    import onnxscript
    try:
        from onnxruntime.quantization import CalibrationMethod
    except ImportError:
        CalibrationMethod = None
except ImportError:
    raise ImportError(
        "onnxruntime is not installed. Please install it with "
        "'pip install onnx onnxruntime onnxscript'"
    )

# --- torch.ao.quantization (FX) Imports ---
try:
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    import torch.ao.quantization
    print("Successfully imported torch.ao.quantization (FX).")
except ImportError:
    print("Warning: torch.ao.quantization is not installed. FX Quantization paths will be unavailable.")
    torch.ao.quantization = None # Set to None to allow checks

# --- torchao (AO) Quantization Import ---
try:
    import torchao.quantization as torchao_quant
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig
    print("Successfully imported torchao.quantization.")
except ImportError:
    print("Warning: torchao.quantization is not installed. 'torchao' Quantization path will be unavailable.")
    torchao_quant = None
try:
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
        get_default_x86_inductor_quantization_config,
    )
    print("Successfully imported torchao.quantization.pt2e.")
except ImportError:
    print("Warning: torchao.quantization.pt2e is not installed. 'torchao pt2e' Quantization path will be unavailable.")

def print_separator(title: str):
    """Prints a formatted separator to the console."""
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80)

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

def try_ONNX_fp8_quantize(model, example_inputs, model_name, model_output_dir, results_list):
    """Attempts to apply ONNX FP8 quantization."""
    name = "ONNX quantization (FP8)"
    print_separator(f"Attempting: {name}")

    onnx_output_path = os.path.join(model_output_dir, f"{model_name}_fp32.onnx")
    quantized_onnx_output_path = os.path.join(model_output_dir, f"{model_name}_fp8_q.onnx")

    input_names = ["image", "velocity", "quaternion", "h_in", "c_in"]
    data_reader = MultiInputCalibrationDataReader(example_inputs[0], input_names, num_samples=10)

    try:
        static_quantize_runner(
            model_input=onnx_output_path,
            model_output=quantized_onnx_output_path,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QOperator,
            activation_type=QuantType.QFLOAT8E4M3FN,
            weight_type=QuantType.QFLOAT8E4M3FN,
            per_channel=False,
            calibrate_method=CalibrationMethod.Distribution,
            extra_options={
            'ActivationSymmetric': True,
            'WeightSymmetric': True
            }
        )
    except Exception as e:
        print(f"Error during ONNX Runtime quantization: {e}")

def try_torchao_quantize_weights_only(model_fp32, results_list):
    """Attempts to apply torchao INT8 weights only quantization."""
    name = "torchao.quantization (Int8WeightOnlyConfig)"
    print_separator(f"Attempting: {name}")

    if torchao_quant is None:
        print("FAILED: torchao.quantization is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torchao.quantization not installed'})
        return None

    try:
        config = torchao_quant.Int8WeightOnlyConfig()
        model_to_quant = copy.deepcopy(model_fp32).eval()
        
        print("Applying torchao quantization inplace...")
        torchao_quant.quantize_(model_to_quant, config)
        
        print("torchao quantization successful.")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
        return model_to_quant
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})
        return None

def try_torch_onnx_export(model, example_inputs, model_name, model_type, model_output_dir, results_list):
    """Attempts to export model to ONNX"""
    name = f"torch.onnx.export to ONNX ({model_type})"
    print_separator(f"Attempting: {name}")

    onnx_output_path = os.path.join(model_output_dir, f"{model_name}_{model_type}.onnx")
    input_names = ["image", "velocity", "quaternion", "h_in", "c_in"]
    output_names = ["output", "h_out", "c_out"]

    try:
        torch.onnx.export(
            model,
            example_inputs,
            onnx_output_path,
            input_names=input_names,
            output_names=output_names,
            opset_version=20, # <-- Set to 17 for better IREE compatibility
            dynamic_axes=None, # Explicitly disable dynamic axes for quantization
        )
        print(f"Successfully exported FP32 model to {onnx_output_path}")
    except Exception as e:
        print(f"Error during FP32 ONNX export: {e}")
        exit(1)

def try_torch_export(model, example_inputs, model_name, model_type, model_output_dir, results_list):
    """Exports the model with torch.export.export."""
    name = f"torch.export ({model_type})"
    print_separator(f"Attempting: {name}")
    
    output_path = os.path.join(model_output_dir, f"{model_name}_{model_type}.pt2")

    try:
        print(f"Exporting model to {output_path}...")
        exported_module = torch_export.export(model, example_inputs)
        torch_export.save(exported_module, output_path)
        print(f"Successfully exported to {output_path}")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})

def get_sample_inputs(model_name):
    batch = 5
    image = torch.randn(batch, 1, 60, 90)
    velocity = torch.randn(batch, 1)
    quaternion = torch.zeros(batch, 4)
    quaternion[:, 0] = 1.0
    h_in = torch.zeros(3, batch, 128)
    c_in = torch.zeros(3, batch, 128)

    if model_name == "LSTMNetVIT":
        return (image, velocity, quaternion, h_in, c_in)
    return (image, velocity, quaternion)

def get_forward_args_for_export(model_name, sample_inputs):
    if model_name == "LSTMNetVIT":
        return sample_inputs  # the LSTMNetVIT requires (self, image, velocity, quaternion, h_in, c_in)
    return (sample_inputs,) # the others use (self, X)

def get_model(model_name, checkpoint_path=None):
    """Loads a pretrained model based on the provided name."""
    print(f"--- Loading model: {model_name} ---")
    try:
        if model_name == "ConvNet":
            model = ConvNet().eval()
        elif model_name == "LSTMnet":
            model = LSTMNet().eval()
        elif model_name == "UNet":
            model = UNetConvLSTMNet().eval()
        elif model_name == "ViT":
            model = ViT().eval()
        elif model_name == "LSTMNetVIT":
            model = LSTMNetVIT().eval()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state_dict = state.get("state_dict", state)
        model.load_state_dict(state_dict, strict=True)
        print(f"Loaded {model_name} from {checkpoint_path}")
    else:
        if checkpoint_path:
            print(f"Checkpoint not found at {os.path.abspath(checkpoint_path)}; {model_name} created with random init.")
        else:
            print(f"{model_name} created with random init (no checkpoint loaded)")
    return model

def strip_spectral_norm(model):
    """Strips spectral norm from the model."""
    for module in model.modules():
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm):
                try:
                    remove_spectral_norm(module, hook.name)
                except Exception:
                    pass
    return model
    
def main():
    # similar to full_quantize_test, but changed to deep dive into a single model quantization process.    
    parser = argparse.ArgumentParser(
        description="Quantize a VitFly model and export to ONNX."
    )
    parser.add_argument(
        "--model", type=str, default="LSTMnet",
        choices=["ConvNet", "LSTMnet", "UNet", "ViT", "LSTMNetVIT"]
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(script_dir, "pretrained_models", f"{args.model}_model.pth")
    model_output_dir = os.path.join(script_dir, "pretrained_models", "quantize_test_export")

    print(f"Torch version: {torch.__version__}")
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Saving model artifacts to: {model_output_dir}") 

    model_fp32 = get_model(args.model) 
    if model_fp32 is None:
        print(f"FAILED: Could not load model {args.model}")
        return

    model_fp32 = strip_spectral_norm(model_fp32)
    sample_inputs = get_sample_inputs(args.model)
    forward_args = get_forward_args_for_export(args.model, sample_inputs)
    with torch.no_grad():
        _ = model_fp32(*forward_args)
    print("Forward pass OK (inputs match vitfly_models.forward).")

    # # --- Step 2: Run all export paths on FP32 model ---
    model_type_str="fp32"
    results = []
    try_torch_onnx_export(model_fp32, forward_args, args.model, model_type_str, model_output_dir, results)

    model_type_str="fp8"
    try_ONNX_fp8_quantize(model_fp32, forward_args, args.model, model_output_dir, results)

if __name__ == "__main__":
    main()