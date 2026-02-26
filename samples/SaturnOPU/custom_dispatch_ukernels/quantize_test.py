import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os
import copy
import traceback
from contextlib import suppress
from custom_models.vitfly_models_july import ConvNet, LSTMNet, LSTMNetVIT, ViT, UNetConvLSTMNet
from export_models_torch import print_separator
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

# --- torch.ao.quantization (FX) Imports ---
try:
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization.qconfig_mapping import QConfigMapping
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

def try_torchao_quantize(model_fp32, results_list): # for other reference: https://docs.pytorch.org/ao/main/eager_tutorials/first_quantization_example.html
    """Attempts to apply torchao INT8 dynamic quantization."""
    name = "torchao.quantization (Int8DynamicActivationInt8Weight)"
    print_separator(f"Attempting: {name}")

    if torchao_quant is None:
        print("FAILED: torchao.quantization is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torchao.quantization not installed'})
        return None

    try:
        config = torchao_quant.Int8DynamicActivationInt8WeightConfig()
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

    output_path = os.path.join(model_output_dir, f"{model_name}_{model_type}_torch_onnx.onnx")
    
    if torch_onnx is None:
        print("FAILED: torch.onnx is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torch.onnx not installed'})
        return

    try:
        model.eval()
        print(f"Exporting model to {output_path}...")
        torch_onnx.export(
            model,
            example_inputs,
            output_path,
            input_names=["input"],
            output_names=["output"],
            dynamo=False,
            opset_version=21
        )
        
        print(f"Successfully exported to {output_path}")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})

def try_turbine_aot_export(model, example_inputs, model_name, model_type, model_output_dir, results_list):
    """Exports the model to MLIR using the IREE Turbine AOT exporter."""
    name = f"iree.turbine.aot.export ({model_type})"
    print_separator(f"Attempting: {name}")
    
    output_path = os.path.join(model_output_dir, f"{model_name}_{model_type}_turbine_aot.mlir")

    if turbine_aot is None:
        print("FAILED: iree-turbine is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'iree-turbine not installed'})
        return

    try:
        model.eval()
        # Allow torch.typename in the graph so spectral_norm's setattr's typename doesn't cause Dynamo to raise.
        # If downstream iree fails on the resulting graph then i need to strip spectral norm or something
        # did not work :skull: torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor Explanation: torch.* ops that return a non-Tensor cannot be traced into the Dynamo FX graph output
        # torch.compiler.allow_in_graph(torch.typename)

        # turbine_aot.externalize_module_parameters(model)

        print(f"Exporting model to {output_path}...")

        # with torch.no_grad():
        #     exported = export( # might have to do exported: ExportedProgram = export(model, example_inputs, strict=False)
        #         model,
        #         example_inputs,
        #         strict=False
        #     )
    
        exported_module = turbine_aot.export(model, *example_inputs)
        # exported_module = turbine_aot.export(exported)
        mlir_str = str(exported_module.mlir_module)
        
        with open(output_path, "w") as f:
            f.write(mlir_str)
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

def main():
    parser = argparse.ArgumentParser(
        description="Quantize a VitFly model and export to MLIR."
    )
    parser.add_argument(
        "--model", type=str, default="LSTMnet",
        choices=["ConvNet", "LSTMnet", "UNet", "ViT", "LSTMNetVIT"]
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path of pretrained model checkpoint (default: script-relative samples/pretrained_models/LSTMnet.pth)."
    )
    parser.add_argument(
        "--output_dir", type=str, default="quantize_test_export",
        help="Directory to save the exported files and report."
    )
    parser.add_argument(
        "--quant",
        type=str,
        default="torchao_dynamic_int8",
        choices=["none", "torchao_dynamic_int8", "torchao_int8_weight_only"],
        help="Quantization mode to apply before export.",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.normpath(os.path.join(script_dir, "..", "pretrained_models", "LSTMnet_model.pth"))
    model_output_dir = os.path.normpath(os.path.join(script_dir, "..", "pretrained_models", args.output_dir))

    print(f"Torch version: {torch.__version__}")
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Saving model artifacts to: {model_output_dir}") 
    results = [] # List to store dicts of {'name':, 'status':, 'error':}


    # --- Step 1: Load Model ---
    # model_fp32 = get_model(args.model, args.checkpoint_path)
    # currently failing to pass in checkpoints, vitfly_models and the checkpoints do not match so am using random inits
    # Missing key(s) in state_dict: "fc1.weight_orig", "fc1.weight_u", "fc1.weight_orig", "fc1.weight_u", "fc1.weight_v", "fc2.weight_orig", "fc2.weight_u", "fc2.weight_orig", "fc2.weight_u", "fc2.weight_v", "fc3.weight_orig", "fc3.weight_u", "fc3.weight_orig", "fc3.weight_u", "fc3.weight_v". 
        # Unexpected key(s) in state_dict: "lstm.bias_ih_l0", "lstm.bias_hh_l0", "lstm.bias_ih_l1", "lstm.bias_hh_l1", "lstm.weight_ih_l2", "lstm.weight_hh_l2", "lstm.bias_ih_l2", "lstm.bias_hh_l2", "lstm.weight_ih_l3", "lstm.weight_hh_l3", "lstm.bias_ih_l3", "lstm.bias_hh_l3", "lstm.weight_ih_l4", "lstm.weight_hh_l4", "lstm.bias_ih_l4", "lstm.bias_hh_l4", "fc1.weight", "fc2.weight", "fc3.weight". 
        # size mismatch for lstm.weight_ih_l0: copying a param with shape torch.Size([1024, 660]) from checkpoint, the shape in current model is torch.Size([1580, 665]).
    model_fp32 = get_model(args.model) 
    if model_fp32 is None:
        return

    model_fp32 = strip_spectral_norm(model_fp32)
    sample_inputs = get_sample_inputs(args.model)
    forward_args = get_forward_args_for_export(args.model, sample_inputs)
    with torch.no_grad():
        _ = model_fp32(*forward_args)
    print("Forward pass OK (inputs match vitfly_models.forward).")

    # # --- Step 2: Run all export paths on FP32 model ---
    print_separator(f"STARTING FP32 EXPORT TESTS FOR: {args.model}")
    model_type_str="fp32"
    try_turbine_aot_export(model_fp32, forward_args, args.model, model_type_str, model_output_dir, results)

    # # --- Step 3: Apply Quantization (All Paths) ---
    model_int8 = None
    if args.quant == "torchao_dynamic_int8":
        model_int8 = try_torchao_quantize(model_fp32, results)
    elif args.quant == "torchao_int8_weight_only":
        model_int8 = try_torchao_quantize_weights_only(model_fp32, results)

    # # --- Step 4: Run all export paths on Quantized models ---
    if model_int8 is not None:
        model_type_str = "int8"
        try_turbine_aot_export(model_int8, forward_args, args.model, model_type_str, model_output_dir, results)
    # try_torch_onnx_export(model_int8, forward_args, args.model, model_type_str, model_output_dir, results)

    # # --- Step 5: Print Final Report ---

if __name__ == "__main__":
    main()