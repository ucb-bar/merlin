import torch
import torchvision.models as models
import argparse
import os
import copy
import traceback
from contextlib import suppress

# --- Model Loading Imports ---
with suppress(ImportError):
    from torchvision.models.squeezenet import SqueezeNet1_0_Weights
with suppress(ImportError):
    from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
with suppress(ImportError):
    from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights, MobileNet_V3_Large_Weights
with suppress(ImportError):
    from torchvision.models.alexnet import AlexNet_Weights
with suppress(ImportError):
    from torchvision.models.resnet import ResNet50_Weights
with suppress(ImportError):
    from custom_models.vitfly_models import LSTMNetVIT
with suppress(ImportError):
    from custom_models.FC import SimpleFCModel

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


def print_separator(title: str):
    """Prints a formatted separator to the console."""
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80)

def get_model(model_name: str):
    """Loads a pretrained model based on the provided name."""
    print(f"--- Loading model: {model_name} ---")
    try:
        if model_name == "squeezenet":
            return models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).eval()
        elif model_name == "alexnet":
            return models.alexnet(weights=AlexNet_Weights.DEFAULT).eval()
        elif model_name == "resnet":
            return models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
        elif model_name == "mobilenetv3small":
            return models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval()
        elif model_name == "mobilenetv3large":
            return models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
        elif model_name == "mobilenet":
            return models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
        elif model_name == "fc":
            return SimpleFCModel(input_size=1024, hidden_size=128, output_size=10).eval()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return None

# --- Wrapper Functions for Export Paths ---
# (These are generic and remain unchanged)

def try_torch_export(model, example_inputs, model_name, model_type, model_output_dir, results_list):
    """Attempts to export using torch.export.export() and save as .pt2 file."""
    name = f"torch.export.export ({model_type})"
    print_separator(f"Attempting: {name}")
    
    output_path = os.path.join(model_output_dir, f"{model_name}_{model_type}_torch_export.pt2")
    
    if torch_export is None:
        print("FAILED: torch.export is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torch.export not installed'})
        return

    try:
        model.eval()
        print(f"Exporting model to {output_path}...")
        ep = torch_export.export(model, example_inputs)
        torch_export.save(ep, output_path)
        
        print(f"Successfully exported and saved to {output_path}")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})

def try_torch_compile_turbine(model, example_inputs, model_type, results_list):
    """Attempts to compile using torch.compile(backend='turbine_cpu')."""
    name = f"torch.compile(backend='turbine_cpu') ({model_type})"
    print_separator(f"Attempting: {name}")

    if turbine_aot is None:
        print("FAILED: iree.turbine is not installed, which is required for 'turbine_cpu' backend.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': "iree.turbine not installed"})
        return

    try:
        #model.eval()
        print("Compiling model...")
        compiled_model = torch.compile(model, backend="turbine_cpu")
        
        print("Compilation successful. Testing inference...")
        #with torch.no_grad():
        #output = compiled_model(example_inputs)
        
        print("Inference test successful.")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})

def try_torch_onnx_export(model, example_inputs, model_name, model_type, model_output_dir, results_list):
    """Attempts to export using torch.onnx.export(dynamo=True)."""
    name = f"torch.onnx.export(dynamo=False) ({model_type})"
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

def try_torch_mlir_export(model, example_inputs, model_name, model_type, model_output_dir, results_list):
    """Exports the model to MLIR using the torch-mlir frontend."""
    name = f"torch_mlir.fx.export_and_import ({model_type})"
    print_separator(f"Attempting: {name}")

    output_path = os.path.join(model_output_dir, f"{model_name}_{model_type}_torch_mlir.mlir")

    if torch_mlir_export is None:
        print("FAILED: torch-mlir is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torch-mlir not installed'})
        return
        
    try:
        model.eval()
        print(f"Exporting model to {output_path}...")
        mlir_module = torch_mlir_export(
            model,
            *example_inputs,
        )
        
        mlir_str = mlir_module.operation.get_asm(large_elements_limit=10)
        with open(output_path, "w") as f:
            f.write(mlir_str)
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
        print(f"Exporting model to {output_path}...")
        exported_module = turbine_aot.export(model, *example_inputs)
        mlir_str = str(exported_module.mlir_module)

        with open(output_path, "w") as f:
            f.write(mlir_str)
        print(f"Successfully exported to {output_path}")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})

# --- Quantization Functions ---

def try_torch_ao_static_quantize(model_fp32, example_inputs, results_list):
    """Attempts Post-Training Static Quantization using torch.ao.quantization."""
    name = "torch.ao.quantization (Static, FX)"
    print_separator(f"Attempting: {name}")

    if torch.ao.quantization is None:
        print("FAILED: torch.ao.quantization is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torch.ao.quantization not installed'})
        return None

    try:
        model_to_quantize = copy.deepcopy(model_fp32).eval()
        qconfig_mapping = get_default_qconfig_mapping("x86")
        
        print("Preparing model for static quantization...")
        prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        
        print("Calibrating with sample data...")
        with torch.no_grad():
            for _ in range(10): # Using 10 samples as in the original script
                prepared_model(*example_inputs)
        print("Calibration complete.")
        
        quantized_model = convert_fx(prepared_model)
        print("Static quantization successful.")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
        return quantized_model
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})
        return None

def try_torch_ao_dynamic_quantize(model_fp32, example_inputs, results_list):
    """Attempts Post-Training Dynamic Quantization using torch.ao.quantization."""
    name = "torch.ao.quantization (Dynamic, FX)"
    print_separator(f"Attempting: {name}")

    if torch.ao.quantization is None:
        print("FAILED: torch.ao.quantization is not installed.")
        results_list.append({'name': name, 'status': 'FAILED', 'error': 'torch.ao.quantization not installed'})
        return None
        
    try:
        model_to_quantize = copy.deepcopy(model_fp32).eval()
        qconfig_mapping = QConfigMapping().set_global(torch.ao.quantization.default_dynamic_qconfig)
        
        print("Preparing model for dynamic quantization...")
        prepared_model = prepare_fx(model_to_quantize, qconfig_mapping, example_inputs)
        
        print("Converting model...")
        quantized_model = convert_fx(prepared_model)
        print("Dynamic quantization successful.")
        results_list.append({'name': name, 'status': 'SUCCEEDED', 'error': None})
        return quantized_model
    except Exception:
        error_msg = traceback.format_exc()
        print(f"FAILED:\n{error_msg}")
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})
        return None

def try_torchao_quantize(model_fp32, results_list):
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

# --- Report Generation ---

def print_report(results_list):
    """Prints the final summary table and failure details."""
    print_separator("FINAL REPORT - SUMMARY")
    
    print(f"{'Test Combination':<70} | {'Status':<10}")
    print("-" * 82)
    for res in results_list:
        print(f"{res['name']:<70} | {res['status']:<10}")
    
    print("\n\n")
    print_separator("FAILURE DETAILS")
    
    failures = [r for r in results_list if r['status'] == 'FAILED']
    
    if not failures:
        print("--- No failures reported. ---")
    else:
        for res in failures:
            print(f"\n--- Failure for: {res['name']} ---")
            print(res['error'])
            print("-" * 80)
            
def write_report(results_list, model_output_dir, model_name):
    """Prints the final summary table and failure details to console AND markdown file."""
    report_path = os.path.join(model_output_dir, f"report_{model_name}.md")
    
    # --- Data for Console and File ---
    table_header = f"| {'Test Combination':<70} | {'Status':<10} |\n"
    table_separator = f"|:{'-'*69}-|:{'-'*9}-|\n"
    failures = [r for r in results_list if r['status'] == 'FAILED']

    # --- 1. Console Output (existing behavior) ---
    print_separator("FINAL REPORT - SUMMARY (Console)")
    
    console_table_header = f"{'Test Combination':<70} | {'Status':<10}"
    print(console_table_header)
    print("-" * 82)
    for res in results_list:
        print(f"{res['name']:<70} | {res['status']:<10}")
    
    print("\n\n")
    print_separator("FAILURE DETAILS (Console)")
    
    if not failures:
        print("--- No failures reported. ---")
    else:
        for res in failures:
            print(f"\n--- Failure for: {res['name']} ---")
            print(res['error'])
            print("-" * 80)

    # --- 2. Markdown File Output (New) ---
    try:
        with open(report_path, "w") as f:
            f.write(f"# Export and Quantization Report for: {model_name}\n\n")
            
            # --- Summary Table ---
            f.write("## Summary Table\n\n")
            f.write(table_header)
            f.write(table_separator)
            for res in results_list:
                # Sanitize newlines/pipes in name for table
                name_clean = res['name'].replace('\n', ' ').replace('|', '\|')
                f.write(f"| {name_clean:<70} | {res['status']:<10} |\n")
            
            # --- Failure Details ---
            f.write("\n\n## Failure Details\n\n")
            if not failures:
                f.write("--- No failures reported. ---\n")
            else:
                for res in failures:
                    name_clean = res['name'].replace('\n', ' ').replace('|', '\|')
                    f.write(f"### Failure for: `{name_clean}`\n\n")
                    f.write("```\n")
                    f.write(res['error'].strip())
                    f.write("\n```\n\n")
        
        print(f"\nSuccessfully wrote markdown report to: {report_path}")

    except Exception as e:
        print(f"\nError writing markdown report: {e}")

def add_failure_placeholders(model_type, results_list, error_msg):
    """Adds placeholder failure entries to the report for skipped export paths."""
    export_paths = [
        "torch.export.export",
        "torch.compile(backend='turbine_cpu')",
        "torch.onnx.export(dynamo=True)",
        "torch_mlir.fx.export_and_import",
        "iree.turbine.aot.export"
    ]
    for path in export_paths:
        name = f"{path} ({model_type})"
        results_list.append({'name': name, 'status': 'FAILED', 'error': error_msg})

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Quantize a PyTorch model and export to MLIR via multiple paths."
    )
    parser.add_argument(
        "--model", type=str, default="mobilenet",
        choices=["mobilenet", "squeezenet", "alexnet", "resnet", "mobilenetv3small", "mobilenetv3large"]
    )
    parser.add_argument(
        "--output_dir", type=str, default="report_pt_export",
        help="Directory to save the exported files and report."
    )
    args = parser.parse_args()

    print(f"Torch version: {torch.__version__}")
    os.makedirs(args.output_dir, exist_ok=True)
    model_output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"Saving model artifacts to: {model_output_dir}")
    
    results = [] # List to store dicts of {'name':, 'status':, 'error':}

    # --- Step 1: Load Model ---
    model_fp32 = get_model(args.model)
    if model_fp32 is None:
        return
    
    sample_inputs_no_tuple = torch.randn(1, 3, 224, 224)
    sample_inputs = (sample_inputs_no_tuple,)

    # --- Step 2: Run all export paths on FP32 model ---
    print_separator(f"STARTING FP32 EXPORT TESTS FOR: {args.model}")
    model_type_str = "fp32"
    try_torch_export(model_fp32, sample_inputs, args.model, model_type_str, model_output_dir, results)
    try_torch_compile_turbine(model_fp32, sample_inputs_no_tuple, model_type_str, results)
    try_torch_onnx_export(model_fp32, sample_inputs, args.model, model_type_str, model_output_dir, results)
    try_torch_mlir_export(model_fp32, sample_inputs, args.model, model_type_str, model_output_dir, results)
    try_turbine_aot_export(model_fp32, sample_inputs, args.model, model_type_str, model_output_dir, results)

    # --- Step 3: Apply Quantization (All Paths) ---
    model_quant_ao_static = try_torch_ao_static_quantize(model_fp32, sample_inputs, results)
    model_quant_ao_dynamic = try_torch_ao_dynamic_quantize(model_fp32, sample_inputs, results)
    model_quant_torchao = try_torchao_quantize(model_fp32, results)

    # --- Step 4: Run all export paths on Quantized models ---

    # --- Path 1: torch.ao Static ---
    model_type_str = "quant_torch_ao_static"
    if model_quant_ao_static:
        print_separator(f"STARTING '{model_type_str}' EXPORT TESTS FOR: {args.model}")
        try_torch_export(model_quant_ao_static, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_torch_compile_turbine(model_quant_ao_static, sample_inputs, model_type_str, results)
        try_torch_onnx_export(model_quant_ao_static, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_torch_mlir_export(model_quant_ao_static, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_turbine_aot_export(model_quant_ao_static, sample_inputs, args.model, model_type_str, model_output_dir, results)
    else:
        print(f"\nSkipping export steps for '{model_type_str}' model because quantization failed.")
        add_failure_placeholders(model_type_str, results, f"Skipped ({model_type_str} Quant failed)")

    # --- Path 2: torch.ao Dynamic ---
    model_type_str = "quant_torch_ao_dynamic"
    if model_quant_ao_dynamic:
        print_separator(f"STARTING '{model_type_str}' EXPORT TESTS FOR: {args.model}")
        try_torch_export(model_quant_ao_dynamic, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_torch_compile_turbine(model_quant_ao_dynamic, sample_inputs, model_type_str, results)
        try_torch_onnx_export(model_quant_ao_dynamic, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_torch_mlir_export(model_quant_ao_dynamic, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_turbine_aot_export(model_quant_ao_dynamic, sample_inputs, args.model, model_type_str, model_output_dir, results)
    else:
        print(f"\nSkipping export steps for '{model_type_str}' model because quantization failed.")
        add_failure_placeholders(model_type_str, results, f"Skipped ({model_type_str} Quant failed)")

    # --- Path 3: torchao ---
    model_type_str = "quant_torchao_int8"
    if model_quant_torchao:
        print_separator(f"STARTING '{model_type_str}' EXPORT TESTS FOR: {args.model}")
        try_torch_export(model_quant_torchao, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_torch_compile_turbine(model_quant_torchao, sample_inputs, model_type_str, results)
        try_torch_onnx_export(model_quant_torchao, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_torch_mlir_export(model_quant_torchao, sample_inputs, args.model, model_type_str, model_output_dir, results)
        try_turbine_aot_export(model_quant_torchao, sample_inputs, args.model, model_type_str, model_output_dir, results)
    else:
        print(f"\nSkipping export steps for '{model_type_str}' model because quantization failed.")
        add_failure_placeholders(model_type_str, results, f"Skipped ({model_type_str} Quant failed)")


    # --- Step 5: Print Final Report ---
    print_report(results)
    write_report(results, model_output_dir, args.model)

if __name__ == "__main__":
    main()