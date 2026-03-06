import onnx
import argparse
import os
import json
import numpy as np
import sys

# --- ONNX Runtime Imports ---
try:
    import onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader, QuantFormat
except ImportError:
    print("❌ Error: onnxruntime is not installed. Run: pip install onnx onnxruntime")
    sys.exit(1)

# ==============================================================================
# 1. Calibration Data Reader
# ==============================================================================
class JSONConfigCalibrationDataReader(CalibrationDataReader):
    """
    Generates random calibration data based on shapes defined in the JSON config.
    """
    def __init__(self, input_names, input_shapes, num_samples=50):
        self.input_names = input_names
        self.input_shapes = input_shapes
        self.num_samples = num_samples
        self.data_generator = self._data_generator()

        print(f"  📊 Calibration Reader ready. Inputs: {dict(zip(input_names, input_shapes))}")

    def _data_generator(self):
        for _ in range(self.num_samples):
            sample = {}
            for name, shape in zip(self.input_names, self.input_shapes):
                sample[name] = np.random.randn(*shape).astype(np.float32)
            yield sample

    def get_next(self):
        return next(self.data_generator, None)

    def rewind(self):
        self.data_generator = self._data_generator()

# ==============================================================================
# 2. Main Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Quantize an existing ONNX model using a JSON config.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model key in models_config.json")
    parser.add_argument("--config", type=str, default="models_config.json", help="Path to the configuration JSON.")
    args = parser.parse_args()

    # --- Robust Config Resolution ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = args.config

    if not os.path.exists(config_path):
        # Fallback: Try finding it relative to the script directory
        candidate = os.path.join(script_dir, os.path.basename(config_path))
        if os.path.exists(candidate):
            config_path = candidate
        else:
            print(f"❌ Error: Config file '{args.config}' not found.")
            sys.exit(1)

    print(f"  📄 Using config file: {config_path}")

    with open(config_path, 'r') as f:
        full_config = json.load(f)

    if args.model not in full_config:
        print(f"❌ Error: Model '{args.model}' not found in config.")
        sys.exit(1)

    model_config = full_config[args.model]

    # --- Robust ONNX Path Resolution ---
    source_onnx_path = model_config.get("source_path")
    
    if source_onnx_path and not os.path.exists(source_onnx_path):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        candidate_source = os.path.join(config_dir, source_onnx_path)
        if os.path.exists(candidate_source):
            source_onnx_path = candidate_source

    if not source_onnx_path or not os.path.exists(source_onnx_path):
        print(f"❌ Error: Source ONNX file not found: {model_config.get('source_path')}")
        sys.exit(1)

    # --- Output Path (Same Folder) ---
    model_dir = os.path.dirname(source_onnx_path)
    filename = os.path.basename(source_onnx_path)
    filename_no_ext = os.path.splitext(filename)[0]
    
    # Save directly next to the original file
    output_onnx_path = os.path.join(model_dir, f"{filename_no_ext}.q.int8.onnx")

    print(f"==================================================")
    print(f"🚀 Processing: {args.model}")
    print(f"   Input:  {source_onnx_path}")
    print(f"   Output: {output_onnx_path}")
    print(f"==================================================")

    # --- Auto-Detect Input Names from Model ---
    # This prevents the "Required inputs missing" error by ignoring JSON names 
    # and using what the model actually wants.
    print("  🔍 Inspecting model for input names...")
    model_proto = onnx.load(source_onnx_path)
    real_input_names = [node.name for node in model_proto.graph.input]
    
    # Filter out initializers (weights) that might appear in input list
    initializers = {init.name for init in model_proto.graph.initializer}
    real_input_names = [name for name in real_input_names if name not in initializers]

    print(f"     Found inputs: {real_input_names}")

    input_shapes = model_config.get("input_shapes")
    if not input_shapes:
        print("❌ Error: 'input_shapes' must be defined in JSON.")
        sys.exit(1)

    if len(real_input_names) != len(input_shapes):
        print(f"❌ Error: Model has {len(real_input_names)} inputs, but JSON defines {len(input_shapes)} shapes.")
        sys.exit(1)

    # --- Calibration Setup ---
    data_reader = JSONConfigCalibrationDataReader(
        input_names=real_input_names, # Use the REAL names from the model
        input_shapes=input_shapes
    )

    # --- Quantization Execution ---
    print("\n⚙️  Running Static Quantization (QDQ)...")
    
    try:
        quantize_static(
            model_input=source_onnx_path,
            model_output=output_onnx_path,
            calibration_data_reader=data_reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=False,
            extra_options={
                'ActivationSymmetric': True,
                'WeightSymmetric': True
            }
        )
        print(f"\n✅ Success! Saved to: {output_onnx_path}")
        
    except Exception as e:
        print(f"\n❌ Quantization Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()