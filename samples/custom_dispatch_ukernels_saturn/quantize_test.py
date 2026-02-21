import torch
import torchvision.models as models
import argparse
import os
import copy
import traceback
from contextlib import suppress
from custom_models.vitfly_models_july import ConvNet, LSTMNet, LSTMNetVIT, ViT, UNetConvLSTMNet

def get_sample_inputs(model_name: str):
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


def get_forward_args_for_export(model_name: str, sample_inputs):
    if model_name == "LSTMNetVIT":
        return sample_inputs  # the LSTMNetVIT requires (self, image, velocity, quaternion, h_in, c_in)
    return (sample_inputs,) # the others use (self, X)

def get_model(model_name: str, checkpoint_path=None):
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
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.normpath(os.path.join(script_dir, "..", "pretrained_models", "LSTMnet_model.pth"))
    default_output_dir = os.path.normpath(os.path.join(script_dir, "..", "pretrained_models", args.output_dir))

    print(f"Torch version: {torch.__version__}")
    os.makedirs(default_output_dir, exist_ok=True)
    print(f"Saving model artifacts to: {default_output_dir}")

    # --- Step 1: Load Model ---
    # model_fp32 = get_model(args.model, args.checkpoint_path)
    # currently failing to pass in checkpoints, vitfly_models and the checkpoints do not match so am using random inits
    # Missing key(s) in state_dict: "fc1.weight_orig", "fc1.weight_u", "fc1.weight_orig", "fc1.weight_u", "fc1.weight_v", "fc2.weight_orig", "fc2.weight_u", "fc2.weight_orig", "fc2.weight_u", "fc2.weight_v", "fc3.weight_orig", "fc3.weight_u", "fc3.weight_orig", "fc3.weight_u", "fc3.weight_v". 
        # Unexpected key(s) in state_dict: "lstm.bias_ih_l0", "lstm.bias_hh_l0", "lstm.bias_ih_l1", "lstm.bias_hh_l1", "lstm.weight_ih_l2", "lstm.weight_hh_l2", "lstm.bias_ih_l2", "lstm.bias_hh_l2", "lstm.weight_ih_l3", "lstm.weight_hh_l3", "lstm.bias_ih_l3", "lstm.bias_hh_l3", "lstm.weight_ih_l4", "lstm.weight_hh_l4", "lstm.bias_ih_l4", "lstm.bias_hh_l4", "fc1.weight", "fc2.weight", "fc3.weight". 
        # size mismatch for lstm.weight_ih_l0: copying a param with shape torch.Size([1024, 660]) from checkpoint, the shape in current model is torch.Size([1580, 665]).
    model_fp32 = get_model(args.model) 
    if model_fp32 is None:
        return

    print(model_fp32)

    sample_inputs = get_sample_inputs(args.model)
    forward_args = get_forward_args_for_export(args.model, sample_inputs)
    with torch.no_grad():
        _ = model_fp32(*forward_args)
    print("Forward pass OK (inputs match vitfly_models.forward).")

    # # --- Step 2: Run all export paths on FP32 model ---
    # Use export(model_fp32, *forward_args) so LSTMNetVIT gets 5 args, others get 1 arg (X tuple).
    # try_torch_mlir_export(model_fp32, forward_args, args.model, model_type_str, model_output_dir, results)

    # # --- Step 3: Apply Quantization (All Paths) ---
    # model_quant_ao_dynamic = try_torch_ao_dynamic_quantize(model_fp32, sample_inputs, results)

    # # --- Step 4: Run all export paths on Quantized models ---
    # try_torch_mlir_export(model_fp32, sample_inputs, args.model, model_type_str, model_output_dir, results)

    # # --- Step 5: Print Final Report ---

if __name__ == "__main__":
    main()