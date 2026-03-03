import os
import sys
import torch
import torch.nn as nn

# Add Depth-Anything-V2 submodule to path
depth_anything_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Depth-Anything-V2")
sys.path.insert(0, depth_anything_path)

from depth_anything_v2.dpt import DepthAnythingV2

# Helper function to initialize weights randomly
def _init_weights_random(m):
    """Initialize weights randomly using Kaiming normal for conv/linear layers"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# ONNX-compatible replacement for AdaptiveMaxPool2d(1) and AdaptiveAvgPool2d(1)
class ONNXCompatibleGlobalPool2d(nn.Module):
    """ONNX-compatible replacement for AdaptiveMaxPool2d(1) and AdaptiveAvgPool2d(1)"""
    def __init__(self, mode='max'):
        super(ONNXCompatibleGlobalPool2d, self).__init__()
        self.mode = mode
    
    def forward(self, x):
        # Global pooling: reduce spatial dimensions to 1x1
        if self.mode == 'max':
            # Use torch.max which is ONNX-compatible
            x = torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        else:  # 'avg'
            # Use torch.mean which is ONNX-compatible
            x = torch.mean(x, dim=(-1, -2), keepdim=True)
        return x

def replace_adaptive_pooling_with_onnx_compatible(module):
    """Recursively replace AdaptiveMaxPool2d and AdaptiveAvgPool2d with ONNX-compatible versions"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveMaxPool2d):
            # Check if output size is (1, 1)
            if child.output_size == (1, 1) or child.output_size == 1:
                # Replace with ONNX-compatible version
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='max'))
            else:
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='max'))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            # Check if output size is (1, 1)
            if child.output_size == (1, 1) or child.output_size == 1:
                # Replace with ONNX-compatible version
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='avg'))
            else:
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='avg'))
        else:
            # Recursively process child modules
            replace_adaptive_pooling_with_onnx_compatible(child)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Depth-Anything-V2 model to ONNX')
    parser.add_argument('--model_type', type=str, default='vits',
                        choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Model type to export (vits = Small)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (optional, uses random weights if not provided)')
    parser.add_argument('--input_size', type=int, default=518,
                        help='Input image size (default: 518, must be multiple of 14, e.g., 224, 518)')
    
    args = parser.parse_args()
    
    # Model configurations
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print(f"Creating model: Depth-Anything-V2-{args.model_type.upper()}")
    config = model_configs[args.model_type]
    model = DepthAnythingV2(**config)
    
    # Load checkpoint if provided
    if args.model_path is not None and os.path.exists(args.model_path):
        print(f"Loading checkpoint from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    else:
        if args.model_path is None:
            print("No model path provided, using random weights...")
        else:
            print(f"Checkpoint not found at {args.model_path}, using random weights...")
        # Initialize with random weights
        model.apply(lambda m: _init_weights_random(m) if isinstance(m, (nn.Linear, nn.Conv2d)) else None)
    
    # Replace AdaptivePool2d with ONNX-compatible versions
    print("Replacing AdaptivePool2d with ONNX-compatible operations...")
    replace_adaptive_pooling_with_onnx_compatible(model)
    
    model.eval()
    model.to(device)
    
    # Create dummy input (batch_size, channels, height, width)
    # Input size should be multiple of 14 (patch size)
    input_size = args.input_size
    if input_size % 14 != 0:
        input_size = ((input_size // 14) + 1) * 14
        print(f"Adjusting input size to {input_size} (multiple of 14)")
    
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Test forward pass first
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        print(f"Forward pass successful! Output shape: {test_output.shape}")
    except Exception as e:
        print(f"ERROR: Forward pass failed with: {e}")
        print("This might be due to shape mismatches with random weights.")
        print("Try using a pretrained checkpoint with --model_path")
        raise
    
    # Export to ONNX (always use fixed filename)
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "depth_anything_v2.onnx")
    print(f"Exporting ONNX model to {onnx_path}...")
    print(f"Input shape: {dummy_input.shape}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['depth'],
        opset_version=18,  # Use 18 to match PyTorch's default and avoid version conversion issues
    )
    
    print(f"Successfully exported ONNX model to {onnx_path}")

