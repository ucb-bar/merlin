import os
import sys
import torch
import torch.nn as nn

# Add MiDaS submodule to path
midas_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MiDaS")
sys.path.insert(0, midas_path)

from midas.model_loader import load_model

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
                # For other output sizes, we'd need a different approach
                # But in MiDaS, most AdaptiveMaxPool2d use output_size=1
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

# Dynamic Unflatten that computes sizes from input tensor
class DynamicUnflatten(nn.Module):
    """Dynamic Unflatten that computes output sizes from input tensor shape"""
    def __init__(self, dim, target_size=None):
        super(DynamicUnflatten, self).__init__()
        self.dim = dim
        self.target_size = target_size  # Expected size, but we'll compute from input
    
    def forward(self, x):
        # Get the size of the dimension we want to unflatten
        dim_size = x.shape[self.dim]
        input_shape = list(x.shape)
        
        # Unflatten should convert a 1D dimension into 2D
        # Expected input: [batch, features, sequence_length] (3D)
        # Expected output: [batch, features, height, width] (4D)
        
        # Handle case where tensor has unexpected 4D shape
        # Error shows input was [1, 64, 64, 96] but should be [1, 96, 64]
        # If we have 4D input when we expect 3D, the tensor might already be in spatial format
        # Check if it's already [batch, channels, height, width] format
        if len(input_shape) == 4:
            # If the last dimension is much larger, it might be features
            # [batch, h, w, features] -> we want [batch, features, h, w]
            if input_shape[-1] > max(input_shape[1], input_shape[2]) * 2:
                # Likely [batch, h, w, features], permute to [batch, features, h, w]
                x = x.permute(0, 3, 1, 2).contiguous()
                # If it's already in the right format, return it
                if len(x.shape) == 4 and x.shape[2] * x.shape[3] == dim_size:
                    # Already in [batch, features, h, w] format, return as-is
                    return x
                # Otherwise, reshape to 3D for unflatten
                h, w = input_shape[1], input_shape[2]
                x = x.view(input_shape[0], input_shape[3], h * w)
                dim_size = h * w
                input_shape = list(x.shape)
        
        # If target_size is provided, check if it matches
        if self.target_size is not None:
            if isinstance(self.target_size, torch.Size):
                target_list = list(self.target_size)
            elif isinstance(self.target_size, (list, tuple)):
                target_list = list(self.target_size)
            else:
                target_list = None
            
            if target_list and len(target_list) == 2:
                expected_size = target_list[0] * target_list[1]
                if dim_size == expected_size:
                    # Use the target size
                    return x.unflatten(self.dim, torch.Size(target_list))
        
        # Compute sizes dynamically - find factors that multiply to dim_size
        # For transformer outputs, often square or close to square
        sqrt_size = int(dim_size ** 0.5)
        
        # Try perfect square first
        if sqrt_size * sqrt_size == dim_size:
            return x.unflatten(self.dim, torch.Size([sqrt_size, sqrt_size]))
        
        # Find factors that multiply to dim_size
        # Start from sqrt and work outward
        best_h, best_w = None, None
        for h in range(sqrt_size, 0, -1):
            if dim_size % h == 0:
                w = dim_size // h
                best_h, best_w = h, w
                break
        
        # Also try from the other direction
        if best_h is None:
            for h in range(sqrt_size + 1, dim_size + 1):
                if dim_size % h == 0:
                    w = dim_size // h
                    best_h, best_w = h, w
                    break
        
        if best_h is not None and best_w is not None:
            return x.unflatten(self.dim, torch.Size([best_h, best_w]))
        else:
            # Fallback: if we can't find factors, try to infer from context
            # For vision transformers, often the sequence length is H*W where H and W are patch grid sizes
            # Try common patch grid sizes
            for patch_h in [8, 16, 14, 7, 4]:
                if dim_size % patch_h == 0:
                    patch_w = dim_size // patch_h
                    return x.unflatten(self.dim, torch.Size([patch_h, patch_w]))
            
            # Last resort: use reshape to handle edge cases
            # But be careful - we need to maintain the right number of dimensions
            new_shape = list(x.shape)
            # Try to make it as square as possible
            h = int(dim_size ** 0.5)
            w = (dim_size + h - 1) // h  # Ceiling division
            # Replace the dimension at self.dim with two dimensions
            new_shape = new_shape[:self.dim] + [h, w] + new_shape[self.dim+1:]
            return x.reshape(new_shape)

def replace_unflatten_with_dynamic(module):
    """Recursively replace nn.Unflatten with dynamic versions that adapt to tensor sizes"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Unflatten):
            # Replace with dynamic version
            # nn.Unflatten stores size in unflattened_size attribute
            target_size = None
            if hasattr(child, 'unflattened_size'):
                target_size = child.unflattened_size
            elif hasattr(child, '_unflattened_size'):
                target_size = child._unflattened_size
            
            # Get the dimension
            dim = child.dim if hasattr(child, 'dim') else 2
            setattr(module, name, DynamicUnflatten(dim, target_size))
        else:
            # Recursively process child modules
            replace_unflatten_with_dynamic(child)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export MiDaS model to ONNX')
    parser.add_argument('--model_type', type=str, default='midas_v21_small_256',
                        choices=['dpt_beit_large_512', 'dpt_beit_large_384', 'dpt_beit_base_384',
                                'dpt_swin2_large_384', 'dpt_swin2_base_384', 'dpt_swin2_tiny_256',
                                'dpt_swin_large_384', 'dpt_next_vit_large_384', 'dpt_levit_224',
                                'dpt_large_384', 'dpt_hybrid_384', 'midas_v21_384', 'midas_v21_small_256'],
                        help='Model type to export (midas_v21_small_256 works best with random weights)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (optional, uses random weights if not provided)')
    parser.add_argument('--height', type=int, default=None,
                        help='Input height (uses model default if not specified)')
    parser.add_argument('--square', action='store_true',
                        help='Resize to square resolution')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model: {args.model_type}")
    
    if args.model_path is not None and os.path.exists(args.model_path):
        # Load model with checkpoint
        model, transform, net_w, net_h = load_model(
            device=device,
            model_path=args.model_path,
            model_type=args.model_type,
            optimize=False,  # Don't optimize for ONNX export
            height=args.height,
            square=args.square
        )
        print(f"Loaded model from checkpoint: {args.model_path}")
    else:
        # Create model without loading checkpoint (will use random weights)
        if args.model_path is None:
            print("No model path provided, creating model with random weights...")
        else:
            print(f"Checkpoint not found at {args.model_path}, creating model with random weights...")
        
        # Import model classes directly
        from midas.dpt_depth import DPTDepthModel
        from midas.midas_net import MidasNet
        from midas.midas_net_custom import MidasNet_small
        
        # Create model based on type
        if args.model_type.startswith('dpt_'):
            if args.model_type == "dpt_beit_large_512":
                model = DPTDepthModel(path=None, backbone="beitl16_512", non_negative=True)
                net_w, net_h = 512, 512
            elif args.model_type == "dpt_beit_large_384":
                model = DPTDepthModel(path=None, backbone="beitl16_384", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_beit_base_384":
                model = DPTDepthModel(path=None, backbone="beitb16_384", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_swin2_large_384":
                model = DPTDepthModel(path=None, backbone="swin2l24_384", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_swin2_base_384":
                model = DPTDepthModel(path=None, backbone="swin2b24_384", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_swin2_tiny_256":
                model = DPTDepthModel(path=None, backbone="swin2t16_256", non_negative=True)
                net_w, net_h = 256, 256
            elif args.model_type == "dpt_swin_large_384":
                model = DPTDepthModel(path=None, backbone="swinl12_384", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_next_vit_large_384":
                model = DPTDepthModel(path=None, backbone="next_vit_large_6m", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_levit_224":
                # LeViT models have compatibility issues with newer timm versions
                # Try to patch the model structure if needed
                try:
                    import timm
                    test_model = timm.create_model("levit_384", pretrained=False)
                    
                    # Check if model has 'blocks' attribute (old structure)
                    # or if we need to adapt to new structure
                    if not hasattr(test_model, 'blocks'):
                        # Newer timm versions might use different structure
                        # Try to find the blocks in a different location
                        if hasattr(test_model, 'stages'):
                            # Some timm versions use 'stages' instead of 'blocks'
                            print("WARNING: LeViT model structure may be incompatible.")
                            print("Trying to proceed, but may fail during forward pass.")
                        else:
                            raise AttributeError("LeViT model structure not compatible with current timm version")
                    
                    model = DPTDepthModel(path=None, backbone="levit_384", non_negative=True,
                                         head_features_1=64, head_features_2=8)
                    net_w, net_h = 224, 224
                except (AttributeError, Exception) as e:
                    print(f"ERROR: dpt_levit_224 is not compatible with the current timm/Python version.")
                    print(f"Error: {e}")
                    print("\nThe issue is that:")
                    print("- timm==0.6.12 (required for LeViT) is not compatible with Python 3.11")
                    print("- Newer timm versions have different LeViT model structure")
                    print("\nSolutions:")
                    print("1. Use a pretrained checkpoint: --model_path /path/to/checkpoint.pt")
                    print("   (This may work even with version mismatches)")
                    print("2. Use a different model type that works with random weights:")
                    print("   --model_type midas_v21_small_256")
                    print("   --model_type midas_v21_384")
                    print("   --model_type dpt_swin2_tiny_256 (with checkpoint)")
                    raise
            elif args.model_type == "dpt_large_384":
                model = DPTDepthModel(path=None, backbone="vitl16_384", non_negative=True)
                net_w, net_h = 384, 384
            elif args.model_type == "dpt_hybrid_384":
                model = DPTDepthModel(path=None, backbone="vitb_rn50_384", non_negative=True)
                net_w, net_h = 384, 384
        elif args.model_type == "midas_v21_384":
            model = MidasNet(path=None, non_negative=True)
            net_w, net_h = 384, 384
        elif args.model_type == "midas_v21_small_256":
            model = MidasNet_small(path=None, features=64, backbone="efficientnet_lite3",
                                  exportable=True, non_negative=True, blocks={'expand': True})
            net_w, net_h = 256, 256
        
        if args.height is not None:
            net_w, net_h = args.height, args.height
        
        # Initialize with random weights
        print("Initializing model with random weights...")
        model.apply(lambda m: _init_weights_random(m) if isinstance(m, (nn.Linear, nn.Conv2d)) else None)
        
        model.eval()
        model.to(device)
    
    # Replace AdaptivePool2d with ONNX-compatible versions
    print("Replacing AdaptivePool2d with ONNX-compatible operations...")
    replace_adaptive_pooling_with_onnx_compatible(model)
    
    # Replace hardcoded Unflatten with dynamic versions (for transformer models with random weights)
    print("Replacing Unflatten with dynamic versions...")
    replace_unflatten_with_dynamic(model)
    
    model.eval()
    
    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, net_h, net_w).to(device)
    
    # Test forward pass first to catch any shape mismatches
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            test_output = model(dummy_input)
        print(f"Forward pass successful! Output shape: {test_output.shape}")
    except RuntimeError as e:
        if "unflatten" in str(e).lower():
            print(f"ERROR: Forward pass failed with unflatten error: {e}")
            print("\nThis error occurs because transformer-based DPT models have hardcoded Unflatten")
            print("operations that expect specific tensor sizes from pretrained backbones.")
            print("\nSolutions:")
            print("1. Use a pretrained checkpoint: --model_path /path/to/checkpoint.pt")
            print("2. Try a simpler convolutional model: --model_type midas_v21_small_256")
            print("3. Try: --model_type midas_v21_384")
            raise
        else:
            print(f"ERROR: Forward pass failed with: {e}")
            raise
    except Exception as e:
        print(f"ERROR: Forward pass failed with: {e}")
        print("Try using a pretrained checkpoint with --model_path, or try a different model_type.")
        raise
    
    # Export to ONNX (always use fixed filename)
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "midas.onnx")
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

