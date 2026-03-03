import os
import sys
import torch
import torch.nn as nn

# Add TinyDepth submodule to path
tinydepth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TinyDepth")
sys.path.insert(0, tinydepth_path)

# Workaround for mmcv compatibility issue - create a minimal stub if mmcv.fileio doesn't exist
try:
    from mmcv.fileio import FileClient
except (ImportError, ModuleNotFoundError):
    # mmcv 2.x compatibility - create minimal stubs
    import types
    mmcv_custom_path = os.path.join(tinydepth_path, "mmcv_custom")
    if mmcv_custom_path not in sys.path:
        sys.path.insert(0, mmcv_custom_path)
    
    # Create a minimal mmcv module stub
    import tempfile
    temp_mmcv_dir = tempfile.mkdtemp(prefix='mmcv_mock_')
    os.makedirs(os.path.join(temp_mmcv_dir, 'model_zoo'), exist_ok=True)
    # Create empty JSON files that might be accessed
    for json_file in ['open_mmlab.json', 'mmcls.json', 'deprecated.json']:
        json_path = os.path.join(temp_mmcv_dir, 'model_zoo', json_file)
        with open(json_path, 'w') as f:
            f.write('{}')
    
    class MockMMCV:
        class fileio:
            class FileClient:
                pass
            @staticmethod
            def load(*args, **kwargs):
                return {}
        class parallel:
            @staticmethod
            def is_module_wrapper(*args, **kwargs):
                return False
        class utils:
            @staticmethod
            def mkdir_or_exist(*args, **kwargs):
                pass
        class runner:
            @staticmethod
            def get_dist_info():
                return (0, 1)
        __version__ = "2.0.0"
        __path__ = [temp_mmcv_dir]
        @staticmethod
        def mkdir_or_exist(*args, **kwargs):
            pass
    
    # Inject the mock into sys.modules before importing mmcv_custom
    mock_mmcv = MockMMCV()
    sys.modules['mmcv'] = mock_mmcv
    sys.modules['mmcv.fileio'] = types.ModuleType('mmcv.fileio')
    sys.modules['mmcv.fileio'].FileClient = mock_mmcv.fileio.FileClient
    sys.modules['mmcv.fileio'].load = mock_mmcv.fileio.load
    sys.modules['mmcv.parallel'] = types.ModuleType('mmcv.parallel')
    sys.modules['mmcv.parallel'].is_module_wrapper = mock_mmcv.parallel.is_module_wrapper
    sys.modules['mmcv.utils'] = types.ModuleType('mmcv.utils')
    sys.modules['mmcv.utils'].mkdir_or_exist = mock_mmcv.utils.mkdir_or_exist
    sys.modules['mmcv.runner'] = types.ModuleType('mmcv.runner')
    sys.modules['mmcv.runner'].get_dist_info = mock_mmcv.runner.get_dist_info

from networks.configuration import get_config
from networks.build import build_model
from networks.fusion_decoder import FusionDecoder

# Patch TinyViT.init_weights to handle None/empty string properly (bug in original code)
from networks.tiny_vit import TinyViT
_original_init_weights = TinyViT.init_weights
def patched_init_weights(self, pretrained=None):
    if isinstance(pretrained, str) and pretrained.strip():
        # Only load checkpoint if pretrained is a non-empty string
        _original_init_weights(self, pretrained)
    # If pretrained is None or empty string, just skip (weights already initialized in __init__ via _init_weights)
TinyViT.init_weights = patched_init_weights

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
                # But in TinyDepth, all AdaptiveMaxPool2d use output_size=1
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

# Create a simple args object for configuration
class Args:
    def __init__(self):
        self.attn_dropout = 0.0
        self.ffn_dropout = 0.0
        self.dropout = 0.1
        self.conv_kernel_size = 3
        self.width_multiplier = 1.0
        self.backbone_mode = 'small'
        self.backbone = 'mobilevit'
        self.transformer_norm_layer = "layer_norm"
        self.head_dim = None
        self.number_heads = 4
        self.activation_name = "prelu"
        self.conv_init = 'kaiming_normal'
        self.conv_init_std_dev = None
        self.linear_init = 'xavier_uniform'
        self.linear_init_std_dev = 0.01
        self.normalization_name = 'batch_norm'
        self.normalization_momentum = 0.1

class TinyDepthModel(nn.Module):
    """Wrapper model that combines encoder and decoder for ONNX export"""
    def __init__(self, encoder, decoder):
        super(TinyDepthModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        # Encoder forward pass
        features = self.encoder(x)
        # Decoder forward pass
        outputs = self.decoder(features)
        # Return disparity output (scale 0)
        return outputs[("disp", 0)]

if __name__ == "__main__":
    # Default image dimensions for TinyDepth (from config)
    img_height = 192
    img_width = 640
    
    # Create configuration
    args = Args()
    config = get_config(args)
    # Unfreeze config to modify PRETRAINED (set to empty string to skip loading)
    config.defrost()
    config.MODEL.PRETRAINED = ""  # Empty string to skip checkpoint loading
    config.freeze()
    
    # Build encoder
    print("Building encoder...")
    encoder = build_model(config)
    # Encoder already initializes weights randomly in __init__ via _init_weights
    # No checkpoint will be loaded since PRETRAINED is empty string
    encoder.eval()
    
    # Build decoder
    print("Building decoder...")
    num_ch_enc = [64, 64, 128, 160, 320]  # Encoder feature channels
    decoder = FusionDecoder(num_ch_enc)
    # Initialize decoder with random weights (PyTorch default, but explicitly initialize for consistency)
    decoder.apply(lambda m: _init_weights_random(m) if isinstance(m, (nn.Linear, nn.Conv2d)) else None)
    decoder.eval()
    
    # Create combined model
    model = TinyDepthModel(encoder, decoder)
    
    # Replace AdaptiveMaxPool2d and AdaptiveAvgPool2d with ONNX-compatible versions
    print("Replacing AdaptivePool2d with ONNX-compatible operations...")
    replace_adaptive_pooling_with_onnx_compatible(model)
    
    model.eval()
    
    # Create dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, img_height, img_width)
    
    # Export to ONNX
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinydepth.onnx")
    print(f"Exporting ONNX model to {onnx_path}...")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['disparity'],
        opset_version=17,
    )
    
    print(f"Successfully exported ONNX model to {onnx_path}")

