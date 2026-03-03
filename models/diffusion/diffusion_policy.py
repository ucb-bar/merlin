import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F

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
            x = torch.max(torch.max(x, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]
        else:  # 'avg'
            x = torch.mean(x, dim=(-1, -2), keepdim=True)
        return x

def replace_adaptive_pooling_with_onnx_compatible(module):
    """Recursively replace AdaptiveMaxPool2d and AdaptiveAvgPool2d with ONNX-compatible versions"""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveMaxPool2d):
            if child.output_size == (1, 1) or child.output_size == 1:
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='max'))
            else:
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='max'))
        elif isinstance(child, nn.AdaptiveAvgPool2d):
            if child.output_size == (1, 1) or child.output_size == 1:
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='avg'))
            else:
                setattr(module, name, ONNXCompatibleGlobalPool2d(mode='avg'))
        else:
            replace_adaptive_pooling_with_onnx_compatible(child)

# Time embedding for diffusion
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# Simple U-Net block for diffusion
class UNetBlock(nn.Module):
    """Basic U-Net block with residual connection"""
    def __init__(self, in_channels, out_channels, time_emb_dim, downsample=True):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.downsample = downsample
        
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x, time_emb):
        residual = self.residual(x)
        if self.downsample:
            residual = F.interpolate(residual, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        x = self.conv1(x)
        if self.downsample and x.shape != residual.shape:
            x = F.interpolate(x, size=residual.shape[2:], mode='bilinear', align_corners=False)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        x = x + time_emb
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        return x + residual

# Simplified feature processing block (fewer operations)
class SimpleBlock(nn.Module):
    """Simple block with linear + activation + residual"""
    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, features)
    
    def forward(self, x):
        return x + F.relu(self.linear(x))

# Action sequence refinement block for policy head
class ActionRefinementBlock(nn.Module):
    """Refines action sequence with temporal awareness"""
    def __init__(self, action_dim, num_actions):
        super().__init__()
        self.action_dim = action_dim
        self.num_actions = num_actions
        # Process actions as sequence
        self.temporal_conv = nn.Conv1d(action_dim, action_dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(action_dim)
        self.linear = nn.Linear(action_dim, action_dim)
    
    def forward(self, actions):
        # actions: [batch, num_actions, action_dim]
        # Apply temporal convolution
        actions_t = actions.transpose(1, 2)  # [batch, action_dim, num_actions]
        refined = self.temporal_conv(actions_t)  # [batch, action_dim, num_actions]
        refined = refined.transpose(1, 2)  # [batch, num_actions, action_dim]
        
        # Layer norm and residual
        refined = self.norm(refined)
        refined = refined + F.relu(self.linear(refined))
        
        return refined + actions  # Residual connection

# Simplified CNN encoder for images - 2-3x more compute intensive
class ImageEncoder(nn.Module):
    """Simple CNN encoder for RGB images - increased channels for 2-3x more compute"""
    def __init__(self, hidden_dim):
        super().__init__()
        # Simple CNN: 224x224 -> 7x7 (32x downsampling)
        # Increased channels by ~2.5x: 3 -> 16 -> 32 -> 64 -> 128 -> 128
        self.conv_layers = nn.Sequential(
            # 224x224 -> 112x112
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            # 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # 14x14 -> 7x7
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Global average pooling: 7x7 -> 1x1
        # Then project to hidden_dim * 4 (kept same for compatibility)
        self.proj = nn.Linear(128, hidden_dim * 4)
    
    def forward(self, x):
        # x: [batch, 3, 224, 224]
        feat = self.conv_layers(x)  # [batch, 128, 7, 7] (increased from 64)
        # Global average pooling
        feat = feat.mean(dim=(2, 3))  # [batch, 128]
        feat = F.relu(self.proj(feat))  # [batch, hidden*4]
        return feat

# Diffusion Policy Model - simplified for fewer ONNX operators
class DiffusionPolicy(nn.Module):
    """
    Diffusion Policy model for control tasks.
    Takes past robot states (vectors) and predicts action sequences.
    Simplified architecture with fewer, well-supported operators.
    """
    def __init__(self, 
                 state_dim=14,
                 action_dim=7,
                 hidden_dim=256,
                 num_actions=16,
                 num_past_states=4):
        super(DiffusionPolicy, self).__init__()
        
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.state_dim = state_dim
        self.num_past_states = num_past_states
        time_emb_dim = hidden_dim
        
        # Time embedding - simplified
        self.time_embed = SinusoidalPositionEmbeddings(time_emb_dim)
        # Output dimension matches combined features: state + action + image
        feature_dim = hidden_dim * 4
        self.time_proj = nn.Linear(time_emb_dim, feature_dim * 3)
        
        # Image encoder for RGB 224x224
        self.image_encoder = ImageEncoder(hidden_dim)
        
        # State encoder - deeper instead of wider (more substantial)
        encoder_dim = state_dim * num_past_states
        self.state_encoder = nn.Sequential(
            nn.Linear(encoder_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),  # Extra layer for depth
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),  # Extra layer for depth
            nn.ReLU(),
        )
        
        # Action embedding - deeper instead of wider
        action_input_dim = action_dim * num_actions
        self.action_embed = nn.Sequential(
            nn.Linear(action_input_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),  # Extra layer for depth
            nn.ReLU(),
        )
        
        # Image feature projection for attention (to match state/action dim)
        self.image_proj_attn = nn.Linear(feature_dim, feature_dim)
        
        # Multiple cross-attention layers for more substantial processing
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True
            ) for _ in range(3)  # 3 layers for depth
        ])
        
        # More fusion blocks for substantial processing
        self.fusion_blocks = nn.ModuleList([
            SimpleBlock(feature_dim) for _ in range(5)  # 5 blocks for depth
        ])
        
        # Decoder - deeper instead of wider
        # Now includes image features: state + action + image = 3 * feature_dim
        combined_dim = feature_dim * 3
        self.decoder = nn.Sequential(
            nn.Linear(combined_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),  # Extra layer
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),  # Extra layer
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim * num_actions),
        )
        
        # Policy head layers - refine action sequence
        # Action sequence refinement blocks
        self.action_refinement = nn.ModuleList([
            ActionRefinementBlock(action_dim, num_actions) for _ in range(2)  # 2 refinement passes
        ])
        
        # Optional: Action normalization (tanh for bounded actions)
        # Uncomment if actions should be in [-1, 1] range
        # self.action_norm = nn.Tanh()
        
    def denoise_step(self, past_states, actions, timestep, image=None):
        """
        Single denoising step - simplified with fewer operations.
        
        Args:
            past_states: past robot states [batch, num_past_states, state_dim] or flattened
            actions: noisy action sequence [batch, num_actions, action_dim]
            timestep: diffusion timestep [batch]
            image: RGB image [batch, 3, 224, 224] (optional)
        
        Returns:
            predicted noise [batch, num_actions, action_dim]
        """
        batch_size = past_states.shape[0]
        
        # Flatten past states if needed
        if len(past_states.shape) == 3:
            past_states = past_states.view(batch_size, -1)  # [batch, num_past_states * state_dim]
        
        # Encode past states - deeper processing
        state_feat = self.state_encoder(past_states)  # [batch, hidden*4]
        
        # Encode image if provided
        if image is not None:
            image_feat = self.image_encoder(image)  # [batch, hidden*4]
        else:
            # Use zero features if no image
            image_feat = torch.zeros(batch_size, hidden_dim * 4, device=past_states.device)  # [batch, hidden*4]
        
        # Embed actions - deeper processing
        actions_flat = actions.view(batch_size, -1)  # [batch, num_actions * action_dim]
        action_feat = self.action_embed(actions_flat)  # [batch, hidden*4]
        
        # Time embedding
        time_emb = self.time_embed(timestep.float())  # [batch, time_emb_dim]
        time_emb = F.relu(self.time_proj(time_emb))  # [batch, hidden*4 * 3]
        
        # Multiple cross-attention layers for substantial processing
        # Project image features for attention
        image_feat_proj = F.relu(self.image_proj_attn(image_feat))  # [batch, hidden*4]
        image_feat_attn = image_feat_proj.unsqueeze(1)  # [batch, 1, hidden*4]
        state_action_feat = (state_feat + action_feat).unsqueeze(1)  # [batch, 1, hidden*4]
        
        # Apply multiple cross-attention layers
        attended_feat = image_feat_attn
        for cross_attn in self.cross_attention_layers:
            attended_feat, _ = cross_attn(
                attended_feat, state_action_feat, state_action_feat
            )
        attended_feat = attended_feat.squeeze(1)  # [batch, hidden*4]
        
        # More fusion blocks for substantial processing
        fused_feat = attended_feat
        for fusion_block in self.fusion_blocks:
            fused_feat = fusion_block(fused_feat)
        
        # Combine features: state + action + image
        combined = torch.cat([state_feat, action_feat, image_feat], dim=1)  # [batch, hidden*4 * 3]
        
        # Add time conditioning
        combined = combined + time_emb
        
        # Decode to action prediction
        pred = self.decoder(combined)  # [batch, num_actions * action_dim]
        pred = pred.view(batch_size, self.num_actions, self.action_dim)
        
        # Apply policy head refinement layers
        for refinement_block in self.action_refinement:
            pred = refinement_block(pred)  # [batch, num_actions, action_dim]
        
        # Optional: Normalize actions to [-1, 1] if needed
        # pred = self.action_norm(pred)
        
        return pred
    
    def forward(self, past_states, actions, timestep, image=None):
        """
        Single denoising step - simplified to one iteration.
        
        Args:
            past_states: past robot states [batch, num_past_states, state_dim] or flattened
            actions: noisy action sequence [batch, num_actions, action_dim]
            timestep: diffusion timestep [batch]
            image: RGB image [batch, 3, 224, 224] (optional)
        
        Returns:
            predicted noise/action [batch, num_actions, action_dim]
        """
        return self.denoise_step(past_states, actions, timestep, image)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export Diffusion Policy model to ONNX')
    parser.add_argument('--state_dim', type=int, default=14,
                        help='Robot state dimension (default: 14 for 7-DOF arm with pos+vel)')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Action dimension (default: 7 for 7-DOF arm)')
    parser.add_argument('--num_actions', type=int, default=16,
                        help='Number of actions in sequence (default: 16)')
    parser.add_argument('--num_past_states', type=int, default=4,
                        help='Number of past states to use (default: 4)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension (default: 256, comparable to depth networks)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    print(f"Creating Diffusion Policy model...")
    print(f"  Past states: {args.num_past_states} states, {args.state_dim} dimensions each")
    print(f"  Actions: {args.num_actions} actions, {args.action_dim} dimensions each")
    print(f"  Hidden dimension: {args.hidden_dim} (substantial architecture)")
    
    model = DiffusionPolicy(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        num_actions=args.num_actions,
        num_past_states=args.num_past_states
    )
    
    # Initialize with random weights
    print("Initializing model with random weights...")
    model.apply(lambda m: _init_weights_random(m) if isinstance(m, (nn.Linear, nn.Conv2d)) else None)
    
    # Replace AdaptivePool2d with ONNX-compatible versions
    print("Replacing AdaptivePool2d with ONNX-compatible operations...")
    replace_adaptive_pooling_with_onnx_compatible(model)
    
    model.eval()
    model.to(device)
    
    # Create dummy inputs
    # Past states can be provided as [batch, num_past_states, state_dim] or flattened
    dummy_past_states = torch.randn(1, args.num_past_states, args.state_dim).to(device)
    dummy_actions = torch.randn(1, args.num_actions, args.action_dim).to(device)
    dummy_timestep = torch.randint(0, 1000, (1,)).to(device)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)  # RGB 224x224
    
    # Test forward pass (single denoising step)
    print("Testing forward pass...")
    try:
        with torch.no_grad():
            test_output = model(dummy_past_states, dummy_actions, dummy_timestep, dummy_image)
        print(f"Forward pass successful! Output shape: {test_output.shape}")
    except Exception as e:
        print(f"ERROR: Forward pass failed with: {e}")
        raise
    
    # Export to ONNX (always use fixed filename)
    onnx_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion.onnx")
    print(f"Exporting ONNX model to {onnx_path}...")
    print(f"Input shapes:")
    print(f"  Past states: {dummy_past_states.shape}")
    print(f"  Actions: {dummy_actions.shape}")
    print(f"  Timestep: {dummy_timestep.shape}")
    print(f"  Image: {dummy_image.shape}")
    
    torch.onnx.export(
        model,
        (dummy_past_states, dummy_actions, dummy_timestep, dummy_image),
        onnx_path,
        input_names=['past_states', 'actions', 'timestep', 'image'],
        output_names=['predicted_actions'],
        opset_version=18,
    )
    
    print(f"Successfully exported ONNX model to {onnx_path}")

