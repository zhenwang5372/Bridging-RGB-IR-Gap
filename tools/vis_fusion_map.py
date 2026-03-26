#!/usr/bin/env python3
# Copyright (c) 2025. All rights reserved.
"""
Visualization script for LightweightCrossFusion module.

Visualizes the RGB-IR fusion attention mechanism without modifying source code.

Usage:
    python tools/vis_fusion_map.py \
        --config configs/custom_flir/yolow_v2_rgb_ir_flir.py \
        --checkpoint work_dir/rgb_ir_filr/best_coco_bbox_mAP_50_epoch_183.pth \
        --img-rgb /path/to/rgb_image.jpg \
        --img-ir /path/to/ir_image.jpeg \
        --output-dir vis_results/fusion_maps
"""

import argparse
import os
import os.path as osp
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# MMEngine and MMYOLO imports
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import init_detector


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize LightweightCrossFusion attention mechanism')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--img-rgb', type=str, required=True,
                        help='Path to RGB image')
    parser.add_argument('--img-ir', type=str, required=True,
                        help='Path to IR image')
    parser.add_argument('--output-dir', type=str, default='vis_results/fusion_maps',
                        help='Directory to save visualization results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run inference (default: cuda:0)')
    parser.add_argument('--level', type=int, default=1,
                        choices=[0, 1, 2],
                        help='Pyramid level to visualize: 0=P3, 1=P4, 2=P5 (default: 1)')
    parser.add_argument('--colormap', type=str, default='jet',
                        choices=['jet', 'viridis', 'hot', 'inferno', 'magma'],
                        help='Colormap for heatmaps (default: jet)')
    return parser.parse_args()


def register_modules():
    """Register all YOLO-World modules."""
    from mmyolo.utils import register_all_modules
    register_all_modules(init_default_scope=False)
    
    # Import yolo_world to register custom modules
    import yolo_world  # noqa: F401


def load_and_preprocess_image(img_path: str, 
                               target_size: Tuple[int, int] = (640, 640),
                               mean: List[float] = [0., 0., 0.],
                               std: List[float] = [255., 255., 255.],
                               bgr_to_rgb: bool = True) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess an image according to the config.
    
    Args:
        img_path: Path to image file
        target_size: Target size (H, W)
        mean: Normalization mean
        std: Normalization std
        bgr_to_rgb: Whether to convert BGR to RGB
        
    Returns:
        Tuple of (preprocessed_tensor [1, 3, H, W], original_image [H, W, 3])
    """
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    original_img = img.copy()
    
    # Convert BGR to RGB if needed
    if bgr_to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Letter resize (maintain aspect ratio with padding)
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scale
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Pad to target size
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Convert to tensor [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_padded).float().permute(2, 0, 1)
    
    # Normalize
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    img_tensor = (img_tensor - mean_tensor) / std_tensor
    
    # Add batch dimension [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, original_img


def get_fusion_modules(model) -> List:
    """
    Extract LightweightCrossFusion modules from the model.
    
    Returns:
        List of fusion modules [P3, P4, P5]
    """
    # Model structure: model.backbone.fusion_module.fusion_modules
    backbone = model.backbone
    
    if hasattr(backbone, 'fusion_module'):
        fusion_module = backbone.fusion_module
        if hasattr(fusion_module, 'fusion_modules'):
            return list(fusion_module.fusion_modules)
    
    raise ValueError("Could not find fusion modules in model. "
                    "Expected: model.backbone.fusion_module.fusion_modules")


def get_backbone_features(model, img_rgb: torch.Tensor, img_ir: torch.Tensor):
    """
    Extract RGB and IR features from backbone before fusion.
    
    Args:
        model: The detector model
        img_rgb: RGB input tensor [B, C, H, W]
        img_ir: IR input tensor [B, C, H, W]
        
    Returns:
        Tuple of (rgb_feats, ir_feats) - each is a tuple of tensors (P3, P4, P5)
    """
    backbone = model.backbone
    
    # Extract RGB features using image_model
    rgb_feats = backbone.image_model(img_rgb)
    
    # Extract IR features using ir_model
    ir_feats = backbone.ir_model(img_ir)
    
    return rgb_feats, ir_feats


def manual_fusion_forward(fusion_module, x_rgb: torch.Tensor, x_ir: torch.Tensor):
    """
    Manually execute fusion forward pass to extract intermediate results.
    
    This replicates LightweightCrossFusion.forward() but returns intermediates.
    
    Args:
        fusion_module: LightweightCrossFusion instance
        x_rgb: RGB features [B, C_rgb, H, W]
        x_ir: IR features [B, C_ir, H, W]
        
    Returns:
        dict with keys:
            - 'ir_aligned': IR features after channel alignment
            - 'attention_map': Attention map from IR
            - 'rgb_attended': RGB features after attention multiplication
            - 'output': Final fused output
    """
    results = {}
    
    # Step 1: Align IR channels to RGB
    x_ir_aligned = fusion_module.ir_align(x_ir)
    
    # Resize IR features if spatial dimensions don't match
    if x_ir_aligned.shape[-2:] != x_rgb.shape[-2:]:
        x_ir_aligned = F.interpolate(
            x_ir_aligned, size=x_rgb.shape[-2:],
            mode='bilinear', align_corners=False
        )
    results['ir_aligned'] = x_ir_aligned.detach()
    
    # Step 2: Generate spatial attention from IR features
    attention_map = fusion_module.attention_gen(x_ir_aligned)
    results['attention_map'] = attention_map.detach()
    
    # Step 3: Apply attention to RGB features (element-wise multiplication)
    x_rgb_attended = x_rgb * attention_map
    results['rgb_attended'] = x_rgb_attended.detach()
    
    # Step 4: Cross-modality fusion
    combined = torch.cat([x_rgb_attended, x_ir_aligned], dim=1)
    fused = fusion_module.cross_conv(combined)
    
    # Step 5: Residual connection with learnable weight
    output = x_rgb + fusion_module.gamma * fused
    results['output'] = output.detach()
    
    return results


def tensor_to_heatmap(tensor: torch.Tensor, colormap: str = 'jet') -> np.ndarray:
    """
    Convert a feature tensor to a heatmap image.
    
    Args:
        tensor: Feature tensor [B, C, H, W] or [C, H, W] or [H, W]
        colormap: Colormap name ('jet', 'viridis', 'hot', etc.)
        
    Returns:
        Heatmap image [H, W, 3] in BGR format
    """
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0]  # [C, H, W]
    
    # Average over channels if multi-channel
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=0)  # [H, W]
    
    # Move to CPU and convert to numpy
    heatmap = tensor.cpu().numpy()
    
    # Normalize to [0, 255]
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Apply colormap
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'viridis': cv2.COLORMAP_VIRIDIS,
        'hot': cv2.COLORMAP_HOT,
        'inferno': cv2.COLORMAP_INFERNO,
        'magma': cv2.COLORMAP_MAGMA,
    }
    cv_colormap = colormap_dict.get(colormap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(heatmap, cv_colormap)
    
    return heatmap_colored


def resize_heatmap_to_image(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize heatmap to match target image size."""
    return cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)


def create_visualization(
    original_rgb: np.ndarray,
    original_ir: np.ndarray,
    rgb_feat_heatmap: np.ndarray,
    attention_heatmap: np.ndarray,
    multiply_heatmap: np.ndarray,
    output_heatmap: np.ndarray,
    level_name: str,
    output_path: str,
    figsize: Tuple[int, int] = (20, 10)
):
    """
    Create and save the visualization figure.
    
    Args:
        original_rgb: Original RGB image [H, W, 3] in BGR
        original_ir: Original IR image [H, W, 3] in BGR
        rgb_feat_heatmap: RGB feature heatmap [H, W, 3] in BGR
        attention_heatmap: Attention map heatmap [H, W, 3] in BGR
        multiply_heatmap: Multiply result heatmap [H, W, 3] in BGR
        output_heatmap: Final output heatmap [H, W, 3] in BGR
        level_name: Pyramid level name (e.g., "P4")
        output_path: Path to save the figure
        figsize: Figure size
    """
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
    original_ir_rgb = cv2.cvtColor(original_ir, cv2.COLOR_BGR2RGB)
    rgb_feat_heatmap = cv2.cvtColor(rgb_feat_heatmap, cv2.COLOR_BGR2RGB)
    attention_heatmap = cv2.cvtColor(attention_heatmap, cv2.COLOR_BGR2RGB)
    multiply_heatmap = cv2.cvtColor(multiply_heatmap, cv2.COLOR_BGR2RGB)
    output_heatmap = cv2.cvtColor(output_heatmap, cv2.COLOR_BGR2RGB)
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 4, figure=fig, wspace=0.05, hspace=0.15)
    
    # Title for the whole figure
    fig.suptitle(f'LightweightCrossFusion Visualization @ {level_name}', 
                 fontsize=16, fontweight='bold')
    
    # Row 1: Original images and inputs
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original_rgb)
    ax1.set_title('Original RGB Image', fontsize=12)
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(original_ir_rgb)
    ax2.set_title('Original IR Image', fontsize=12)
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(rgb_feat_heatmap)
    ax3.set_title('RGB Feature (Input)', fontsize=12)
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(attention_heatmap)
    ax4.set_title('IR → Attention Map', fontsize=12, color='red')
    ax4.axis('off')
    
    # Row 2: Fusion process and output
    ax5 = fig.add_subplot(gs[1, 0:2])
    ax5.imshow(multiply_heatmap)
    ax5.set_title('RGB × Attention (Element-wise Multiply)', fontsize=12, color='blue')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.imshow(output_heatmap)
    ax6.set_title('Fused Output (Final)', fontsize=12, color='green')
    ax6.axis('off')
    
    # Add annotation explaining the fusion mechanism
    fig.text(0.5, 0.02, 
             'Fusion Mechanism: output = RGB + γ × CrossConv(RGB×Attention ∥ IR_aligned)\n'
             'Where Attention = Sigmoid(Conv(Conv(IR_aligned)))',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved visualization to: {output_path}")


def create_attention_overlay(
    original_rgb: np.ndarray,
    attention_heatmap: np.ndarray,
    output_path: str,
    alpha: float = 0.5
):
    """
    Create an overlay of attention map on original RGB image.
    
    Args:
        original_rgb: Original RGB image [H, W, 3] in BGR
        attention_heatmap: Attention heatmap [H, W, 3] in BGR (will be resized)
        output_path: Path to save the overlay
        alpha: Blending alpha for overlay
    """
    h, w = original_rgb.shape[:2]
    attention_resized = cv2.resize(attention_heatmap, (w, h))
    
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, attention_resized, alpha, 0)
    
    cv2.imwrite(output_path, overlay)
    print(f"✅ Saved attention overlay to: {output_path}")


def main():
    args = parse_args()
    
    # Register all modules
    print("📦 Registering YOLO-World modules...")
    register_modules()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and model
    print(f"📄 Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    
    print(f"🔧 Loading model from: {args.checkpoint}")
    model = init_detector(cfg, args.checkpoint, device=args.device)
    model.eval()
    
    # Get fusion modules
    fusion_modules = get_fusion_modules(model)
    print(f"✅ Found {len(fusion_modules)} fusion modules (P3, P4, P5)")
    
    # Load and preprocess images
    print(f"🖼️  Loading RGB image: {args.img_rgb}")
    print(f"🌡️  Loading IR image: {args.img_ir}")
    
    # Get preprocessing params from config
    data_preprocessor = cfg.model.get('data_preprocessor', {})
    mean = data_preprocessor.get('mean', [0., 0., 0.])
    std = data_preprocessor.get('std', [255., 255., 255.])
    mean_ir = data_preprocessor.get('mean_ir', mean)
    std_ir = data_preprocessor.get('std_ir', std)
    bgr_to_rgb = data_preprocessor.get('bgr_to_rgb', True)
    
    img_scale = cfg.get('img_scale', (640, 640))
    
    img_rgb_tensor, original_rgb = load_and_preprocess_image(
        args.img_rgb, target_size=img_scale, 
        mean=mean, std=std, bgr_to_rgb=bgr_to_rgb
    )
    img_ir_tensor, original_ir = load_and_preprocess_image(
        args.img_ir, target_size=img_scale,
        mean=mean_ir, std=std_ir, bgr_to_rgb=bgr_to_rgb
    )
    
    # Move to device
    img_rgb_tensor = img_rgb_tensor.to(args.device)
    img_ir_tensor = img_ir_tensor.to(args.device)
    
    print(f"📐 RGB tensor shape: {img_rgb_tensor.shape}")
    print(f"📐 IR tensor shape: {img_ir_tensor.shape}")
    
    # Extract backbone features (before fusion)
    print("🔍 Extracting backbone features...")
    with torch.no_grad():
        rgb_feats, ir_feats = get_backbone_features(model, img_rgb_tensor, img_ir_tensor)
    
    level_names = ['P3', 'P4', 'P5']
    print(f"   RGB features: {[f.shape for f in rgb_feats]}")
    print(f"   IR features: {[f.shape for f in ir_feats]}")
    
    # Visualize specified level
    level = args.level
    level_name = level_names[level]
    fusion_module = fusion_modules[level]
    rgb_feat = rgb_feats[level]
    ir_feat = ir_feats[level]
    
    print(f"\n🎯 Visualizing fusion at {level_name}...")
    print(f"   RGB feature shape: {rgb_feat.shape}")
    print(f"   IR feature shape: {ir_feat.shape}")
    
    # Manually execute fusion forward to get intermediates
    with torch.no_grad():
        fusion_results = manual_fusion_forward(fusion_module, rgb_feat, ir_feat)
    
    print("   ✓ IR aligned shape:", fusion_results['ir_aligned'].shape)
    print("   ✓ Attention map shape:", fusion_results['attention_map'].shape)
    print("   ✓ RGB attended shape:", fusion_results['rgb_attended'].shape)
    print("   ✓ Output shape:", fusion_results['output'].shape)
    
    # Print gamma value (learnable residual weight)
    gamma_val = fusion_module.gamma.item()
    print(f"   γ (gamma) value: {gamma_val:.4f}")
    
    # Convert to heatmaps
    print(f"\n🎨 Generating heatmaps with colormap: {args.colormap}")
    rgb_feat_heatmap = tensor_to_heatmap(rgb_feat, args.colormap)
    attention_heatmap = tensor_to_heatmap(fusion_results['attention_map'], args.colormap)
    multiply_heatmap = tensor_to_heatmap(fusion_results['rgb_attended'], args.colormap)
    output_heatmap = tensor_to_heatmap(fusion_results['output'], args.colormap)
    
    # Resize heatmaps to original image size for better visualization
    h, w = original_rgb.shape[:2]
    target_size = (w, h)
    rgb_feat_heatmap = resize_heatmap_to_image(rgb_feat_heatmap, target_size)
    attention_heatmap = resize_heatmap_to_image(attention_heatmap, target_size)
    multiply_heatmap = resize_heatmap_to_image(multiply_heatmap, target_size)
    output_heatmap = resize_heatmap_to_image(output_heatmap, target_size)
    
    # Get base filename
    rgb_basename = osp.splitext(osp.basename(args.img_rgb))[0]
    
    # Create and save visualization
    output_path = osp.join(args.output_dir, f'{rgb_basename}_fusion_{level_name}.png')
    create_visualization(
        original_rgb, original_ir,
        rgb_feat_heatmap, attention_heatmap,
        multiply_heatmap, output_heatmap,
        level_name, output_path
    )
    
    # Create attention overlay on original image
    overlay_path = osp.join(args.output_dir, f'{rgb_basename}_attention_overlay_{level_name}.png')
    create_attention_overlay(original_rgb, attention_heatmap, overlay_path)
    
    # Optionally visualize all levels with original images on the left
    print("\n📊 Generating all-levels comparison...")
    
    # Create figure: 3 rows (P3, P4, P5) x 6 columns (RGB, IR, RGB_feat, Attention, Multiply, Output)
    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(3, 6, figure=fig, wspace=0.08, hspace=0.15,
                  width_ratios=[1, 1, 1, 1, 1, 1])
    
    fig.suptitle('LightweightCrossFusion - All Pyramid Levels\n'
                 'Left: Original Images | Right: Feature Maps & Fusion Process', 
                 fontsize=14, fontweight='bold')
    
    # Convert original images to RGB for matplotlib (do once)
    original_rgb_plt = cv2.cvtColor(original_rgb, cv2.COLOR_BGR2RGB)
    original_ir_plt = cv2.cvtColor(original_ir, cv2.COLOR_BGR2RGB)
    
    for i, (lvl_name, fusion_mod, rgb_f, ir_f) in enumerate(
            zip(level_names, fusion_modules, rgb_feats, ir_feats)):
        
        with torch.no_grad():
            results = manual_fusion_forward(fusion_mod, rgb_f, ir_f)
        
        # Generate heatmaps
        rgb_hm = tensor_to_heatmap(rgb_f, args.colormap)
        att_hm = tensor_to_heatmap(results['attention_map'], args.colormap)
        mul_hm = tensor_to_heatmap(results['rgb_attended'], args.colormap)
        out_hm = tensor_to_heatmap(results['output'], args.colormap)
        
        # Convert BGR to RGB for matplotlib
        rgb_hm = cv2.cvtColor(rgb_hm, cv2.COLOR_BGR2RGB)
        att_hm = cv2.cvtColor(att_hm, cv2.COLOR_BGR2RGB)
        mul_hm = cv2.cvtColor(mul_hm, cv2.COLOR_BGR2RGB)
        out_hm = cv2.cvtColor(out_hm, cv2.COLOR_BGR2RGB)
        
        # Column 0: Original RGB image
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(original_rgb_plt)
        if i == 0:
            ax0.set_title('Original RGB', fontsize=11, fontweight='bold')
        ax0.set_ylabel(f'{lvl_name}', fontsize=12, fontweight='bold', rotation=0, 
                      labelpad=30, va='center')
        ax0.axis('off')
        
        # Column 1: Original IR image
        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(original_ir_plt)
        if i == 0:
            ax1.set_title('Original IR', fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # Column 2: RGB Feature heatmap
        ax2 = fig.add_subplot(gs[i, 2])
        ax2.imshow(rgb_hm)
        if i == 0:
            ax2.set_title('RGB Feature', fontsize=11)
        ax2.axis('off')
        
        # Column 3: Attention map from IR
        ax3 = fig.add_subplot(gs[i, 3])
        ax3.imshow(att_hm)
        if i == 0:
            ax3.set_title('IR→Attention', fontsize=11, color='red')
        ax3.axis('off')
        
        # Column 4: RGB × Attention
        ax4 = fig.add_subplot(gs[i, 4])
        ax4.imshow(mul_hm)
        if i == 0:
            ax4.set_title('RGB×Attention', fontsize=11, color='blue')
        ax4.axis('off')
        
        # Column 5: Fused Output
        ax5 = fig.add_subplot(gs[i, 5])
        ax5.imshow(out_hm)
        if i == 0:
            ax5.set_title('Fused Output', fontsize=11, color='green')
        ax5.axis('off')
    
    # Add annotation at bottom
    fig.text(0.5, 0.02, 
             'Fusion: output = RGB + γ × CrossConv(RGB×Attention ∥ IR_aligned), '
             f'γ = {fusion_modules[0].gamma.item():.4f}',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    all_levels_path = osp.join(args.output_dir, f'{rgb_basename}_all_levels.png')
    plt.savefig(all_levels_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"✅ Saved all-levels comparison to: {all_levels_path}")
    
    print("\n🎉 Visualization complete!")
    print(f"📁 Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

