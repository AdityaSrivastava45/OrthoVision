"""DINOv2 encoder wrapper for MRI 2.5D inputs.

DINOv2 normally expects 3-channel RGB input. Our 2.5D MRI representation
already has exactly three channels:
- channel 0 = previous/neighboring MRI slice
- channel 1 = center MRI slice
- channel 2 = next/neighboring MRI slice

We do NOT convert to grayscale or duplicate channels. The 3-channel
MRI input enters DINOv2 directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger("orthovision.models.dinov2_encoder")


@dataclass
class DINOv2Config:
    """Configuration for DINOv2 encoder."""
    variant: str = "dinov2_vits14"  # dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
    pretrained_path: str | None = None  # Path to local pretrained weights
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0  # Number of last blocks to unfreeze (0 = all frozen)
    img_size: int = 224
    patch_size: int = 14
    in_chans: int = 3
    embed_dim: int = 384  # ViT-Small
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.0


def _load_state_dict_from_path(path: str | Path) -> dict[str, Any]:
    """Load state dict from various formats (pytorch, safetensors, etc.).

    Uses safe loading to avoid arbitrary code execution.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {path}")

    if path.suffix in (".pt", ".pth"):
        # Use weights_only=True for safe loading (PyTorch 2.0+)
        return torch.load(path, map_location="cpu", weights_only=True)
    elif path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
            return load_file(str(path))
        except ImportError:
            raise ImportError("safetensors package required for .safetensors files")
    else:
        raise ValueError(f"Unsupported weight file format: {path.suffix}")


def _build_dinov2_vits14(config: DINOv2Config) -> nn.Module:
    """Build DINOv2 ViT-Small/14 model architecture."""
    # ViT-Small configuration from DINOv2
    # https://github.com/facebookresearch/dinov2
    from torchvision.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        image_size=config.img_size,
        patch_size=config.patch_size,
        num_layers=config.depth,
        num_heads=config.num_heads,
        hidden_dim=config.embed_dim,
        mlp_dim=int(config.embed_dim * config.mlp_ratio),
        dropout=0.0,
        attention_dropout=0.0,
        num_classes=0,  # No classification head - we use embeddings
        representation_size=None,
        norm_layer=nn.LayerNorm,
        conv_stem_configs=None,
    )

    # Modify first conv layer for in_chans if not 3
    if config.in_chans != 3:
        original_conv = model.conv_proj
        model.conv_proj = nn.Conv2d(
            config.in_chans,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )
        # Initialize new conv weights by averaging original RGB weights
        with torch.no_grad():
            model.conv_proj.weight[:] = original_conv.weight.mean(dim=1, keepdim=True).repeat(1, config.in_chans, 1, 1)

    # Wrap the forward to return class token embeddings
    original_forward = model.forward
    
    def new_forward(x: torch.Tensor) -> torch.Tensor:
        # Process through conv_proj to get patch embeddings
        n, c, h, w = x.shape
        x = model.conv_proj(x)  # [B, hidden_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_dim]
        
        # Add class token
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Add positional embeddings
        x = x + model.encoder.pos_embedding
        x = model.encoder.dropout(x)
        
        # Pass through encoder layers
        x = model.encoder.layers(x)
        x = model.encoder.ln(x)
        
        # Return class token (first token) as embedding
        return x[:, 0]  # [B, hidden_dim]
    
    model.forward = new_forward
    
    return model


def create_dinov2_encoder(config: DINOv2Config) -> nn.Module:
    """Create DINOv2 encoder with optional pretrained weights."""
    if config.variant == "dinov2_vits14":
        model = _build_dinov2_vits14(config)
    else:
        raise ValueError(f"Unsupported DINOv2 variant: {config.variant}")

    # Load pretrained weights if provided
    if config.pretrained_path:
        log.info(f"Loading pretrained weights from {config.pretrained_path}")
        state_dict = _load_state_dict_from_path(config.pretrained_path)

        # Handle different state dict formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove 'module.' prefix if present (from DDP)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Handle missing keys (e.g., first conv layer if in_chans != 3)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if missing_keys:
            log.warning(f"Missing keys when loading pretrained weights: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")
        log.info("Pretrained weights loaded successfully")
    else:
        log.warning("No pretrained weights provided - using random initialization")

    return model


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder for MRI 2.5D inputs.

    Input: [B, 3, H, W] - batch of 2.5D MRI samples
    Output: [B, embed_dim] - embeddings for each sample
    """

    def __init__(self, config: DINOv2Config):
        super().__init__()
        self.config = config
        self.backbone = create_dinov2_encoder(config)
        self.embed_dim = config.embed_dim

        # Apply freezing
        self._apply_freezing()

    def _apply_freezing(self) -> None:
        """Apply freezing/unfreezing based on config."""
        if self.config.freeze_backbone:
            # Freeze all parameters first
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Unfreeze last N blocks if specified
            if self.config.unfreeze_last_n_blocks > 0:
                # For VisionTransformer, blocks are in encoder.layers
                if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layers"):
                    layers = list(self.backbone.encoder.layers)
                    for layer in layers[-self.config.unfreeze_last_n_blocks:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    log.info(f"Unfroze last {self.config.unfreeze_last_n_blocks} transformer blocks")
                else:
                    log.warning("Could not find encoder.layers to unfreeze blocks")

            log.info("DINOv2 backbone frozen")
        else:
            log.info("DINOv2 backbone fully trainable")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: [B, 3, H, W] - batch of 2.5D MRI samples

        Returns:
            [B, embed_dim] - embeddings
        """
        return self.backbone(x)

    def train(self, mode: bool = True) -> "DINOv2Encoder":
        """Override train to respect frozen parameters."""
        super().train(mode)
        if self.config.freeze_backbone:
            # Keep frozen parts in eval mode
            self.backbone.eval()
        return self