"""
models/resnet50_da.py

ResNet-50 backbone with a domain-adaptation head.

forward() returns BOTH logits and features so that:
  - logits  → used for loss computation
  - features → used by the active sampler (diversity / uncertainty strategies)
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50DA(nn.Module):
    def __init__(self, num_classes: int = 12, bottleneck_dim: int = 256):
        super().__init__()

        # ── Backbone (ImageNet pretrained) ────────────────────────────────────
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.feature_dim = backbone.fc.in_features          # 2048

        # Remove the original classifier
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])   # → (B, 2048, 1, 1)

        # ── Bottleneck (optional but common in DA literature) ─────────────────
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(inplace=True),
        )

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor):
        """
        Returns:
            logits   (B, num_classes)   – use for loss / prediction
            features (B, bottleneck_dim) – use for sampler / visualisation
        """
        feat = self.backbone(x)                # (B, 2048, 1, 1)
        feat = feat.view(feat.size(0), -1)     # (B, 2048)
        feat = self.bottleneck(feat)           # (B, 256)
        logits = self.classifier(feat)         # (B, num_classes)
        return logits, feat

    # ── Convenience: features only (used by sampler) ──────────────────────────
    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns bottleneck features without computing gradients."""
        self.eval()
        _, feat = self.forward(x)
        return feat