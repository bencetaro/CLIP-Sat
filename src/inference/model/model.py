import torch
import torch.nn as nn


def get_image_features_safe(clip_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Mirror the training notebook's safe feature extraction."""
    try:
        features = clip_model.get_image_features(pixel_values=pixel_values)
    except Exception:
        outputs = clip_model.vision_model(pixel_values=pixel_values)
        features = outputs.pooler_output

    if not isinstance(features, torch.Tensor):
        features = features.pooler_output

    return features

class CLIPClassifier(nn.Module):
    """Training-notebook compatible CLIP image classifier head."""

    def __init__(self, clip_model, num_classes: int):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(clip_model.config.projection_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_features = get_image_features_safe(self.clip, pixel_values)
        return self.classifier(image_features)
