"""Differentiable losses used by neural style transfer."""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F


def gram_matrix(features: Tensor) -> Tensor:
    """Return a channel-correlation matrix normalized by feature-map size."""

    if features.ndim != 4:
        raise ValueError("features must have shape (batch, channels, height, width)")

    batch, channels, height, width = features.shape
    flattened = features.reshape(batch, channels, height * width)
    gram = torch.bmm(flattened, flattened.transpose(1, 2))
    return gram / (channels * height * width)


def content_loss(generated: Tensor, target: Tensor) -> Tensor:
    """Mean-squared distance between generated and target content features."""

    return F.mse_loss(generated, target)


def style_loss(
    generated_features: Mapping[str, Tensor],
    style_targets: Mapping[str, Tensor],
    layer_weights: Sequence[tuple[str, float]],
) -> Tensor:
    """Weighted Gram-matrix loss over multiple feature layers."""

    if not generated_features:
        raise ValueError("generated_features cannot be empty")

    reference = next(iter(generated_features.values()))
    loss = reference.new_zeros(())
    total_weight = sum(weight for _, weight in layer_weights)
    if total_weight <= 0:
        raise ValueError("layer weights must have a positive sum")

    for layer_name, weight in layer_weights:
        if layer_name not in generated_features or layer_name not in style_targets:
            raise KeyError(f"missing style features for layer: {layer_name}")
        loss = loss + (weight / total_weight) * F.mse_loss(
            gram_matrix(generated_features[layer_name]), style_targets[layer_name]
        )
    return loss


def total_variation_loss(image: Tensor) -> Tensor:
    """Penalize abrupt changes between neighboring pixels."""

    if image.ndim != 4:
        raise ValueError("image must have shape (batch, channels, height, width)")
    horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    return horizontal + vertical
