"""Optimization loop for neural style transfer."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from neural_style_transfer.config import StyleTransferConfig
from neural_style_transfer.losses import (
    content_loss,
    gram_matrix,
    style_loss,
    total_variation_loss,
)
from neural_style_transfer.model import VGG19FeatureExtractor


@dataclass(frozen=True, slots=True)
class LossSnapshot:
    step: int
    total: float
    content: float
    style: float
    total_variation: float


@dataclass(frozen=True, slots=True)
class StyleTransferResult:
    image: Tensor
    history: tuple[LossSnapshot, ...]
    device: str


ProgressCallback = Callable[[LossSnapshot, Tensor], None]


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic accelerator choice."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return device


def _validate_image_tensor(image: Tensor, name: str) -> None:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(f"{name} must have shape (1, 3, height, width)")
    if image.shape[2] < 2 or image.shape[3] < 2:
        raise ValueError(f"{name} must be at least 2 by 2 pixels")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} must contain only finite values")


def _style_targets(
    features: Mapping[str, Tensor], layers: tuple[tuple[str, float], ...]
) -> dict[str, Tensor]:
    return {name: gram_matrix(features[name]).detach() for name, _ in layers}


def run_style_transfer(
    content_image: Tensor,
    style_image: Tensor,
    config: StyleTransferConfig | None = None,
    *,
    feature_extractor: nn.Module | None = None,
    callback: ProgressCallback | None = None,
) -> StyleTransferResult:
    """Optimize an image that matches content structure and style statistics."""

    config = config or StyleTransferConfig()
    _validate_image_tensor(content_image, "content_image")
    _validate_image_tensor(style_image, "style_image")

    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    requested_layers = {config.content_layer, *(name for name, _ in config.style_layers)}
    extractor = feature_extractor or VGG19FeatureExtractor(requested_layers)
    extractor = extractor.to(device).eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)

    content = content_image.to(device=device, dtype=torch.float32).clamp(0, 1)
    style = style_image.to(device=device, dtype=torch.float32).clamp(0, 1)

    with torch.no_grad():
        content_features = extractor(content)
        style_features = extractor(style)
        if config.content_layer not in content_features:
            raise KeyError(f"extractor did not return content layer: {config.content_layer}")
        content_target = content_features[config.content_layer].detach()
        style_targets = _style_targets(style_features, config.style_layers)

    initial = content.clone() if config.initialization == "content" else torch.rand_like(content)

    generated = nn.Parameter(initial)
    optimizer = torch.optim.Adam([generated], lr=config.learning_rate)
    history: list[LossSnapshot] = []

    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        generated_features = extractor(generated)

        current_content_loss = content_loss(
            generated_features[config.content_layer], content_target
        )
        current_style_loss = style_loss(generated_features, style_targets, config.style_layers)
        current_tv_loss = total_variation_loss(generated)
        total_loss = (
            config.content_weight * current_content_loss
            + config.style_weight * current_style_loss
            + config.total_variation_weight * current_tv_loss
        )
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            generated.clamp_(0, 1)

        should_report = step == 1 or step == config.steps or step % config.progress_interval == 0
        if should_report:
            with torch.no_grad():
                reported_features = extractor(generated)
                reported_content_loss = content_loss(
                    reported_features[config.content_layer], content_target
                )
                reported_style_loss = style_loss(
                    reported_features, style_targets, config.style_layers
                )
                reported_tv_loss = total_variation_loss(generated)
                reported_total_loss = (
                    config.content_weight * reported_content_loss
                    + config.style_weight * reported_style_loss
                    + config.total_variation_weight * reported_tv_loss
                )
            snapshot = LossSnapshot(
                step=step,
                total=float(reported_total_loss.cpu()),
                content=float(reported_content_loss.cpu()),
                style=float(reported_style_loss.cpu()),
                total_variation=float(reported_tv_loss.cpu()),
            )
            history.append(snapshot)
            if callback is not None:
                callback(snapshot, generated.detach().cpu().clone())

    return StyleTransferResult(
        image=generated.detach().cpu(), history=tuple(history), device=str(device)
    )
