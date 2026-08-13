"""Image loading, conversion, saving, and color-preservation helpers."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor


def resize_long_edge(image: Image.Image, max_size: int) -> Image.Image:
    """Shrink an image so its longest edge is at most ``max_size``."""

    if max_size < 1:
        raise ValueError("max_size must be positive")
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_size:
        return image.copy()
    scale = max_size / long_edge
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def pil_to_tensor(image: Image.Image) -> Tensor:
    """Convert a PIL image into a float tensor in the [0, 1] range."""

    rgb = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(rgb, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def tensor_to_pil(tensor: Tensor) -> Image.Image:
    """Convert a 3D tensor or single-image batch into an RGB PIL image."""

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("a batched tensor must contain exactly one image")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 3:
        raise ValueError("tensor must have shape (3, height, width)")

    array = (
        tensor.detach()
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def load_image(path: str | Path, max_size: int) -> Tensor:
    """Load an image from disk, normalize orientation, and resize it."""

    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    with Image.open(image_path) as image:
        return pil_to_tensor(resize_long_edge(image.convert("RGB"), max_size))


def save_image(tensor: Tensor, path: str | Path) -> Path:
    """Save a tensor as an image, creating its parent directory if needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(output_path)
    return output_path


def preserve_content_colors(generated: Image.Image, content: Image.Image) -> Image.Image:
    """Use generated luminance while preserving the content image's chroma."""

    generated_rgb = generated.convert("RGB")
    content_rgb = ImageOps.fit(
        content.convert("RGB"), generated_rgb.size, method=Image.Resampling.LANCZOS
    )
    generated_y, _, _ = generated_rgb.convert("YCbCr").split()
    _, content_cb, content_cr = content_rgb.convert("YCbCr").split()
    return Image.merge("YCbCr", (generated_y, content_cb, content_cr)).convert("RGB")
